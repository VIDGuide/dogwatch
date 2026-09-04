#!/usr/bin/env python3
"""dogwatch_check.py — periodic event verifier (vision model + Telegram + siren).

Reads the notifier's append-only event log, dedupes each incident, verifies
every event against a vision model, reports confirm / false-alarm / uncertain
to Telegram, and — only when the model confirms *digging* — fires the optional
Home Assistant siren via ``dog-alarm.sh``.

**This used to be ~590 lines of Python inside a quoted heredoc in
``dogwatch-check.sh``.** That made it invisible to ``py_compile`` and to
pytest (CI could only run ``bash -n`` on the wrapper), which left the
watermark, dedupe and verification logic — the highest-risk code in the
repo — completely untested. It is now an ordinary importable module; the
shell script is a thin ``flock`` wrapper.

Behavioural fixes made during the extraction (all called out in the assessment):

* **Atomic watermark.** The old plain ``open(w)`` + ``write`` could leave a
  torn value, and the subsequent ``float()`` was outside any try/except — so
  one bad write wedged *every future cycle* with a traceback and alerts
  stopped permanently. Now tmp+fsync+replace on write, and a defensive parse
  on read (see ``CheckState``).
* **Incremental watermark.** It was only advanced after the entire vision
  loop, so any mid-loop crash re-alerted every in-window event (duplicate
  Telegram, duplicate siren attempt). Now advanced per completed event, with
  a ``finally`` to persist progress even on an unhandled exception.
* **No more wall-clock CUTOFF cliff.** The old 7-minute cutoff had to exceed
  the loop period *plus* the cycle duration, but ``alarm_followup`` sleeps 30s
  per confirmed-digging event inside the serial loop, so a busy cycle easily
  ran past it and events were dropped **silently** — reintroducing the exact
  bug the cutoff was raised to fix. The watermark already prevents
  reprocessing, so the age limit's only remaining job is "don't replay ancient
  events after downtime"; it is now generous (30 min) and configurable.
* **Bounded read.** The event log is append-only and never rotated, and every
  cycle re-parsed the whole file just to recompute the high-water mark. The
  state file now also stores a byte offset, so a cycle reads only what's new.
* **Dedupe window anchored.** The window was compared against the *latest*
  entry's ts and then overwritten with it, so a long incident emitting a
  repeat every ~80s slid the 90s window forward indefinitely and collapsed
  into a single reported event. It is now anchored to the first ts.
* **Fresh captures are validated.** ``capture_fresh`` only checked
  ``size > 1000``, so grey/partial-decode frames reached the vision model and
  produced "false alarm" verdicts for real dogs. Now uses the same
  three-layer validation as the notifier (``image_quality``).
* **Model output can't break the message.** Vision-model prose went into
  Telegram with ``parse_mode=Markdown`` unescaped: one unbalanced ``*`` or
  backtick returned HTTP 400, which ``tg_send`` swallowed, so the verdict was
  silently lost. Model text is now sanitised and length-capped, and a failed
  Markdown send is retried as plain text.
* **Credentials stay out of the log.** ffmpeg/requests failures stringify the
  credential-bearing camera URL; all such paths go through ``redact()``.
* **Portable + no injection-shaped shell.** Cutoff arithmetic no longer needs
  GNU ``date -d``, and chat id / bot token resolution no longer interpolates a
  filesystem path into ``python3 -c``.

Configuration is by environment variable so the shell wrapper stays trivial;
see ``Config`` for the full list and defaults.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from dw_redact import redact

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_VISION_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_VISION_MODEL = "qwen/qwen3.7-flash"
DEFAULT_FALLBACK_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_FALLBACK_MODEL = "gemini-3-flash-preview"
DEFAULT_FALLBACK2_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_FALLBACK2_MODEL = "deepseek-v4-flash-vision-exp"

# Telegram's hard caption limit is 1024 chars; keep well clear of it and leave
# room for our own formatting around the model's sentence.
MAX_DESCRIPTION_CHARS = 300
TELEGRAM_CAPTION_LIMIT = 900

# Directories an event's ``snapshot`` field is allowed to point into, and the
# filename prefixes allowed inside them. See ``safe_snapshot_path``.
DEFAULT_SNAPSHOT_ALLOW_DIRS = ("/tmp",)
SNAPSHOT_NAME_PREFIXES = ("dogwatch_snap_", "dogwatch_check_", "dogwatch_")
SNAPSHOT_SUFFIXES = (".jpg", ".jpeg")


def _env_float(env, name, default):
    raw = env.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        print(f"  WARN: {name}={raw!r} is not a number; using {default}",
              file=sys.stderr)
        return default


class Config:
    """Runtime configuration, all overridable by environment variable."""

    def __init__(self, env=None):
        env = env if env is not None else os.environ
        self.status_file = env.get("DOGWATCH_STATUS_FILE", "/tmp/dogwatch-events.jsonl")
        self.state_file = env.get("DOGWATCH_STATE_FILE", "/tmp/dogwatch-check-state.json")
        # Legacy bare-float watermark, read once for a seamless upgrade.
        self.legacy_ts_file = env.get("DOGWATCH_LAST_TS_FILE", "/tmp/dogwatch-last-ts")
        self.workspace_dir = env.get(
            "DOGWATCH_WORKSPACE_DIR",
            os.path.join(os.path.expanduser("~"), ".openclaw/workspace/dogwatch_snaps"),
        )
        # Directories an event's ``snapshot`` path may resolve into. Colon
        # separated, same convention as PATH. The workspace dir is always
        # allowed (that is where we stage copies ourselves).
        raw_allow = env.get("DOGWATCH_SNAPSHOT_ALLOW_DIRS", "")
        allow = ([p for p in raw_allow.split(":") if p]
                 if raw_allow else list(DEFAULT_SNAPSHOT_ALLOW_DIRS))
        allow.append(self.workspace_dir)
        # realpath so the comparison in safe_snapshot_path is symlink-stable
        # (on macOS /tmp is itself a symlink to /private/tmp, for instance).
        self.snapshot_allow_dirs = tuple(
            dict.fromkeys(os.path.realpath(os.path.expanduser(p)) for p in allow))
        self.secrets_file = os.path.expanduser(
            env.get("DOGWATCH_SECRETS_FILE", "~/.openclaw/secrets.json"))
        self.notify_config = env.get(
            "DOGWATCH_NOTIFY_CONFIG",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dogwatch-notify.config.json"),
        )
        self.chat_id = env.get("DOGWATCH_CHAT_ID", "")
        self.bot_token = env.get("DOGWATCH_BOT_TOKEN", "")

        self.vision_url = env.get("DOGWATCH_VISION_API_URL", DEFAULT_VISION_URL)
        self.vision_model = env.get("DOGWATCH_VISION_MODEL", DEFAULT_VISION_MODEL)
        self.vision_key = env.get("DOGWATCH_VISION_API_KEY", "")
        self.fallback_url = env.get("DOGWATCH_VISION_FALLBACK_API_URL", DEFAULT_FALLBACK_URL)
        self.fallback_model = env.get("DOGWATCH_VISION_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        self.fallback_key = env.get("DOGWATCH_VISION_FALLBACK_API_KEY", "")
        self.fallback2_url = env.get("DOGWATCH_VISION_FALLBACK2_API_URL", DEFAULT_FALLBACK2_URL)
        self.fallback2_model = env.get("DOGWATCH_VISION_FALLBACK2_MODEL", DEFAULT_FALLBACK2_MODEL)
        self.fallback2_key = env.get("DOGWATCH_VISION_FALLBACK2_API_KEY", "")

        self.alarm_script = env.get("DOGWATCH_ALARM_SCRIPT", "/app/dog-alarm.sh")
        self.stats_script = env.get("DW_STATS_SCRIPT", "/app/stats.py")

        # Jarvis announce hook (optional): env overrides for the fire-and-
        # forget digging webhook; otherwise read from the notify config's
        # "jarvis_hook" section (url + token). Both empty = feature off.
        self.jarvis_hook_url = env.get("DOGWATCH_JARVIS_HOOK_URL", "")
        self.jarvis_hook_token = env.get("DOGWATCH_JARVIS_HOOK_TOKEN", "")

        self.dedupe_window = _env_float(env, "DOGWATCH_DEDUPE_WINDOW", 90.0)
        # Replaces the old 7-minute wall-clock CUTOFF. The watermark is what
        # prevents reprocessing; this only stops a replay of ancient history
        # after extended downtime, so it can be generous without risk.
        self.max_event_age = _env_float(env, "DOGWATCH_MAX_EVENT_AGE", 1800.0)
        self.followup_delay = _env_float(env, "DOGWATCH_ALARM_FOLLOWUP_SECONDS", 30.0)
        self.capture_timeout = _env_float(env, "DOGWATCH_CAPTURE_TIMEOUT", 15.0)
        self.vision_timeout = _env_float(env, "DOGWATCH_VISION_TIMEOUT", 30.0)


# ---------------------------------------------------------------------------
# Watermark / read-offset state
# ---------------------------------------------------------------------------

class CheckState:
    """Persistent ``{ts, offset}`` watermark with a crash-safe write.

    ``ts``     — newest event timestamp already processed.
    ``offset`` — byte offset already consumed in the event log, so a cycle
                 reads only new lines instead of re-parsing the whole
                 (never-rotated, ever-growing) file.

    Every read path is defensive on purpose: this file is the single point of
    failure that could previously stop all alerting forever. A missing,
    empty, truncated, or garbage file must degrade to "start from zero", never
    to an exception.
    """

    def __init__(self, path, legacy_path=None):
        self.path = path
        self.legacy_path = legacy_path
        self.ts = 0.0
        self.offset = 0

    def load(self):
        data = self._read_json(self.path)
        if data is None:
            # First run under the new format — inherit the legacy bare-float
            # watermark so upgrading doesn't re-alert recent history.
            self.ts = self._read_legacy_ts()
            self.offset = 0
            return self
        self.ts = self._coerce_float(data.get("ts"), 0.0)
        self.offset = int(self._coerce_float(data.get("offset"), 0.0))
        if self.offset < 0:
            self.offset = 0
        return self

    @staticmethod
    def _coerce_float(value, default):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        # NaN/inf would poison every future comparison.
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out

    def _read_json(self, path):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except (OSError, UnicodeDecodeError):
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            print(f"  WARN: unreadable state file {path}; starting from zero",
                  file=sys.stderr)
            return None
        return data if isinstance(data, dict) else None

    def _read_legacy_ts(self):
        if not self.legacy_path:
            return 0.0
        try:
            with open(self.legacy_path) as f:
                return self._coerce_float(f.read().strip(), 0.0)
        except (OSError, UnicodeDecodeError):
            return 0.0

    def advance(self, ts=None, offset=None):
        if ts is not None:
            self.ts = max(self.ts, self._coerce_float(ts, self.ts))
        if offset is not None:
            self.offset = max(0, int(offset))

    def save(self):
        """Atomic tmp + fsync + replace, mirroring stats.py's _save.

        A plain truncate-and-write here could be interrupted, and the old
        reader called ``float()`` on the result outside any try/except — one
        torn write permanently killed the alert pipeline.
        """
        payload = {"ts": round(self.ts, 6), "offset": self.offset}
        tmp = f"{self.path}.tmp"
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            return True
        except OSError as exc:
            print(f"  WARN: could not persist state: {exc}", file=sys.stderr)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False


def read_new_events(status_file, state):
    """Yield ``(event_dict, end_offset)`` for lines after ``state.offset``.

    Resets to offset 0 if the file has shrunk (recreated on container
    restart), so a rotated/replaced log is read from the start rather than
    skipped.
    """
    try:
        size = os.path.getsize(status_file)
    except OSError:
        return
    start = state.offset if state.offset <= size else 0
    if start != state.offset:
        print(f"  event log shrank ({size} < {state.offset}) — re-reading from start",
              file=sys.stderr)
    try:
        # Binary mode + readline(): tell() is disallowed inside a `for line in
        # f` iteration, and byte offsets must be exact for the resume to be
        # correct on non-ASCII content.
        with open(status_file, "rb") as f:
            f.seek(start)
            while True:
                raw = f.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    # Partial trailing line: a concurrent append is mid-write.
                    # Stop without consuming it so the complete line is picked
                    # up next cycle.
                    break
                pos = f.tell()
                text = raw.decode("utf-8", "replace").strip()
                if not text:
                    yield None, pos
                    continue
                try:
                    yield json.loads(text), pos
                except json.JSONDecodeError as exc:
                    # Previously `except (...): pass` with no log at all, so a
                    # dropped event vanished without trace.
                    print(f"  WARN: skipping malformed event line at {pos}: {exc}",
                          file=sys.stderr)
                    yield None, pos
    except OSError as exc:
        print(f"  WARN: cannot read {status_file}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

# Legacy-Markdown significant characters. Telegram's legacy Markdown has no
# reliable backslash escape, so untrusted text is *stripped* of these rather
# than escaped — the description is plain English prose, so removing them is
# lossless in practice and cannot desync the parser.
_MD_SPECIALS = re.compile(r"[_*\[\]`]")


def sanitize_md(text, limit=MAX_DESCRIPTION_CHARS):
    """Make model-authored text safe to drop into a Markdown message.

    Vision-model prose reached Telegram unescaped with
    ``parse_mode=Markdown``. A single unbalanced ``*``/``_``/backtick makes
    the API return 400 "can't parse entities", and the send helper swallowed
    the exception — so the verdict was silently lost while the pipeline
    reported success.
    """
    if not text:
        return ""
    cleaned = _MD_SPECIALS.sub("", str(text)).replace("\n", " ").strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "\u2026"
    return cleaned


class Telegram:
    """Minimal Telegram client. Never raises; reports failures loudly."""

    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    @property
    def enabled(self):
        return bool(self.bot_token and self.chat_id)

    def _post(self, method, data, headers=None, timeout=15):
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        req = urllib.request.Request(url, data=data, method="POST")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def send(self, text, parse_mode="Markdown"):
        """Send a text message, falling back to plain text on a parse error.

        The fallback is the point: a formatting problem must never be able to
        silently discard a verdict.
        """
        if not self.enabled:
            print("  TG send skipped: no bot token / chat id", file=sys.stderr)
            return False
        for mode in (parse_mode, None):
            payload = {"chat_id": self.chat_id, "text": text}
            if mode:
                payload["parse_mode"] = mode
            try:
                result = self._post("sendMessage", urllib.parse.urlencode(payload).encode())
                if result.get("ok"):
                    return True
                print(f"  TG API error (parse_mode={mode}): {result.get('description')}",
                      file=sys.stderr)
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                print(f"  TG send HTTP {exc.code} (parse_mode={mode}): {body}",
                      file=sys.stderr)
            except Exception as exc:
                print(f"  TG send error (parse_mode={mode}): {exc}", file=sys.stderr)
                return False  # network-level: retrying without markdown won't help
        return False

    def send_photo(self, photo_path, caption):
        if not self.enabled:
            print("  TG photo skipped: no bot token / chat id", file=sys.stderr)
            return False
        try:
            with open(photo_path, "rb") as f:
                img_data = f.read()
        except OSError as exc:
            print(f"  TG photo: cannot read {photo_path}: {exc}", file=sys.stderr)
            return False

        caption = caption[:TELEGRAM_CAPTION_LIMIT]
        # Random boundary: the old fixed '----DogWatchBoundary' with an
        # unescaped, uncapped caption meant model-authored text containing the
        # boundary (or CRLF) could inject or truncate form fields.
        boundary = f"----DogWatch{uuid.uuid4().hex}"
        parts = [
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
            f"{self.chat_id}\r\n",
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="caption"\r\n\r\n'
            f"{caption}\r\n",
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="photo"; filename="dogwatch.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n",
        ]
        body = parts[0].encode() + parts[1].encode() + parts[2].encode()
        body += img_data + f"\r\n--{boundary}--\r\n".encode()
        try:
            result = self._post(
                "sendPhoto", body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                timeout=20,
            )
            if result.get("ok"):
                return True
            print(f"  TG photo API error: {result.get('description')}", file=sys.stderr)
        except Exception as exc:
            print(f"  TG photo error: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def make_stats_bumper(stats_script):
    """Return a ``bump(key, amount=1)`` that can never break the pipeline."""
    def bump(key, amount=1):
        try:
            subprocess.run(
                [sys.executable, stats_script, "bump", key, str(amount)],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass
    return bump


# ---------------------------------------------------------------------------
# Secrets / credentials
# ---------------------------------------------------------------------------

def load_secrets(path):
    """Load the OpenClaw secrets file, or return {} if unavailable.

    Previously a missing file was fatal (``sys.exit(1)``) *even when every
    credential was supplied by environment variable*, and the except clause
    caught ``KeyError``/``FileNotFoundError`` only — so a permission error or
    malformed JSON produced an uncaught traceback.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"  WARN: cannot read secrets {path}: {exc}", file=sys.stderr)
        return {}


def resolve_bot_token(cfg, secrets, notify_cfg):
    """Env var -> notify config botToken -> secrets 'dogwatch' -> 'default'."""
    if cfg.bot_token:
        return cfg.bot_token
    token = str(notify_cfg.get("botToken", "") or "").strip()
    if token:
        return token
    accounts = (secrets.get("channels", {}).get("telegram", {})
                .get("accounts", {}))
    if not isinstance(accounts, dict):
        return ""
    for name in ("dogwatch", "default"):
        entry = accounts.get(name)
        if isinstance(entry, dict) and entry.get("botToken"):
            return str(entry["botToken"])
    return ""


def resolve_provider_key(api_url, secrets):
    """Pick the secrets provider key matching the endpoint being called."""
    providers = secrets.get("models", {}).get("providers", {})
    if not isinstance(providers, dict):
        return ""
    if "openrouter.ai" in api_url:
        name = "openrouter"
    elif "deepseek.com" in api_url:
        name = "deepseek"
    else:
        name = "google"
    entry = providers.get(name)
    if isinstance(entry, dict):
        return str(entry.get("apiKey", "") or "")
    return ""


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_fresh(camera_name, cameras, timeout=15.0, tmp_dir="/tmp"):
    """Grab and validate a clean frame now (RTSP via ffmpeg, HTTP fallback).

    Used when an event reached the log without a usable snapshot (notifier
    debounce race, write race, tmp cleanup), so a snapshot-less digging event
    still gets vision-verified and the siren still gets its chance.

    Unlike the previous version this **validates pixel content**, not just
    file size — see image_quality for why that mattered.
    """
    cam = cameras.get(camera_name)
    if not isinstance(cam, dict):
        return ""
    from image_quality import validate_image_file

    snap_path = os.path.join(tmp_dir, f"dogwatch_check_{camera_name}_{int(time.time())}.jpg")

    def _discard():
        try:
            os.remove(snap_path)
        except OSError:
            pass

    url = cam.get("snapshot_rtsp_fallback", cam.get("snapshot_url", ""))
    if url:
        try:
            subprocess.run(
                ["ffmpeg", "-rtsp_transport", "tcp", "-skip_frame", "nokey",
                 "-i", url, "-frames:v", "1", "-q:v", "2", "-update", "1",
                 "-y", snap_path],
                capture_output=True, timeout=timeout)
            if os.path.exists(snap_path) and validate_image_file(
                    snap_path, log=lambda m: print(m, file=sys.stderr)):
                return snap_path
            _discard()
        except Exception as exc:
            # redact(): TimeoutExpired/CalledProcessError stringify the whole
            # argv, which contains rtsp://user:pass@host.
            print(f"  capture_fresh RTSP failed for {camera_name}: {redact(exc)}",
                  file=sys.stderr)
            _discard()

    su = cam.get("snapshot_url", "")
    if su.startswith("http://") or su.startswith("https://"):
        try:
            import requests
            from requests.auth import HTTPDigestAuth
            parsed = requests.utils.urlparse(su)
            user, pw = parsed.username, parsed.password
            clean_url = su.replace(f"{user}:{pw}@", "") if user else su
            auth = HTTPDigestAuth(user, pw) if user else None
            resp = requests.get(clean_url, auth=auth, timeout=10)
            resp.raise_for_status()
            with open(snap_path, "wb") as f:
                f.write(resp.content)
            if validate_image_file(snap_path,
                                   log=lambda m: print(m, file=sys.stderr)):
                return snap_path
            _discard()
        except Exception as exc:
            print(f"  capture_fresh HTTP failed for {camera_name}: {redact(exc)}",
                  file=sys.stderr)
            _discard()
    return ""


# ---------------------------------------------------------------------------
# Vision verification
# ---------------------------------------------------------------------------

PROMPT_TEXT = (
    "You are analysing a backyard security snapshot to detect a dog "
    "near/under a fence and whether it is digging.\n"
    "Consider motion blur, lighting, and common false positives "
    "(leaves, shadows, wind, cars, people).\n"
    "Digging cues: head/nose lowered to the ground, front paws at "
    "the soil, a paw/scratching motion, or freshly disturbed dirt "
    "directly under the dog.\n"
    "Respond with STRICT JSON only, no prose, in exactly this form:\n"
    '{"dog": "DOG"|"NO_DOG"|"UNCERTAIN", "digging": "YES"|"NO"|"UNCERTAIN", '
    '"description": "short plain-English sentence"}\n'
    "dog = DOG if a dog is clearly or very likely present, NO_DOG if "
    "definitely not, UNCERTAIN if you cannot tell. "
    "digging = YES only if the dog appears to be digging, NO if a dog "
    "is present but not digging, UNCERTAIN otherwise.\n"
    "description = one short natural sentence (max ~15 words) saying "
    'what is actually in the frame and how many dogs — e.g. "2 dogs '
    'digging near the fence", "1 dog lying in the sun near the fence", '
    '"leaves blowing across the yard", "empty yard". Always fill it '
    "in, whatever the verdict."
)


def _parse_vision_content(combined, provider_label):
    """Turn the model's reply text into ``(dog, digging, description)``."""
    dog, digging, description = "UNCERTAIN", None, ""
    try:
        parsed = json.loads(combined)
        dog = str(parsed.get("dog", "UNCERTAIN")).upper()
        dig_raw = str(parsed.get("digging", "UNCERTAIN")).upper()
        digging = True if dig_raw == "YES" else (False if dig_raw == "NO" else None)
        description = str(parsed.get("description", "") or "").strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        up = combined.upper()
        for kw in ("NO_DOG", "UNCERTAIN", "DOG"):
            if kw in up:
                dog = kw
                break
        if '"DIGGING": "YES"' in up or "DIGGING: YES" in up:
            digging = True
        elif '"DIGGING": "NO"' in up or "DIGGING: NO" in up:
            digging = False
        print(f"  vision_verify[{provider_label}]: non-JSON response: {combined}",
              file=sys.stderr)
    if dog not in ("DOG", "NO_DOG", "UNCERTAIN"):
        dog = "UNCERTAIN"
    return dog, digging, description


def vision_verify_with(image_path, api_url, model, api_key, provider_label,
                       timeout=30.0, bump=None):
    """Call one OpenAI-compatible vision endpoint.

    Returns ``{'dog', 'digging', 'description'}`` or None on any failure
    (API error, rate limit, truncated/empty response).
    """
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except OSError as exc:
        print(f"  vision_verify[{provider_label}]: cannot read {image_path}: {exc}",
              file=sys.stderr)
        return None

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEXT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(api_url, data=json.dumps(payload).encode(),
                                method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    if "openrouter.ai" in api_url:
        req.add_header("HTTP-Referer", "https://github.com/VIDGuide/dogwatch")
        req.add_header("X-Title", "DogWatch")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        print(f"  vision_verify[{provider_label}] API error: {exc}", file=sys.stderr)
        return None

    combined = ""
    for choice in result.get("choices", []):
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, str):
            combined += content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    combined += part.get("text", "")
    combined = combined.strip()

    # A truncated (finish_reason: length) or empty reply is an API failure, not
    # an UNCERTAIN verdict — UNCERTAIN would read as "confirmed" downstream.
    if not combined or len(combined) < 5:
        finish = (result.get("choices", [{}]) or [{}])[0].get("finish_reason", "")
        print(f"  vision_verify[{provider_label}]: truncated/empty response "
              f"({finish}): {combined!r}", file=sys.stderr)
        return None

    dog, digging, description = _parse_vision_content(combined, provider_label)
    print(f"  vision_verify[{provider_label}] OK: dog={dog} digging={digging} "
          f"desc={description!r}", file=sys.stderr)
    if bump:
        stat_key = {"primary": "vision_primary_ok",
                    "fallback": "vision_fallback_ok",
                    "fallback2": "vision_fallback2_ok"}.get(provider_label,
                                                            "vision_fallback_ok")
        bump(stat_key)
    return {"dog": dog, "digging": digging, "description": description}


def make_vision_verifier(cfg, bump=None):
    """Return ``verify(image_path)`` trying primary, then fallback, then fallback2.

    A tier is skipped when it has no key, or when it would duplicate an
    already-tried endpoint+key (an account-level 429 or auth failure would
    reject the retry identically — a wasted call, not a fallback).
    """
    tiers = [
        ("primary", cfg.vision_url, cfg.vision_model, cfg.vision_key),
        ("fallback", cfg.fallback_url, cfg.fallback_model, cfg.fallback_key),
        ("fallback2", cfg.fallback2_url, cfg.fallback2_model, cfg.fallback2_key),
    ]

    def verify(image_path):
        tried = []  # (label, url, key) already attempted
        for label, url, model, key in tiers:
            if not key:
                if label != "primary":
                    print(f"  vision_verify: {label} has no key configured - "
                          "skipping", file=sys.stderr)
                continue
            if any(u == url and k == key for (_, u, k) in tried):
                print(f"  vision_verify: {label} uses the same endpoint+key as "
                      f"an earlier tier - skipping (configure "
                      f"DOGWATCH_VISION_{label.upper()}_API_KEY with a "
                      "different provider for a real fallback)",
                      file=sys.stderr)
                continue
            tried.append((label, url, key))
            if label != "primary":
                print(f"  vision_verify: primary failed -> trying {label} "
                      f"({model} @ {url})", file=sys.stderr)
            result = vision_verify_with(image_path, url, model, key, label,
                                        timeout=cfg.vision_timeout, bump=bump)
            if result is not None:
                return result
        return None

    return verify


# ---------------------------------------------------------------------------
# Event collection / dedupe
# ---------------------------------------------------------------------------

def label_for(topic):
    slug = topic.split("/")[-1]
    return slug


def safe_snapshot_path(cfg, snap):
    """Return *snap* if it is a snapshot we are willing to read, else "".

    The ``snapshot`` field comes out of the event log, and everything we do
    with it is an exfiltration primitive: ``_stage_snapshot`` copies it into
    the workspace dir, ``Telegram.send_photo`` uploads it to a chat, and
    ``verify`` base64s it to a third-party vision API. Nothing downstream ever
    re-checks what the path points at.

    The event log's own default location is ``/tmp/dogwatch-events.jsonl`` —
    a predictable name in a world-writable directory, and an append-only
    JSONL file, so "append one more line" is the whole attack. A single
    ``{"ts": ..., "topic": ..., "state": "ON", "snapshot": "/root/.openclaw/
    secrets.json"}`` would have had us Telegram the file out. The same holds
    on the host-cron deployment the README documents, where /tmp is shared
    with every local account.

    So the path is constrained to what the writers actually produce:

    * ``dogwatch-notify.py`` writes ``/tmp/dogwatch_snap_<camera>_<ts>.jpg``
    * ``capture_fresh`` writes ``/tmp/dogwatch_check_<camera>_<ts>.jpg``
    * ``_stage_snapshot`` writes ``<workspace>/dogwatch_<ts>.jpg``

    Checks: the path must resolve (``realpath``, so a planted symlink is
    followed *before* the decision, not after) to a regular file sitting
    directly in an allowed directory, with an expected name prefix and a JPEG
    suffix. Override the directory list with
    ``DOGWATCH_SNAPSHOT_ALLOW_DIRS`` if you relocate the notifier's tmp dir.
    """
    if not snap or not isinstance(snap, str):
        return ""

    def reject(reason):
        print(f"  WARN: ignoring snapshot path {snap!r} from event log: {reason}",
              file=sys.stderr)
        return ""

    if not os.path.isabs(snap):
        return reject("not an absolute path")
    real = os.path.realpath(snap)
    parent, name = os.path.split(real)
    if parent not in cfg.snapshot_allow_dirs:
        return reject(f"resolves outside the allowed directories "
                      f"({', '.join(cfg.snapshot_allow_dirs)})")
    if not name.lower().endswith(SNAPSHOT_SUFFIXES):
        return reject("not a .jpg/.jpeg file")
    if not name.startswith(SNAPSHOT_NAME_PREFIXES):
        return reject("filename is not a dogwatch snapshot name")
    # islink() on the original catches the planted-symlink case explicitly so
    # it is logged as such; realpath above already made it non-exploitable.
    if os.path.islink(snap):
        return reject(f"is a symlink (-> {real})")
    if not os.path.isfile(real):
        return reject("not a regular file")
    return real


def collect_pending(cfg, state, cameras, now=None):
    """Read new events and return ``(pending, max_seen_ts, end_offset)``.

    Dedupe keeps one entry per ``(camera, label)`` per incident window,
    preferring whichever has a real stored snapshot. The window is anchored to
    the *first* entry's timestamp, so a long incident can't slide it forward
    indefinitely and collapse into one report.
    """
    now = now if now is not None else time.time()
    min_ts = now - cfg.max_event_age
    pending = []
    seen = {}          # (camera, label) -> index into pending
    anchor = {}        # (camera, label) -> ts of the first entry in the window
    max_seen = state.ts
    end_offset = state.offset

    for event, pos in read_new_events(cfg.status_file, state):
        end_offset = pos
        if event is None:
            continue
        try:
            ts = float(event["ts"])
            topic = event["topic"]
            st = event["state"]
        except (KeyError, TypeError, ValueError):
            print(f"  WARN: event missing required fields: {event!r}", file=sys.stderr)
            continue

        max_seen = max(max_seen, ts)
        if ts <= state.ts or ts < min_ts or st != "ON":
            continue

        camera = event.get("camera", "camera")
        label = label_for(topic)
        ts_local = time.strftime("%H:%M:%S", time.localtime(ts))
        # safe_snapshot_path also subsumes the old os.path.exists() check: it
        # returns "" for anything missing, and only ever returns a path we are
        # willing to copy, upload and send to the vision API.
        snap = safe_snapshot_path(cfg, event.get("snapshot", ""))
        has_stored = bool(snap)
        key = (camera, label)

        if key in seen:
            if ts - anchor[key] < cfg.dedupe_window:
                existing = pending[seen[key]]
                if has_stored and not existing["snapshot"]:
                    # Upgrade the kept entry to the real frame, but keep the
                    # window anchored to the original timestamp.
                    existing["snapshot"] = _stage_snapshot(cfg, snap, ts)
                    existing["bbox"] = event.get("bbox")
                    existing["score"] = event.get("score", 0.0)
                else:
                    print(f"  dedupe: skip repeat {label} at {ts_local} "
                          f"(same incident as {existing['time']})", file=sys.stderr)
                continue
            # Outside the window — a genuinely new incident.
            del seen[key]
            del anchor[key]

        ws_path = ""
        if has_stored:
            ws_path = _stage_snapshot(cfg, snap, ts)
        else:
            fresh = capture_fresh(camera, cameras, timeout=cfg.capture_timeout)
            if fresh:
                ws_path = fresh
                print(f"  fresh capture for {label} at {ts_local}", file=sys.stderr)

        seen[key] = len(pending)
        anchor[key] = ts
        pending.append({
            "ts": ts,
            "type": label,
            "time": ts_local,
            "snapshot": ws_path,
            "temp_snapshot": not has_stored and bool(ws_path),
            "bbox": event.get("bbox"),
            "score": event.get("score", 0.0),
            "camera": camera,
        })

    return pending, max_seen, end_offset


def _stage_snapshot(cfg, snap, ts):
    """Copy the notifier's /tmp snapshot into the workspace dir."""
    try:
        os.makedirs(cfg.workspace_dir, exist_ok=True)
        ws_path = os.path.join(cfg.workspace_dir, f"dogwatch_{int(ts)}.jpg")
        shutil.copy2(snap, ws_path)
        return ws_path
    except OSError as exc:
        print(f"  WARN: cannot stage snapshot {snap}: {exc}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pretty(label):
    return label.replace("_", " ").title()


def report_verdict(tg, p, result, cfg, bump, run_alarm):
    """Send the confirm / false-alarm / uncertain follow-up for one event."""
    verdict = result["dog"]
    digging = result["digging"]
    description = sanitize_md(result.get("description", ""))
    event_label = pretty(p["type"])

    if verdict == "DOG":
        bump("vision_dog_confirmed")
        dig_line = ""
        if digging is True:
            dig_line = "\n\u26a0\ufe0f *DIGGING detected* — dog appears to be digging!"
        elif digging is False:
            dig_line = "\n\U0001f43e Not digging."
        desc_line = f"\n\U0001f441\ufe0f {description}" if description else ""
        caption = (
            f"\u2705 *Dog Confirmed* at {p['time']}\n"
            f"\U0001f415 Type: {event_label}"
            f"{desc_line}{dig_line}"
        )
        tg.send_photo(p["snapshot"], caption)
        if digging is True:
            run_alarm(p, event_label)
    elif verdict == "NO_DOG":
        bump("vision_false_alarm")
        if description:
            tg.send(f"\u274c *False alarm* — the {event_label} at {p['time']} — "
                    f"{description}")
        else:
            tg.send(f"\u274c *False alarm* — the {event_label} at {p['time']} "
                    f"was just wind/leaves/shadow.")
    else:
        bump("vision_uncertain")
        suffix = f" ({description})" if description else ""
        tg.send(f"\u2753 *Inconclusive* — vision could not confirm or deny the "
                f"{event_label} at {p['time']}.{suffix} Check the snapshot manually.")


# ---------------------------------------------------------------------------
# Jarvis announce hook (optional)
# ---------------------------------------------------------------------------
#
# Vision-confirmed digging events are POSTed as plain event FACTS to the
# OpenClaw hooks rail (agentId=jarvis). The jarvis brain phrases the spoken
# line and speaks it over the office speakers; DogWatch never sends
# pre-written speech, only facts. ``deliver`` is always false — the spoken
# announce IS the delivery, the turn result must not echo to any chat.
#
# Config: a ``jarvis_hook`` section in the notify config (url + token) or
# DOGWATCH_JARVIS_HOOK_URL / DOGWATCH_JARVIS_HOOK_TOKEN env vars.
# Missing url or token (or enabled:false) = feature off, zero overhead.

JARVIS_HOOK_TIMEOUT = 5.0


def _load_jarvis_hook_cfg(cfg):
    """Resolve the jarvis hook ``{url, token}`` for this run.

    Env overrides win; otherwise read the notify config's ``jarvis_hook``
    section. Returns None when the feature is off.
    """
    notify = {}
    try:
        if cfg.notify_config and os.path.exists(cfg.notify_config):
            with open(cfg.notify_config) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                notify = loaded.get("jarvis_hook") or {}
    except Exception as exc:
        print(f"  WARN: cannot read jarvis_hook from {cfg.notify_config}: {exc}",
              file=sys.stderr)
    if not isinstance(notify, dict) or notify.get("enabled") is False:
        return None
    url = str(cfg.jarvis_hook_url or notify.get("url") or "").strip()
    token = str(cfg.jarvis_hook_token or notify.get("token") or "").strip()
    if not url or not token:
        return None
    return {"url": url, "token": token}


def _jarvis_digging_facts(p, event_label):
    """Plain-sentence event facts for the jarvis rail.

    No commands, no urgency claims, no exclamation marks — the jarvis brain
    writes the spoken line and decides priority.
    """
    camera = str(p.get("camera") or "camera")
    cam_label = "main camera" if camera == "camera" else f"{camera} camera"
    started = str(p.get("time") or "")[:5] or "unknown time"
    score = p.get("score") or 0.0
    facts = [f"digging detected - {cam_label}", f"started {started}"]
    if score and score > 0:
        facts.append(f"detector confidence {score:.2f}")
    return "DogWatch event: " + ", ".join(facts) + "."


def _jarvis_announce(cfg, p, event_label):
    """Fire-and-forget POST of digging facts to the OpenClaw hooks rail.

    Runs in a daemon thread with a short timeout so it can never delay the
    siren or break the Telegram/snapshot path; failures are logged to the
    check log only.
    """
    hook = _load_jarvis_hook_cfg(cfg)
    if not hook:
        return

    def post():
        payload = {
            "agentId": "jarvis",
            "name": "dogwatch-digging",
            "message": _jarvis_digging_facts(p, event_label),
            "wakeMode": "now",
            "sessionMode": "isolated",
            "deliver": False,
        }
        req = urllib.request.Request(
            hook["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {hook['token']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=JARVIS_HOOK_TIMEOUT) as resp:
                body = resp.read(200).decode("utf-8", "replace").strip()
            print(f"  jarvis announce hook: HTTP {resp.status} {body}",
                  file=sys.stderr)
        except Exception as exc:
            print(f"  jarvis announce hook error: {exc}", file=sys.stderr)

    try:
        threading.Thread(target=post, daemon=True).start()
    except Exception as exc:
        print(f"  jarvis announce hook error: {exc}", file=sys.stderr)


def make_alarm_runner(cfg, tg, bump, verify, cameras):
    """Return ``run_alarm(pending_entry, event_label)``.

    Fires the siren only for a vision-confirmed digging event, then re-checks
    the camera to see whether the dog was actually distracted.
    """
    def run_alarm(p, event_label):
        # Jarvis spoken announce (optional): fired for every vision-confirmed
        # digging event, independent of the siren's own window/replay guards
        # (the announce quiet hours are enforced on the jarvis side).
        _jarvis_announce(cfg, p, event_label)

        if not cfg.alarm_script or not os.path.exists(cfg.alarm_script):
            return
        reason = f"vision confirmed digging — {event_label} at {p['time']}"
        rc = None
        try:
            rc = subprocess.run([cfg.alarm_script, reason], timeout=60).returncode
        except Exception as exc:
            print(f"  dog-alarm hook error: {exc}", file=sys.stderr)
        if rc == 0:
            _alarm_followup(p)

    def _alarm_followup(p):
        delay = cfg.followup_delay
        if delay <= 0:
            return
        time.sleep(delay)
        bump("alarm_followups")
        snap = capture_fresh(p.get("camera", "camera"), cameras,
                             timeout=cfg.capture_timeout)
        if not snap:
            tg.send("\u26a0\ufe0f *Siren follow-up* — could not grab a fresh frame "
                    "to check whether the dog was distracted.")
            bump("alarm_followup_uncertain")
            return
        try:
            result = verify(snap)
            if result is None:
                tg.send("\u26a0\ufe0f *Siren follow-up* — vision check failed on the "
                        "fresh frame; could not confirm the dog was distracted.")
                bump("alarm_followup_uncertain")
                return
            dog, digging = result["dog"], result["digging"]
            description = sanitize_md(result.get("description", ""))
            if dog == "DOG" and digging is True:
                bump("alarm_followup_still_digging")
                head = "\U0001f50a *Siren follow-up* — dog is *still digging*!"
            elif dog == "DOG":
                bump("alarm_followup_present")
                head = "\U0001f50a *Siren follow-up* — dog still at the fence, not digging."
            elif dog == "NO_DOG":
                bump("alarm_followup_clear")
                head = "\u2705 *Siren follow-up* — dog has left the fence — siren worked."
            else:
                bump("alarm_followup_uncertain")
                head = "\u2753 *Siren follow-up* — could not tell from the fresh frame."
            caption = f"{head}\n\U0001f441\ufe0f {description}" if description else head
            tg.send_photo(snap, caption)
        finally:
            # The follow-up capture is scratch data; the old code left every
            # one of these behind in /tmp forever.
            try:
                os.remove(snap)
            except OSError:
                pass

    return run_alarm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None, env=None):
    cfg = Config(env)

    if not os.path.exists(cfg.status_file):
        return 0

    state = CheckState(cfg.state_file, cfg.legacy_ts_file).load()

    notify_cfg = {}
    if cfg.notify_config and os.path.exists(cfg.notify_config):
        try:
            with open(cfg.notify_config) as f:
                notify_cfg = json.load(f)
        except Exception as exc:
            print(f"  WARN: cannot load notify config {cfg.notify_config}: {exc}",
                  file=sys.stderr)
    cameras = notify_cfg.get("cameras", {}) if isinstance(notify_cfg, dict) else {}
    # Example configs carry "_comment_*" keys inside "cameras"; they are
    # documentation, not cameras.
    cameras = {k: v for k, v in cameras.items()
               if isinstance(v, dict) and not k.startswith("_")}

    secrets = load_secrets(cfg.secrets_file)
    chat_id = cfg.chat_id or str(notify_cfg.get("chat_id", "") or "").strip()
    bot_token = resolve_bot_token(cfg, secrets, notify_cfg)
    if not cfg.vision_key:
        cfg.vision_key = resolve_provider_key(cfg.vision_url, secrets)
    if not cfg.fallback_key:
        cfg.fallback_key = resolve_provider_key(cfg.fallback_url, secrets)
    if not cfg.fallback2_key:
        cfg.fallback2_key = resolve_provider_key(cfg.fallback2_url, secrets)

    if not cfg.vision_key:
        print("ERROR: no vision API key configured — set DOGWATCH_VISION_API_KEY "
              "or add secrets.json models.providers.<provider>.apiKey",
              file=sys.stderr)
        return 1
    if not bot_token or not chat_id:
        print("ERROR: no Telegram bot token / chat id configured — set "
              "DOGWATCH_BOT_TOKEN + DOGWATCH_CHAT_ID, or botToken + chat_id in "
              f"{cfg.notify_config}", file=sys.stderr)
        return 1

    tg = Telegram(bot_token, chat_id)
    bump = make_stats_bumper(cfg.stats_script)
    verify = make_vision_verifier(cfg, bump=bump)
    run_alarm = make_alarm_runner(cfg, tg, bump, verify, cameras)

    pending, max_seen, end_offset = collect_pending(cfg, state, cameras)

    if not pending:
        state.advance(ts=max_seen, offset=end_offset)
        state.save()
        return 0

    lines = []
    for p in pending:
        icon = " \U0001f4f8" if p["snapshot"] else ""
        lines.append(f'  \u2022 {pretty(p["type"])} at {p["time"]}{icon}')
    count = len(pending)
    tg.send(
        f'\U0001f4f9 *DogWatch Alert* — {count} event{"s" if count > 1 else ""} detected\n'
        + "\n".join(lines)
        + "\n\n_Verifying with vision…_"
    )

    # Advance the watermark per completed event, and persist whatever progress
    # we made even if something below raises. Previously the watermark was only
    # written after the whole loop, so any mid-loop failure re-alerted every
    # in-window event (and re-attempted the siren) on the next cycle.
    try:
        for p in pending:
            try:
                _process_one(p, tg, cfg, bump, verify, run_alarm)
            finally:
                state.advance(ts=p["ts"])
                _discard_temp(p)
            time.sleep(1)
        state.advance(ts=max_seen, offset=end_offset)
    finally:
        state.save()
    return 0


def _process_one(p, tg, cfg, bump, verify, run_alarm):
    event_label = pretty(p["type"])
    if not p["snapshot"]:
        tg.send(f"\u26a0\ufe0f *No snapshot available* for {event_label} at "
                f"{p['time']} — fresh capture failed too. Vision check skipped; "
                f"no alarm decision made for this event.")
        return
    bump("vision_checks")
    result = verify(p["snapshot"])
    if result is None:
        bump("vision_failed")
        tg.send(f"\u26a0\ufe0f *Vision check failed* for {event_label} at "
                f"{p['time']} — see script logs for the API error. Detection "
                f"alert above is still valid; this only affects the "
                f"confirm/false-alarm follow-up.")
        return
    report_verdict(tg, p, result, cfg, bump, run_alarm)


def _discard_temp(p):
    """Remove a fresh-capture scratch file once the event is reported.

    Fresh captures written to /tmp by capture_fresh were never deleted by
    anything (only the notifier's own snapshots were), so they accumulated
    indefinitely. Staged workspace copies are intentionally kept.
    """
    if not p.get("temp_snapshot") or not p.get("snapshot"):
        return
    try:
        os.remove(p["snapshot"])
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())

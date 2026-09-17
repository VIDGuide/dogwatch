#!/usr/bin/env python3
"""bins-watch.py — weekly kerbside bin put-out verification.

Why this exists
---------------
Michael has four wheelie bins: two yellow-lid recycling, one green-lid
organics, one red-lid garbage.  They live on a concrete pad beside the
house, which is visible from the front driveway camera when it is parked
on PTZ preset "West Side".

Spotting a bin *at the kerb* proved unreliable: on collection day the
neighbours' bins are on their verges too, and the vision model happily
counts those.  So this watches the **inverse** signal instead — how many
bins are still sitting on the storage pad.  That pad is a tight,
neighbour-free scene, and its composition maps directly onto state.

It is deliberately a VERIFIER, not a convenience.  The week is derived
from a hard anchor date rather than inferred from what the camera sees,
so putting the wrong bins out is detected as a mistake instead of being
rationalised into "it must be the other week".

Collection calendar (Michael's, Hunter region)
----------------------------------------------
  Week A : red (garbage) + green (organics) go out
  Week B : all four go out
Anchor: Mon 2026-09-14 is a Week A week; it alternates weekly.

Expected pad composition once the job is DONE
---------------------------------------------
  Week A -> 2 yellow left on the pad
  Week B -> pad empty

Phases
------
  --phase putout   Thursday evening: watch until the correct bins are out.
                   Announces every real state change; nags from nag_start.
  --phase backin   Friday late morning: confirm the bins came home.

Exit codes: 0 ok (whether or not it spoke), 1 config/usage error,
2 camera unreachable, 3 vision unavailable (no state change announced).
"""

import argparse
import base64
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "dogwatch-notify.config.json")
SECRETS_FILE = os.path.expanduser("~/.openclaw/secrets.json")
STATE_FILE = os.path.expanduser("~/source/dogTracker/bins-watch.state")

# ---------------------------------------------------------------- config ----

DEFAULTS = {
    "enabled": True,
    "cam_host": "192.168.1.56",
    "cam_user": "admin",
    "preset": 2,          # "West Side" — frames the storage pad
    "home_preset": 0,     # "Centre" — the camera's parked view
    # Storage-pad region of the preset frame, normalised x1,y1,x2,y2.
    "pad_roi": [0.66, 0.32, 1.00, 0.98],
    "week_anchor": "2026-09-14",   # a Monday that is Week A
    # How many of each colour actually leave the pad on collection night.
    # Week B is ALL FOUR, which means both yellow bins — a set would under-count it.
    "out_sets": {"A": {"red": 1, "green": 1},
                 "B": {"yellow": 2, "green": 1, "red": 1}},
    "pad_full": {"yellow": 2, "green": 1, "red": 1},
    "put_out_start": "18:00",
    "nag_start": "20:00",
    "nag_stop": "23:00",   # no overnight spam; the nag window ends here
    "vision_url": ("https://generativelanguage.googleapis.com/v1beta/"
                   "openai/chat/completions"),
    "vision_model": "gemini-2.5-flash",
    # Fallback used when the primary errors or rate-limits (the project already
    # carries an OpenRouter key for exactly this reason).
    "fallback_url": "https://openrouter.ai/api/v1/chat/completions",
    "fallback_model": "qwen/qwen3.7-flash",
    "vision_samples": 3,   # majority vote; one flaky reply must not move state
    "sample_gap": 4,       # seconds between samples (stay under free-tier RPM)
}

PROMPT = (
    "This crop shows the bin storage pad beside a house (concrete slab, "
    "fence behind) at an Australian home.\n"
    "Up to four wheelie bins can stand here: two with YELLOW lids "
    "(recycling), one with a GREEN lid (organics), one with a RED lid "
    "(garbage). Bodies are dark green or dark grey.\n"
    "Count how many wheelie bins are actually in this storage pad right "
    "now, by lid colour. Do not count shadows, the fence, plants, hoses, "
    "or the car.\n"
    'Reply STRICT JSON only: {"count": <int>, "yellow": <int>, '
    '"green": <int>, "red": <int>, "description": "<=15 words"}'
)


def load_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: cannot read config {path}: {exc}")
    bins = dict(DEFAULTS)
    bins.update(cfg.get("bins", {}) or {})
    bins["chat_id"] = str((cfg.get("bins", {}) or {}).get("chat_id")
                          or cfg.get("chat_id", ""))
    bins["bot_token"] = cfg.get("botToken", "")
    # camera credentials: dedicated block, else reuse the NVR's (same box password)
    fd = cfg.get("find_dogs", {}) or {}
    bins.setdefault("cam_user", fd.get("nvr_user", "admin"))
    if not bins.get("cam_password"):
        bins["cam_password"] = fd.get("nvr_password", "")
    # env overrides (handy for the spike / a different camera)
    for env_key, cfg_key in (("DOGWATCH_BINS_CAM_HOST", "cam_host"),
                             ("DOGWATCH_BINS_CAM_USER", "cam_user"),
                             ("DOGWATCH_BINS_CAM_PASSWORD", "cam_password"),
                             ("DOGWATCH_BINS_PRESET", "preset"),
                             ("DOGWATCH_BINS_CHAT_ID", "chat_id"),
                             ("DOGWATCH_BINS_WEEK_ANCHOR", "week_anchor")):
        val = os.environ.get(env_key)
        if val:
            bins[cfg_key] = val
    if str(bins.get("preset", "")).isdigit():
        bins["preset"] = int(bins["preset"])
    return bins


def vision_key(bins):
    key = os.environ.get("DOGWATCH_VISION_API_KEY", "")
    if key:
        return key
    return provider_key("google")


def provider_key(name):
    try:
        with open(SECRETS_FILE) as f:
            secrets = json.load(f)
        return secrets["models"]["providers"][name]["apiKey"]
    except Exception:
        return ""


# ------------------------------------------------------------------ week ----

def parse_hhmm(text):
    hh, mm = str(text).split(":")
    return dt.time(int(hh), int(mm))


def week_label(cfg, when=None):
    """'A' or 'B' — derived from the anchor, never from the camera."""
    when = when or dt.datetime.now()
    anchor = dt.date.fromisoformat(cfg["week_anchor"])
    monday = when.date() - dt.timedelta(days=when.weekday())
    weeks = (monday - anchor).days // 7
    return "A" if weeks % 2 == 0 else "B"


def expected_pad(cfg, label):
    """Pad composition once the right bins are out."""
    full = dict(cfg["pad_full"])
    for colour, n in cfg["out_sets"][label].items():
        full[colour] = full.get(colour, 0) - int(n)
    return {k: v for k, v in full.items() if v > 0}


def describe_out(cfg, label):
    """'red + green', or '2× yellow + green + red' for Week B."""
    parts = [f"{n}× {c}" if int(n) > 1 else c
             for c, n in cfg["out_sets"][label].items()]
    return " + ".join(parts)


# ---------------------------------------------------------------- camera ----

class Camera:
    """Minimal Reolink CGI client (the NVR cannot PTZ this channel)."""

    def __init__(self, host, user, password, timeout=10):
        self.base = f"http://{host}/cgi-bin/api.cgi"
        self.user, self.password, self.timeout = user, password, timeout
        self.token = None

    def _post(self, cmd, body):
        url = f"{self.base}?cmd={cmd}"
        if self.token:
            url += f"&token={urllib.parse.quote(self.token)}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        if payload and payload[0].get("code") not in (0, None):
            raise RuntimeError(payload[0].get("error", {}).get("detail", "api error"))
        return payload

    def login(self):
        body = [{"cmd": "Login", "param": {"User": {
            "userName": self.user, "password": self.password}}}]
        payload = self._post("Login", body)
        self.token = payload[0]["value"]["Token"]["name"]
        return self.token

    def ptz_preset(self, preset_id, speed=32):
        return self._post("PtzCtrl", [{"cmd": "PtzCtrl", "action": 0, "param": {
            "channel": 0, "op": "ToPos", "id": int(preset_id), "speed": speed}}])

    def snap(self):
        """Full-resolution JPEG from the camera's own snapshot endpoint."""
        url = (f"{self.base}?cmd=Snap&channel=0&rs={int(time.time())}"
               f"&token={urllib.parse.quote(self.token or '')}")
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            data = resp.read()
        if not data.startswith(b"\xff\xd8"):
            raise RuntimeError("snapshot endpoint did not return a JPEG")
        return data


def crop_pad(jpeg_bytes, roi):
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg_bytes))
    w, h = im.size
    box = (int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h))
    return im.crop(box)


# ---------------------------------------------------------------- vision ----

def ask_count(image, cfg, key, url=None, model=None):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": model or cfg["vision_model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 2000,   # gemini-2.5-flash spends part of this on internal
                               # thinking; 300 truncated the JSON intermittently
    }
    url = url or cfg["vision_url"]
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {key}")
    if "openrouter.ai" in url:
        req.add_header("HTTP-Referer", "https://github.com/VIDGuide/dogwatch")
        req.add_header("X-Title", "DogWatch")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # surface the API's own message — otherwise an unattended run just
        # logs "HTTP Error 400" with nothing to act on
        body = exc.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"vision HTTP {exc.code}: {body}") from exc
    text = "".join(c.get("message", {}).get("content", "") or ""
                   for c in result.get("choices", [])).strip()
    # The reply may be wrapped in ```json fences; be tolerant, then take the
    # outermost {...} so a stray prefix/suffix can't break the parse.
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"vision reply had no JSON object: {text[:120]!r}")
    parsed = json.loads(text[start:end + 1])
    return {"yellow": int(parsed.get("yellow", 0)),
            "green": int(parsed.get("green", 0)),
            "red": int(parsed.get("red", 0))}


def pad_composition(image, cfg, key, log):
    """Majority vote over several samples — a single flaky reply, or a
    transient rate-limit on one provider, must not move the state machine.
    Primary provider first; the fallback is tried only if a sample fails."""
    fb_key = provider_key("openrouter")
    samples, votes = [], {}
    gap = float(cfg.get("sample_gap", 4))
    for i in range(max(1, int(cfg["vision_samples"]))):
        if i:
            time.sleep(gap)          # pace calls to stay under free-tier RPM
        counts = None
        for attempt in range(2):
            try:
                counts = ask_count(image, cfg, key)
                break
            except Exception as exc:
                log(f"  vision sample {i + 1} attempt {attempt + 1} failed: {exc}")
                if attempt == 0:
                    time.sleep(6)
        if counts is None and fb_key:
            try:
                counts = ask_count(image, cfg, fb_key, cfg["fallback_url"],
                                   cfg["fallback_model"])
                log(f"  vision sample {i + 1} recovered on fallback provider")
            except Exception as exc:
                log(f"  vision sample {i + 1} fallback failed: {exc}")
        if counts is None:
            continue
        samples.append(counts)
        votes[tuple(sorted(counts.items()))] = votes.get(tuple(sorted(counts.items())), 0) + 1
    if not samples:
        return None, samples
    winner = max(votes.items(), key=lambda kv: kv[1])[0]
    return dict(winner), samples


# ------------------------------------------------------------- classify ----

def classify(observed, cfg, label, phase):
    """Return (state, human_sentence). state in
    {not_out, done, wrong, partial, back_in} or None when uncertain."""
    # vision reports explicit zeros; compare only colours actually present
    observed = {k: v for k, v in observed.items() if v > 0}
    full = {k: v for k, v in cfg["pad_full"].items() if v > 0}
    want = expected_pad(cfg, label)
    should = set(cfg["out_sets"][label])

    if phase == "backin":
        if observed == full:
            return "back_in", "all four bins are back on the pad"
        missing = [c for c in full if observed.get(c, 0) < full[c]]
        have = sum(observed.values())
        return "back_in", (f"{have} of 4 bins on the pad; still out: "
                           + ", ".join(sorted(missing)))

    if observed == full:
        return "not_out", "all four bins are still on the pad"
    if observed == want:
        return "done", f"the right bins are out ({describe_out(cfg, label)})"

    out_now = [c for c in full if observed.get(c, 0) < full[c]]
    gone = sorted(set(out_now) - should)
    missing = sorted(should - set(out_now))
    if gone:
        # something left the pad that should NOT have -> the classic
        # wrong-week mistake, which is the whole point of this watcher
        return "wrong", (f"{', '.join(gone)} went out, but Week {label} needs "
                         f"{describe_out(cfg, label)}")
    return "partial", (f"{', '.join(missing)} still on the pad — Week {label} "
                       f"needs {describe_out(cfg, label)}")


def speak(cfg, text, dry_run, log):
    if dry_run:
        log(f"  [dry-run telegram] {text}")
        return
    data = urllib.parse.urlencode({"chat_id": cfg["chat_id"], "text": text,
                                   "disable_notification": "false"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage", data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as exc:
        log(f"  telegram send failed: {exc}")


# ------------------------------------------------------------------ main ----

def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(path, state):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["putout", "backin"], default="putout")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--state", default=STATE_FILE)
    ap.add_argument("--dry-run", action="store_true",
                    help="do the camera + vision work, but print instead of send")
    ap.add_argument("--force", action="store_true",
                    help="ignore the time window (for testing)")
    ap.add_argument("--resolve", action="store_true",
                    help="mark this week resolved without touching the camera")
    ap.add_argument("--cancel", action="store_true",
                    help="mark this week cancelled (e.g. 'skip this week')")
    args = ap.parse_args()

    logs = []
    log = lambda m: (logs.append(m), print(m, file=sys.stderr))  # noqa: E731

    cfg = load_config(args.config)
    if not cfg.get("enabled"):
        log("bins-watch disabled in config")
        return 0

    now = dt.datetime.now()
    label = week_label(cfg, now)
    day = now.date().isoformat()          # state key for today
    api_key = vision_key(cfg)             # vision credential (distinct!)
    state = load_state(args.state)

    if args.cancel or args.resolve:
        state[day] = {"state": "cancelled" if args.cancel else "done",
                      "at": now.isoformat(), "label": label, "manual": True}
        save_state(args.state, state)
        log(f"marked {day} {'cancelled' if args.cancel else 'resolved'}")
        return 0

    if state.get(day, {}).get("state") in ("done", "cancelled"):
        log(f"{day} already {state[day]['state']} — nothing to do")
        return 0

    # ---- time windows (skipped with --force) -------------------------------
    if not args.force:
        if args.phase == "putout":
            if not (parse_hhmm(cfg["put_out_start"]) <= now.time()
                    < parse_hhmm(cfg["nag_stop"])):
                log(f"outside put-out window ({cfg['put_out_start']}-"
                    f"{cfg['nag_stop']}) — skipping")
                return 0
        else:
            target = parse_hhmm("10:00")
            if abs((now - now.replace(hour=target.hour, minute=target.minute,
                                      second=0)).total_seconds()) > 30 * 60:
                log("outside the back-in window — skipping")
                return 0

    # ---- look at the pad --------------------------------------------------
    if not api_key:
        log("  no vision API key available (secrets.json providers.google)")
        return 3
    cam = Camera(cfg["cam_host"], cfg["cam_user"], cfg["cam_password"])
    try:
        cam.login()
        cam.ptz_preset(cfg["preset"])
        time.sleep(4)                       # let the PTZ settle
        jpeg = cam.snap()
        log(f"  snapped {len(jpeg)} bytes at preset {cfg['preset']}")
    except Exception as exc:
        log(f"  camera error: {exc}")
        return 2
    finally:
        try:                                # always park the camera again
            cam.ptz_preset(cfg["home_preset"])
        except Exception:
            pass

    image = crop_pad(jpeg, cfg["pad_roi"])
    observed, samples = pad_composition(image, cfg, api_key, log)
    log(f"  samples: {samples}")
    if observed is None:
        log("  no usable vision sample — leaving state unchanged")
        return 3
    log(f"  pad composition: {observed}")

    verdict, sentence = classify(observed, cfg, label, args.phase)
    log(f"  week {label} / {args.phase} -> {verdict}: {sentence}")

    prev = state.get(day, {}).get("state")
    nagging = (args.phase == "putout"
               and parse_hhmm(cfg["nag_start"]) <= now.time()
               < parse_hhmm(cfg["nag_stop"]))
    nag_due = nagging and (now - dt.datetime.fromisoformat(
        state.get(day, {}).get("last_nag", "1970-01-01T00:00:00"))
        >= dt.timedelta(minutes=15))

    message = None
    if verdict == "done":
        if prev != "done":
            message = f"🟢 Bins out — Week {label}: {sentence}."
        state[day] = {"state": "done", "at": now.isoformat(), "label": label,
                      "observed": observed}
    elif args.phase == "backin":
        # report once per day, and again only if the picture changes
        if prev != verdict or state.get(day, {}).get("report_at") != day:
            message = f"📦 Bins back-in check — {sentence}."
        state[day] = {"state": verdict, "at": now.isoformat(), "label": label,
                      "observed": observed, "report_at": day}
    else:
        if verdict != prev:
            if verdict == "wrong":
                message = f"❌ Wrong bins — {sentence}."
            elif verdict == "partial":
                message = f"🟡 Bins going out — {sentence}."
            else:
                message = None      # not_out yet: only nags, no chatty echo
        if verdict != "done":
            state[day] = {"state": verdict, "at": now.isoformat(), "label": label,
                          "observed": observed,
                          "last_nag": state.get(day, {}).get("last_nag",
                                                             "1970-01-01T00:00:00")}
        if nag_due and verdict in ("not_out", "partial"):
            message = (f"🔔 Week {label} bins reminder — {sentence}. "
                       f"Needed out: {describe_out(cfg, label)}.")
            state[day]["last_nag"] = now.isoformat()

    save_state(args.state, state)
    if message:
        speak(cfg, message, args.dry_run, log)
    else:
        log("  nothing to announce")
    return 0


if __name__ == "__main__":
    sys.exit(main())

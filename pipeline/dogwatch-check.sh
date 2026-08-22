#!/bin/bash
# DogWatch event checker — fully event-driven
# Checks for recent dog events, sends quick Telegram ping,
# runs vision model verification (via any OpenAI-compatible chat completions
# API — defaults to Google Gemini, which has a generous free tier for this
# usage pattern, but any compatible provider/model can be swapped in via
# env vars) on each snapshot, and sends a confirmation / false-alarm
# follow-up.
# Silent exit on no events — no model call, no noise.
STATUS_FILE="/tmp/dogwatch-events.jsonl"
# GNU date syntax (-d) — this script targets Linux cron/systemd hosts and
# will not run as-is on macOS/BSD (which needs `date -v-4M +%s` instead).
# Lookback must EXCEED the loop period (300s) + processing slack. With a
# 4-min cutoff and a 5-min loop, events landing in the last ~1 min of a
# cycle window aged past the cutoff before the next cycle read them and
# were SILENTLY dropped — no vision, no siren, no follow-up message
# (seen 2026-08-22 15:28:38 digging). 7 min = 300s loop + 120s slack.
CUTOFF=$(date +%s -d "7 minutes ago")
WORKSPACE_SNAP_DIR="${DOGWATCH_WORKSPACE_DIR:-$HOME/.openclaw/workspace/dogwatch_snaps}"
MARKER_FILE="/tmp/dogwatch-pending.jsonl"
# Watermark of the newest event ts already seen — prevents the dead-zone
# drop above AND re-processing (a longer lookback alone would re-process
# old events on later cycles). Lives in /tmp next to the events file so
# both reset together on container recreate.
LAST_TS_FILE="/tmp/dogwatch-last-ts"
LAST_TS=$(cat "$LAST_TS_FILE" 2>/dev/null || echo 0)
SECRETS_FILE="$HOME/.openclaw/secrets.json"
# Chat id is loaded from (in order): DOGWATCH_CHAT_ID env, the notify config
# file's "chat_id", so it is not hardcoded in this (publicly-committed) script.
NOTIFY_CONFIG="${DOGWATCH_NOTIFY_CONFIG:-$(dirname "$(readlink -f "$0")")/dogwatch-notify.config.json}"
CHAT_ID="${DOGWATCH_CHAT_ID:-}"
if [ -z "$CHAT_ID" ] && [ -f "$NOTIFY_CONFIG" ]; then
  CHAT_ID=$(python3 -c "import json,sys; print(json.load(open('$NOTIFY_CONFIG')).get('chat_id',''))" 2>/dev/null)
fi
# Bot token is loaded from (in order): DOGWATCH_BOT_TOKEN env, the notify
# config file's "botToken" (recommended — dogwatch-owned, gitignored), then
# the "dogwatch"/"default" accounts in the OpenClaw secrets file.
BOT_TOKEN="${DOGWATCH_BOT_TOKEN:-}"
if [ -z "$BOT_TOKEN" ] && [ -f "$NOTIFY_CONFIG" ]; then
  BOT_TOKEN=$(python3 -c "import json,sys; print(json.load(open('$NOTIFY_CONFIG')).get('botToken',''))" 2>/dev/null)
fi

# Vision model config — all overridable so any OpenAI-compatible vision
# endpoint can be used. Primary default is qwen/qwen3.7-flash via OpenRouter
# (fast, cheap, strong at small objects in wide frames — chosen 2026-08-15
# after Gemini free tier kept 429ing mid-scan).
#   DOGWATCH_VISION_API_URL   — chat completions endpoint (default: OpenRouter)
#   DOGWATCH_VISION_MODEL     — model name (default: qwen/qwen3.7-flash)
#   DOGWATCH_VISION_API_KEY   — API key. Falls back to the "openrouter"
#                               provider key in secrets.json when unset; if
#                               the URL points at Google's endpoint, falls
#                               back to the "google" provider key instead.
VISION_API_URL="${DOGWATCH_VISION_API_URL:-https://openrouter.ai/api/v1/chat/completions}"
VISION_MODEL="${DOGWATCH_VISION_MODEL:-qwen/qwen3.7-flash}"
VISION_API_KEY="${DOGWATCH_VISION_API_KEY:-}"

# Fallback vision provider (OpenRouter by default) — used automatically when
# the primary endpoint fails (quota/429, network, timeout, model error).
# Any OpenAI-compatible provider works; override via
#   DOGWATCH_VISION_FALLBACK_API_URL / DOGWATCH_VISION_FALLBACK_MODEL /
#   DOGWATCH_VISION_FALLBACK_API_KEY
# The key falls back to the "openrouter" provider in secrets.json when unset
# (Google's endpoint is not used for the fallback).
VISION_FALLBACK_API_URL="${DOGWATCH_VISION_FALLBACK_API_URL:-https://openrouter.ai/api/v1/chat/completions}"
VISION_FALLBACK_MODEL="${DOGWATCH_VISION_FALLBACK_MODEL:-google/gemini-3-flash-preview}"
VISION_FALLBACK_API_KEY="${DOGWATCH_VISION_FALLBACK_API_KEY:-}"

# Optional dog-alarm hook — sounded by this script after vision verification
# confirms digging. The alarm script has its own guard rails (time window,
# replay interval, enabled flag), so this stays a dumb fire-and-forget call.
ALARM_SCRIPT="${DOGWATCH_ALARM_SCRIPT:-/app/dog-alarm.sh}"

export DW_ALARM_SCRIPT="$ALARM_SCRIPT"
# Daily stats capture (per-day counters for the Daily Dog Report). The stats
# script lives in the image; override for host testing. Failures are never
# fatal to the alert pipeline.
export DW_STATS_SCRIPT="${DW_STATS_SCRIPT:-/app/stats.py}"

mkdir -p "$WORKSPACE_SNAP_DIR"
rm -f "$MARKER_FILE"

if [ ! -f "$STATUS_FILE" ]; then
  exit 0
fi

# Pass shell vars to Python as env vars so we don't fight with heredoc quoting
export DW_CUTOFF="$CUTOFF"
export DW_LAST_TS="$LAST_TS"
export DW_LAST_TS_FILE="$LAST_TS_FILE"
export DW_WORKSPACE_DIR="$WORKSPACE_SNAP_DIR"
export DW_MARKER_FILE="$MARKER_FILE"
export DW_SECRETS_FILE="$SECRETS_FILE"
export DW_BOT_TOKEN="$BOT_TOKEN"
export DW_CHAT_ID="$CHAT_ID"
export DW_STATUS_FILE="$STATUS_FILE"
export DW_NOTIFY_CONFIG="$NOTIFY_CONFIG"
export DW_VISION_API_URL="$VISION_API_URL"
export DW_VISION_MODEL="$VISION_MODEL"
export DW_VISION_API_KEY="$VISION_API_KEY"
export DW_VISION_FALLBACK_API_URL="$VISION_FALLBACK_API_URL"
export DW_VISION_FALLBACK_MODEL="$VISION_FALLBACK_MODEL"
export DW_VISION_FALLBACK_API_KEY="$VISION_FALLBACK_API_KEY"

python3 << 'PYEOF'
import json, time, sys, shutil, os, subprocess, urllib.request, urllib.parse, base64

CUTOFF = float(os.environ['DW_CUTOFF'])
LAST_TS = float(os.environ.get('DW_LAST_TS', '0') or '0')
WORKSPACE_DIR = os.environ['DW_WORKSPACE_DIR']
MARKER_FILE = os.environ['DW_MARKER_FILE']
SECRETS_FILE = os.path.expanduser(os.environ['DW_SECRETS_FILE'])
CHAT_ID = os.environ['DW_CHAT_ID']
STATUS_FILE = os.environ['DW_STATUS_FILE']
VISION_API_URL = os.environ['DW_VISION_API_URL']
VISION_MODEL = os.environ['DW_VISION_MODEL']
VISION_API_KEY = os.environ.get('DW_VISION_API_KEY', '')
VISION_FALLBACK_API_URL = os.environ['DW_VISION_FALLBACK_API_URL']
VISION_FALLBACK_MODEL = os.environ['DW_VISION_FALLBACK_MODEL']
VISION_FALLBACK_API_KEY = os.environ.get('DW_VISION_FALLBACK_API_KEY', '')

# ---- Load secrets ----
try:
    with open(SECRETS_FILE) as f:
        secrets = json.load(f)
except (KeyError, FileNotFoundError) as e:
    print(f'ERROR: cannot load secrets: {e}', file=sys.stderr)
    sys.exit(1)

# Bot token resolution order:
#   1. DW_BOT_TOKEN env (from notify config "botToken" — dogwatch-owned,
#      gitignored, fully independent of OpenClaw's secrets file)
#   2. "dogwatch" account in the OpenClaw secrets file
#   3. legacy "default" account
bot_token = os.environ.get('DW_BOT_TOKEN', '') or ''
if not bot_token:
    accounts = secrets['channels']['telegram']['accounts']
    bot_token = accounts.get('dogwatch', {}).get('botToken') or accounts['default']['botToken']

# Vision API key: prefer the explicit DOGWATCH_VISION_API_KEY env var (works
# for any provider). Otherwise pick the provider key matching the endpoint
# URL — OpenRouter for openrouter.ai, Google for generativelanguage — so the
# key always matches the API being called, whichever model is configured.
if not VISION_API_KEY:
    try:
        providers = secrets['models']['providers']
        if 'openrouter.ai' in VISION_API_URL:
            VISION_API_KEY = providers.get('openrouter', {}).get('apiKey', '')
        else:
            VISION_API_KEY = providers.get('google', {}).get('apiKey', '')
    except KeyError:
        pass

if not VISION_FALLBACK_API_KEY:
    try:
        providers = secrets['models']['providers']
        if 'openrouter.ai' in VISION_FALLBACK_API_URL:
            VISION_FALLBACK_API_KEY = providers.get('openrouter', {}).get('apiKey', '')
        else:
            VISION_FALLBACK_API_KEY = providers.get('google', {}).get('apiKey', '')
    except KeyError:
        pass

if not VISION_API_KEY:
    print(
        'ERROR: no vision API key configured — set DOGWATCH_VISION_API_KEY '
        'or add secrets.json models.providers.google.apiKey',
        file=sys.stderr,
    )
    sys.exit(1)

# ---- Camera config (fresh-frame fallback when an event has no snapshot) ----
NOTIFY_CONFIG = os.environ.get('DW_NOTIFY_CONFIG', '')
CAMERAS = {}
if NOTIFY_CONFIG and os.path.exists(NOTIFY_CONFIG):
    try:
        with open(NOTIFY_CONFIG) as f:
            CAMERAS = json.load(f).get('cameras', {})
    except Exception as exc:
        print(f'  WARN: cannot load cameras from {NOTIFY_CONFIG}: {exc}',
              file=sys.stderr)


def capture_fresh(camera_name):
    """Grab a clean frame NOW (RTSP via ffmpeg, HTTP ISAPI fallback).

    Used when an event landed in the status file without a usable snapshot
    (notifier debounce race, write race, or tmp cleanup). Mirrors the
    notifier's capture_snapshot so a snapshot-less digging event still gets
    vision-verified — and the dog alarm still gets its chance to fire.
    """
    cam = CAMERAS.get(camera_name)
    if not cam:
        return ''
    snap_path = f'/tmp/dogwatch_check_{camera_name}_{int(time.time())}.jpg'
    url = cam.get('snapshot_rtsp_fallback', cam.get('snapshot_url', ''))
    try:
        subprocess.run(
            ['ffmpeg', '-rtsp_transport', 'tcp', '-skip_frame', 'nokey',
             '-i', url, '-frames:v', '1', '-q:v', '2', '-update', '1',
             '-y', snap_path],
            capture_output=True, timeout=15)
        if os.path.exists(snap_path) and os.path.getsize(snap_path) > 1000:
            return snap_path
        try:
            os.remove(snap_path)
        except OSError:
            pass
    except Exception as exc:
        print(f'  capture_fresh RTSP failed for {camera_name}: {exc}',
              file=sys.stderr)
    su = cam.get('snapshot_url', '')
    if su.startswith('http://') or su.startswith('https://'):
        try:
            import requests
            from requests.auth import HTTPDigestAuth
            parsed = requests.utils.urlparse(su)
            user, pw = parsed.username, parsed.password
            clean_url = su.replace(f'{user}:{pw}@', '') if user else su
            resp = requests.get(clean_url, auth=HTTPDigestAuth(user, pw),
                                timeout=10)
            resp.raise_for_status()
            with open(snap_path, 'wb') as f:
                f.write(resp.content)
            if os.path.getsize(snap_path) > 100:
                return snap_path
        except Exception as exc:
            print(f'  capture_fresh HTTP failed for {camera_name}: {exc}',
                  file=sys.stderr)
    return ''


TG_URL = f'https://api.telegram.org/bot{bot_token}/sendMessage'

# ---- Helpers ----
def tg_send(text, parse_mode='Markdown'):
    data = urllib.parse.urlencode({
        'chat_id': CHAT_ID, 'text': text, 'parse_mode': parse_mode
    }).encode()
    try:
        req = urllib.request.Request(TG_URL, data=data, method='POST')
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True
    except Exception as e:
        print(f'  TG send error: {e}', file=sys.stderr)
        return False

def tg_send_photo(photo_path, caption):
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    boundary = '----DogWatchBoundary'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f'{CHAT_ID}\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="photo"; filename="dogwatch.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode()
    try:
        with open(photo_path, 'rb') as f:
            img_data = f.read()
    except OSError:
        return False
    body += img_data
    body += f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return True
    except Exception:
        return False

def bump_stats(key, amount=1):
    """Record a daily counter for the report; capture must never break the
    alert pipeline, so failures are swallowed."""
    try:
        # stats.py is NOT marked executable in the image, so invoke it via the
        # interpreter (same pattern as dog-alarm.sh / find-dogs-mqtt.py). Direct
        # exec silently failed with Permission denied — vision counters never
        # landed in daily-stats.json (fixed 2026-08-20).
        subprocess.run(
            [sys.executable,
             os.environ.get('DW_STATS_SCRIPT', '/app/stats.py'),
             'bump', key, str(amount)],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def vision_verify_with(image_path, api_url, model, api_key, provider_label):
    """Call one OpenAI-compatible vision endpoint to assess (a) dog presence
    and (b) whether it is digging.

    provider_label is used in log lines (e.g. 'primary' / 'fallback').

    Returns a dict {'dog': 'DOG'|'NO_DOG'|'UNCERTAIN', 'digging': bool|None}
    or None on error (API failure, rate limit, bad/truncated response)."""
    try:
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
    except OSError as e:
        print(f'  vision_verify[{provider_label}]: cannot read {image_path}: {e}', file=sys.stderr)
        return None

    prompt_text = (
        'You are analysing a backyard security snapshot to detect a dog '
        'near/under a fence and whether it is digging.\n'
        'Consider motion blur, lighting, and common false positives '
        '(leaves, shadows, wind, cars, people).\n'
        'Digging cues: head/nose lowered to the ground, front paws at '
        'the soil, a paw/scratching motion, or freshly disturbed dirt '
        'directly under the dog.\n'
        'Respond with STRICT JSON only, no prose, in exactly this form:\n'
        '{"dog": "DOG"|"NO_DOG"|"UNCERTAIN", "digging": "YES"|"NO"|"UNCERTAIN", '
        '"description": "short plain-English sentence"}\n'
        'dog = DOG if a dog is clearly or very likely present, NO_DOG if '
        'definitely not, UNCERTAIN if you cannot tell. '
        'digging = YES only if the dog appears to be digging, NO if a dog '
        'is present but not digging, UNCERTAIN otherwise.\n'
        'description = one short natural sentence (max ~15 words) saying '
        'what is actually in the frame and how many dogs — e.g. "2 dogs '
        'digging near the fence", "1 dog lying in the sun near the fence", '
        '"leaves blowing across the yard", "empty yard". Always fill it '
        'in, whatever the verdict.'
    )

    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt_text},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                },
            ],
        }],
        'max_tokens': 1024,
        'response_format': {'type': 'json_object'},
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(api_url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f'Bearer {api_key}')
    if 'openrouter.ai' in api_url:
        req.add_header('HTTP-Referer', 'https://github.com/VIDGuide/dogwatch')
        req.add_header('X-Title', 'DogWatch')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        combined = ''
        for choice in result.get('choices', []):
            msg = choice.get('message', {})
            content = msg.get('content', '')
            if isinstance(content, str):
                combined += content
            elif isinstance(content, list):
                # Some OpenAI-compatible providers return content as a list
                # of typed parts rather than a plain string.
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        combined += part.get('text', '')
        combined = combined.strip()

        # If the response was truncated (finish_reason: length) or empty,
        # treat it as an API failure rather than defaulting to UNCERTAIN
        # (which maps to "confirmed" and produces false confirmations).
        if not combined or len(combined) < 5:
            finish = result.get('choices', [{}])[0].get('finish_reason', '')
            print(f'  vision_verify[{provider_label}]: truncated/empty response ({finish}): {combined!r}', file=sys.stderr)
            return None

        dog = 'UNCERTAIN'
        digging = None
        description = ''
        # Preferred path: strict JSON response.
        try:
            parsed = json.loads(combined)
            dog = str(parsed.get('dog', 'UNCERTAIN')).upper()
            dig_raw = str(parsed.get('digging', 'UNCERTAIN')).upper()
            digging = True if dig_raw == 'YES' else (False if dig_raw == 'NO' else None)
            description = str(parsed.get('description', '') or '').strip()
        except (json.JSONDecodeError, AttributeError):
            # Fallback: keyword scan if the model didn't return clean JSON.
            up = combined.upper()
            for kw in ('NO_DOG', 'UNCERTAIN', 'DOG'):
                if kw in up:
                    dog = kw
                    break
            if '"DIGGING": "YES"' in up or 'DIGGING: YES' in up:
                digging = True
            elif '"DIGGING": "NO"' in up or 'DIGGING: NO' in up:
                digging = False
            print(f'  vision_verify[{provider_label}]: non-JSON response: {combined}', file=sys.stderr)

        if dog not in ('DOG', 'NO_DOG', 'UNCERTAIN'):
            dog = 'UNCERTAIN'
        print(f'  vision_verify[{provider_label}] OK: dog={dog} digging={digging} desc={description!r}', file=sys.stderr)
        bump_stats('vision_primary_ok' if provider_label == 'primary'
                   else 'vision_fallback_ok')
        return {'dog': dog, 'digging': digging, 'description': description}
    except Exception as e:
        print(f'  vision_verify[{provider_label}] API error: {e}', file=sys.stderr)
        return None


def vision_verify(image_path):
    """Verify a snapshot — primary provider first, OpenRouter fallback if the
    primary fails (Gemini quota/429, network, timeout, bad response). Returns
    the first successful result, or None if every configured provider fails."""
    result = vision_verify_with(
        image_path, VISION_API_URL, VISION_MODEL, VISION_API_KEY, 'primary'
    )
    if result is not None:
        return result
    if VISION_FALLBACK_API_KEY:
        print(
            f'  vision_verify: primary failed → trying fallback '
            f'({VISION_FALLBACK_MODEL} @ {VISION_FALLBACK_API_URL})',
            file=sys.stderr,
        )
        return vision_verify_with(
            image_path, VISION_FALLBACK_API_URL, VISION_FALLBACK_MODEL,
            VISION_FALLBACK_API_KEY, 'fallback',
        )
    print('  vision_verify: primary failed and no fallback key configured', file=sys.stderr)
    return None


# After the siren sounds on a digging event, re-check the same camera a
# short while later to see whether the dog was actually distracted
# (closed-loop deterrent). 0 disables the follow-up.
ALARM_FOLLOWUP_SECONDS = float(os.environ.get('DOGWATCH_ALARM_FOLLOWUP_SECONDS', '30'))


def alarm_followup(p):
    """Re-check the camera ~N seconds after a siren to verify the dog was
    distracted. Sends one follow-up photo + verdict, bumps daily stats."""
    delay = ALARM_FOLLOWUP_SECONDS
    if delay <= 0:
        return
    time.sleep(delay)
    bump_stats('alarm_followups')
    snap = capture_fresh(p.get('camera', 'camera'))
    if not snap:
        tg_send('⚠️ *Siren follow-up* — could not grab a fresh frame to '
                'check whether the dog was distracted.')
        bump_stats('alarm_followup_uncertain')
        return
    result = vision_verify(snap)
    if result is None:
        tg_send('⚠️ *Siren follow-up* — vision check failed on the fresh '
                'frame; could not confirm the dog was distracted.')
        bump_stats('alarm_followup_uncertain')
        return
    dog = result['dog']
    digging = result['digging']
    description = result.get('description', '')
    if dog == 'DOG' and digging is True:
        bump_stats('alarm_followup_still_digging')
        head = '🔊 *Siren follow-up* — dog is *still digging*!'
    elif dog == 'DOG':
        bump_stats('alarm_followup_present')
        head = '🔊 *Siren follow-up* — dog still at the fence, not digging.'
    elif dog == 'NO_DOG':
        bump_stats('alarm_followup_clear')
        head = '✅ *Siren follow-up* — dog has left the fence — siren worked.'
    else:
        bump_stats('alarm_followup_uncertain')
        head = '❓ *Siren follow-up* — could not tell from the fresh frame.'
    caption = f'{head}\n👁️ {description}' if description else head
    tg_send_photo(snap, caption)


# ---- Collect events ----
# Dedupe re-triggers: the detector can emit several ON events for the same
# camera+type within one incident burst (e.g. a dog digging for a minute).
# Each pending entry gets its own vision follow-up, so a burst previously
# sent a confirm AND then a contradictory false-alarm for the same incident
# (seen 2026-08-19: 14:56:40 confirmed digging + siren, then the 14:56:53
# re-trigger had no stored snapshot → fresh capture after the dog left →
# NO_DOG → false alarm). Keep only the first event per (camera, label)
# within the window, preferring whichever has a stored snapshot.
DEDUPE_WINDOW_SECONDS = float(os.environ.get('DOGWATCH_DEDUPE_WINDOW', '90'))
pending = []
seen = {}  # (camera, label) -> index into pending


def advance_watermark():
    """Persist the newest event ts seen so future cycles skip it."""
    if max_seen > LAST_TS:
        try:
            with open(os.environ['DW_LAST_TS_FILE'], 'w') as f:
                f.write(f'{max_seen:.6f}\n')
        except Exception:
            pass

max_seen = 0.0
try:
    with open(STATUS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                max_seen = max(max_seen, e['ts'])
                if e['ts'] > LAST_TS and e['ts'] >= CUTOFF and e['state'] == 'ON':
                    ts_local = time.strftime('%H:%M:%S', time.localtime(e['ts']))
                    topic = e['topic']
                    slug = topic.split('/')[-1]
                    snap = e.get('snapshot', '')
                    label = 'dog_at_fence' if slug == 'dog_at_fence' else 'digging' if slug == 'digging' else slug

                    ws_path = ''
                    key = (e.get('camera', 'camera'), label)
                    has_stored = bool(snap and os.path.exists(snap))
                    # Same incident burst? Only the first entry per
                    # (camera, label) gets a follow-up; a later repeat must
                    # have a stored snapshot to REPLACE a snapshot-less first
                    # entry (prefer the real frame over a fresh capture).
                    if key in seen:
                        existing = pending[seen[key]]
                        if e['ts'] - existing['ts'] < DEDUPE_WINDOW_SECONDS:
                            if has_stored and not existing['snapshot']:
                                basename = f'dogwatch_{int(e["ts"])}.jpg'
                                ws_path = os.path.join(WORKSPACE_DIR, basename)
                                shutil.copy2(snap, ws_path)
                                pending[seen[key]] = {
                                    'ts': e['ts'],
                                    'type': label,
                                    'time': ts_local,
                                    'snapshot': ws_path,
                                    'bbox': e.get('bbox'),
                                    'score': e.get('score', 0.0),
                                    'camera': e.get('camera', 'camera'),
                                }
                            else:
                                print(f'  dedupe: skip repeat {label} at '
                                      f'{ts_local} (same incident as '
                                      f'{existing["time"]})', file=sys.stderr)
                            continue
                    if not has_stored:
                        # No stored snapshot (debounce/race/cleanup) — try a
                        # fresh frame at check time so vision still runs.
                        fresh = capture_fresh(e.get('camera', 'camera'))
                        if fresh:
                            ws_path = fresh
                            print(f'  fresh capture for {label} at {ts_local}',
                                  file=sys.stderr)
                    else:
                        basename = f'dogwatch_{int(e["ts"])}.jpg'
                        ws_path = os.path.join(WORKSPACE_DIR, basename)
                        shutil.copy2(snap, ws_path)

                    entry = {
                        'ts': e['ts'],
                        'type': label,
                        'time': ts_local,
                        'snapshot': ws_path,
                        'bbox': e.get('bbox'),
                        'score': e.get('score', 0.0),
                        'camera': e.get('camera', 'camera'),
                    }
                    seen[key] = len(pending)
                    pending.append(entry)
            except (json.JSONDecodeError, KeyError):
                pass
except FileNotFoundError:
    pass

if not pending:
    advance_watermark()
    sys.exit(0)

# Write marker file (handy for debugging / external tools)
with open(MARKER_FILE, 'w') as f:
    json.dump(pending, f)

# Send initial alert
lines = []
for p in pending:
    snap_icon = ' 📸' if p['snapshot'] else ''
    lines.append(f'  • {p["type"].replace("_", " ").title()} at {p["time"]}{snap_icon}')

count = len(pending)
alert_text = (
    f'📹 *DogWatch Alert* — {count} event{"s" if count > 1 else ""} detected\n'
    + '\n'.join(lines)
    + '\n\n_Verifying with vision…_'
)
tg_send(alert_text)

# ---- Vision verify each event ----
for p in pending:
    if not p['snapshot']:
        # No snapshot and the fresh capture failed — say so explicitly
        # instead of silently dropping the event (a silent skip here is
        # exactly how a digging event could vanish and the alarm never fire).
        tg_send(
            f'⚠️ *No snapshot available* for '
            f'{p["type"].replace("_", " ").title()} at {p["time"]} — '
            f'fresh capture failed too. Vision check skipped; no alarm '
            f'decision made for this event.'
        )
        continue

    bump_stats('vision_checks')
    result = vision_verify(p['snapshot'])
    event_label = p['type'].replace('_', ' ').title()

    if result is None:
        bump_stats('vision_failed')
        # Vision API call failed (rate limit, network error, bad response,
        # etc.) — say so explicitly rather than going silent. Previously
        # this just `continue`d, so a quota exhaustion looked identical to
        # "nothing happened" from the user's perspective: the initial alert
        # + photo still arrived (unaffected — that path doesn't call vision
        # at all), but the "Verifying with vision…" promise was never
        # followed up on, with zero visible sign anything went wrong.
        tg_send(
            f'⚠️ *Vision check failed* for {event_label} at {p["time"]} — '
            f'see script logs for the API error. Detection alert above is '
            f'still valid; this only affects the confirm/false-alarm follow-up.'
        )
        continue

    verdict = result['dog']
    digging = result['digging']
    description = result.get('description', '')

    if verdict == 'DOG':
        bump_stats('vision_dog_confirmed')
        dig_line = ''
        if digging is True:
            dig_line = '\n⚠️ *DIGGING detected* — dog appears to be digging!'
        elif digging is False:
            dig_line = '\n🐾 Not digging.'
        desc_line = f'\n👁️ {description}' if description else ''
        caption = (
            f'✅ *Dog Confirmed* at {p["time"]}\n'
            f'🐕 Type: {event_label}'
            f'{desc_line}'
            f'{dig_line}'
        )
        tg_send_photo(p['snapshot'], caption)

        # Optional siren hook: vision has now CONFIRMED a dog AND that it is
        # digging — the one case Michael wants the backyard alarm for. The
        # dog-alarm script enforces its own guards (time window, replay
        # interval, enabled flag) and raises the event back to the chat.
        if digging is True and os.path.exists(os.environ.get('DW_ALARM_SCRIPT', '')):
            reason = f'vision confirmed digging — {event_label} at {p["time"]}'
            rc = None
            try:
                rc = subprocess.run(
                    [os.environ['DW_ALARM_SCRIPT'], reason], timeout=60
                ).returncode
            except Exception as e:
                print(f'  dog-alarm hook error: {e}', file=sys.stderr)
            if rc == 0:
                # Siren actually sounded — re-check the camera shortly after
                # to see whether the dog was distracted (closed loop).
                alarm_followup(p)
    elif verdict == 'NO_DOG':
        bump_stats('vision_false_alarm')
        if description:
            tg_send(
                f'❌ *False alarm* — the {event_label} at {p["time"]} — '
                f'{description}'
            )
        else:
            tg_send(
                f'❌ *False alarm* — the {event_label} at {p["time"]} '
                f'was just wind/leaves/shadow.'
            )
    elif verdict == 'UNCERTAIN':
        bump_stats('vision_uncertain')
        desc_suffix = f' ({description})' if description else ''
        tg_send(
            f'❓ *Inconclusive* — vision could not confirm or deny the '
            f'{event_label} at {p["time"]}.{desc_suffix} '
            f'Check the snapshot manually.'
        )

    time.sleep(1)

advance_watermark()

PYEOF

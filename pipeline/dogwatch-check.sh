#!/bin/bash
# DogWatch event checker — fully event-driven
# Checks for recent dog events, sends quick Telegram ping,
# runs Gemini vision verification via OpenRouter on each snapshot,
# and sends confirmation/false-alarm follow-up.
# Silent exit on no events — no model call, no noise.
STATUS_FILE="/tmp/dogwatch-events.jsonl"
CUTOFF=$(date +%s -d "4 minutes ago")
WORKSPACE_SNAP_DIR="/home/misaunders/.openclaw/workspace/dogwatch_snaps"
MARKER_FILE="/tmp/dogwatch-pending.jsonl"
SECRETS_FILE="$HOME/.openclaw/secrets.json"
# Chat id is loaded from (in order): DOGWATCH_CHAT_ID env, the notify config
# file's "chat_id", so it is not hardcoded in this (publicly-committed) script.
NOTIFY_CONFIG="${DOGWATCH_NOTIFY_CONFIG:-$(dirname "$(readlink -f "$0")")/dogwatch-notify.config.json}"
CHAT_ID="${DOGWATCH_CHAT_ID:-}"
if [ -z "$CHAT_ID" ] && [ -f "$NOTIFY_CONFIG" ]; then
  CHAT_ID=$(python3 -c "import json,sys; print(json.load(open('$NOTIFY_CONFIG')).get('chat_id',''))" 2>/dev/null)
fi

mkdir -p "$WORKSPACE_SNAP_DIR"
rm -f "$MARKER_FILE"

if [ ! -f "$STATUS_FILE" ]; then
  exit 0
fi

# Pass shell vars to Python as env vars so we don't fight with heredoc quoting
export DW_CUTOFF="$CUTOFF"
export DW_WORKSPACE_DIR="$WORKSPACE_SNAP_DIR"
export DW_MARKER_FILE="$MARKER_FILE"
export DW_SECRETS_FILE="$SECRETS_FILE"
export DW_CHAT_ID="$CHAT_ID"
export DW_STATUS_FILE="$STATUS_FILE"

python3 << 'PYEOF'
import json, time, sys, shutil, os, urllib.request, urllib.parse, base64

CUTOFF = float(os.environ['DW_CUTOFF'])
WORKSPACE_DIR = os.environ['DW_WORKSPACE_DIR']
MARKER_FILE = os.environ['DW_MARKER_FILE']
SECRETS_FILE = os.path.expanduser(os.environ['DW_SECRETS_FILE'])
CHAT_ID = os.environ['DW_CHAT_ID']
STATUS_FILE = os.environ['DW_STATUS_FILE']

# ---- Load secrets ----
try:
    with open(SECRETS_FILE) as f:
        secrets = json.load(f)
    bot_token = secrets['channels']['telegram']['accounts']['default']['botToken']
    google_api_key = secrets['models']['providers']['google']['apiKey']
except (KeyError, FileNotFoundError) as e:
    print(f'ERROR: cannot load secrets: {e}', file=sys.stderr)
    sys.exit(1)

TG_URL = f'https://api.telegram.org/bot{bot_token}/sendMessage'
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={google_api_key}'

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

def vision_verify(image_path):
    """Call Google Gemini to assess (a) dog presence and (b) whether it is digging.

    Returns a dict {'dog': 'DOG'|'NO_DOG'|'UNCERTAIN', 'digging': bool|None}
    or None on error."""
    try:
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
    except OSError as e:
        print(f'  vision_verify: cannot read {image_path}: {e}', file=sys.stderr)
        return None

    payload = {
        'contents': [{
            'parts': [
                {
                    'text': (
                        'You are analysing a backyard security snapshot to detect a dog '
                        'near/under a fence and whether it is digging.\n'
                        'Consider motion blur, lighting, and common false positives '
                        '(leaves, shadows, wind, cars, people).\n'
                        'Digging cues: head/nose lowered to the ground, front paws at '
                        'the soil, a paw/scratching motion, or freshly disturbed dirt '
                        'directly under the dog.\n'
                        'Respond with STRICT JSON only, no prose, in exactly this form:\n'
                        '{"dog": "DOG"|"NO_DOG"|"UNCERTAIN", "digging": "YES"|"NO"|"UNCERTAIN"}\n'
                        'dog = DOG if a dog is clearly or very likely present, NO_DOG if '
                        'definitely not, UNCERTAIN if you cannot tell. '
                        'digging = YES only if the dog appears to be digging, NO if a dog '
                        'is present but not digging, UNCERTAIN otherwise.'
                    ),
                },
                {
                    'inline_data': {
                        'mime_type': 'image/jpeg',
                        'data': b64,
                    },
                },
            ],
        }],
        'generationConfig': {
            'maxOutputTokens': 40,
            'responseMimeType': 'application/json',
        },
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(GEMINI_URL, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        texts = []
        for c in result.get('candidates', []):
            content = c.get('content', {})
            for p in content.get('parts', []):
                t = p.get('text', '')
                if t:
                    texts.append(t)
        combined = ' '.join(texts).strip()

        dog = 'UNCERTAIN'
        digging = None
        # Preferred path: strict JSON response.
        try:
            parsed = json.loads(combined)
            dog = str(parsed.get('dog', 'UNCERTAIN')).upper()
            dig_raw = str(parsed.get('digging', 'UNCERTAIN')).upper()
            digging = True if dig_raw == 'YES' else (False if dig_raw == 'NO' else None)
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
            print(f'  vision_verify: non-JSON response: {combined}', file=sys.stderr)

        if dog not in ('DOG', 'NO_DOG', 'UNCERTAIN'):
            dog = 'UNCERTAIN'
        return {'dog': dog, 'digging': digging}
    except Exception as e:
        print(f'  vision_verify API error: {e}', file=sys.stderr)
        return None

# ---- Collect events ----
pending = []

try:
    with open(STATUS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e['ts'] >= CUTOFF and e['state'] == 'ON':
                    ts_local = time.strftime('%H:%M:%S', time.localtime(e['ts']))
                    topic = e['topic']
                    slug = topic.split('/')[-1]
                    snap = e.get('snapshot', '')
                    label = 'dog_at_fence' if slug == 'dog_at_fence' else 'digging' if slug == 'digging' else slug

                    ws_path = ''
                    if snap and os.path.exists(snap):
                        basename = f'dogwatch_{int(e["ts"])}.jpg'
                        ws_path = os.path.join(WORKSPACE_DIR, basename)
                        shutil.copy2(snap, ws_path)

                    pending.append({
                        'type': label,
                        'time': ts_local,
                        'snapshot': ws_path,
                        'bbox': e.get('bbox'),
                        'score': e.get('score', 0.0),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
except FileNotFoundError:
    pass

if not pending:
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
        continue

    result = vision_verify(p['snapshot'])
    event_label = p['type'].replace('_', ' ').title()

    if result is None:
        continue

    verdict = result['dog']
    digging = result['digging']

    if verdict == 'DOG' or verdict == 'UNCERTAIN':
        dig_line = ''
        if digging is True:
            dig_line = '\n⚠️ *DIGGING detected* — dog appears to be digging!'
        elif digging is False:
            dig_line = '\n🐾 Not digging.'
        caption = (
            f'✅ *Dog Confirmed* at {p["time"]}\n'
            f'🐕 Type: {event_label}'
            f'{dig_line}'
        )
        tg_send_photo(p['snapshot'], caption)
    elif verdict == 'NO_DOG':
        tg_send(
            f'❌ *False alarm* — the {event_label} at {p["time"]} '
            f'was just wind/leaves/shadow.'
        )

    time.sleep(1)

PYEOF

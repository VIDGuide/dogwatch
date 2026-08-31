#!/bin/bash
# dog-alarm.sh — sound the "Dog Alarm" siren via Home Assistant
#
# Optional add-on to the DogWatch alert pipeline. When configured, it is
# called by pipeline/dogwatch-check.sh after vision verification confirms a
# dog is digging, and can also be triggered manually:
#
#   /app/dog-alarm.sh "vision confirmed digging"        # automatic (or any reason)
#   /app/dog-alarm.sh --manual "manual request"          # manual trigger
#
# Guard rails (all configurable, see README "Dog Alarm (Home Assistant)"):
#   * Time window — by default the alarm can NEVER sound before 07:00 or
#     after 20:00 local time (TZ from the container/host).
#   * Replay guard — at most one sound per 60 seconds (default), tracked in
#     a persistent state file so it survives container restarts.
#   * Config gate — if the "alarm" section is absent or enabled=false, the
#     script does nothing and exits silently (exit 3), so the feature is
#     fully optional and safe to leave installed-but-unconfigured.
#
# Every time the alarm is actually sounded, a Telegram message is sent to
# the configured chat ("raise the event back to the channel"). Blocked
# automatic attempts only log; blocked MANUAL attempts also message, so a
# human asking for the alarm gets a clear reason it didn't fire.
#
# Home Assistant access:
#   * ha_token          — long-lived access token (Bearer). Recommended for
#                         most setups: create in HA UI → Profile → Security
#                         → Long-lived access tokens.
#   * ha_refresh_token  — a HA refresh token; the script exchanges it for a
#                         short-lived access token on every run
#                         (POST /auth/token). Use this when long-lived
#                         tokens are unavailable/revoked in your instance.
#   Either one is enough; env overrides (DOGWATCH_HA_TOKEN /
#   DOGWATCH_HA_REFRESH_TOKEN) win over the config file.
#
# Exit codes:
#   0  sounded
#   2  blocked — outside the allowed time window
#   3  disabled or not configured (silent)
#   4  blocked — replay guard (too soon since last sound)
#   5  Home Assistant / configuration error (sound attempted but failed)

set -u

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
NOTIFY_CONFIG="${DOGWATCH_NOTIFY_CONFIG:-$SCRIPT_DIR/dogwatch-notify.config.json}"

log() { echo "dog-alarm: $*"; }

# Daily stats capture (per-day counters for the report) — never fatal.
#
# Defined up here, not further down: bash resolves functions at execution time,
# so the two early-exit paths below (missing config, unreadable config) used to
# run before the definition and emit "bump_stats: command not found" — meaning
# the alarm_errors counter was never incremented for the two most interesting
# failures. Exit codes were unaffected, which is why it went unnoticed.
bump_stats() {
  python3 "$SCRIPT_DIR/stats.py" bump "$1" >/dev/null 2>&1 || true
}

if [ ! -f "$NOTIFY_CONFIG" ]; then
  echo "dog-alarm: notify config not found: $NOTIFY_CONFIG" >&2
  bump_stats alarm_errors
  exit 5
fi

# ---- Resolve config (alarm section + chat/bot for the event message) ----
CFG_JSON=$(python3 - "$NOTIFY_CONFIG" <<'PYEOF'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception as e:
    print(f'ERROR reading config: {e}', file=sys.stderr)
    sys.exit(5)
alarm = c.get('alarm') or {}
out = {
    'enabled': bool(alarm.get('enabled', False)),
    'ha_url': alarm.get('ha_url', 'http://localhost:8123'),
    'ha_token': alarm.get('ha_token', ''),
    'ha_refresh_token': alarm.get('ha_refresh_token', ''),
    'entity_id': alarm.get('entity_id', 'siren.dog_alarm'),
    'window_start': alarm.get('window_start', '07:00'),
    'window_end': alarm.get('window_end', '20:00'),
    'min_interval_sec': int(alarm.get('min_interval_sec', 60)),
    'state_file': alarm.get('state_file', ''),
    'notify_chat': bool(alarm.get('notify_chat', True)),
    'chat_id': str(c.get('chat_id', '')),
    'bot_token': str(c.get('botToken', '')),
}
print(json.dumps(out))
PYEOF
) || { bump_stats alarm_errors; exit 5; }

get_cfg() { echo "$CFG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }

ALARM_ENABLED=$(get_cfg enabled)
HA_URL="${DOGWATCH_HA_URL:-$(get_cfg ha_url)}"
HA_TOKEN="${DOGWATCH_HA_TOKEN:-$(get_cfg ha_token)}"
HA_REFRESH_TOKEN="${DOGWATCH_HA_REFRESH_TOKEN:-$(get_cfg ha_refresh_token)}"
ENTITY_ID="${DOGWATCH_ALARM_ENTITY:-$(get_cfg entity_id)}"
WINDOW_START="${DOGWATCH_ALARM_WINDOW_START:-$(get_cfg window_start)}"
WINDOW_END="${DOGWATCH_ALARM_WINDOW_END:-$(get_cfg window_end)}"
MIN_INTERVAL="${DOGWATCH_ALARM_MIN_INTERVAL:-$(get_cfg min_interval_sec)}"
STATE_FILE="${DOGWATCH_ALARM_STATE_FILE:-$(get_cfg state_file)}"
NOTIFY_CHAT="$(get_cfg notify_chat)"
CHAT_ID="${DOGWATCH_CHAT_ID:-$(get_cfg chat_id)}"
BOT_TOKEN="${DOGWATCH_BOT_TOKEN:-$(get_cfg bot_token)}"

if [ -z "$STATE_FILE" ]; then
  STATE_FILE="${NOTIFY_CONFIG%.json}.alarm.state"
fi

MANUAL=0
if [ "${1:-}" = "--manual" ]; then
  MANUAL=1
  shift
fi
REASON="${*:-}"
[ -n "$REASON" ] || REASON="(no reason given)"

# ---- Bot token fallback (same pattern as dogwatch-check.sh) ----
if [ -z "$BOT_TOKEN" ] && [ -f "$HOME/.openclaw/secrets.json" ]; then
  BOT_TOKEN=$(python3 - "$HOME/.openclaw/secrets.json" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
    a = s['channels']['telegram']['accounts']
    print(a.get('dogwatch', {}).get('botToken') or a['default']['botToken'])
except Exception:
    print('')
PYEOF
)
fi

tg_msg() {
  [ "$NOTIFY_CHAT" = "True" ] || return 0
  [ -n "$CHAT_ID" ] || return 0
  [ -n "$BOT_TOKEN" ] || return 0
  DW_CHAT_ID="$CHAT_ID" DW_BOT_TOKEN="$BOT_TOKEN" python3 - "$1" <<'PYEOF'
import os, sys, urllib.request, urllib.parse
text = sys.argv[1]
data = urllib.parse.urlencode({
    'chat_id': os.environ['DW_CHAT_ID'],
    'text': text,
    'parse_mode': 'Markdown',
}).encode()
try:
    urllib.request.urlopen(
        urllib.request.Request(
            f"https://api.telegram.org/bot{os.environ['DW_BOT_TOKEN']}/sendMessage",
            data=data,
        ),
        timeout=10,
    )
except Exception as e:
    print(f'  TG send error: {e}', file=sys.stderr)
PYEOF
}

# ---- Guard 1: enabled? ----
if [ "$ALARM_ENABLED" != "True" ]; then
  log "disabled or not configured — exiting silently"
  exit 3
fi

# ---- Guard 2: time window (local time; TZ from container/host) ----
NOW_NUM=$(( 10#$(date +%H%M) ))
START_NUM=$(( 10#$(echo "$WINDOW_START" | tr -d ':' | sed 's/^0*//; s/^$/0/') ))
END_NUM=$(( 10#$(echo "$WINDOW_END" | tr -d ':' | sed 's/^0*//; s/^$/0/') ))
if [ "$START_NUM" -le "$END_NUM" ]; then
  IN_WINDOW=$(( NOW_NUM >= START_NUM && NOW_NUM <= END_NUM ))
else
  # window wraps midnight (e.g. 22:00–06:00)
  IN_WINDOW=$(( NOW_NUM >= START_NUM || NOW_NUM <= END_NUM ))
fi
if [ "$IN_WINDOW" -ne 1 ]; then
  log "blocked: outside allowed window $WINDOW_START–$WINDOW_END (now $(date +%H:%M:%S))"
  if [ "$MANUAL" -eq 1 ]; then
    tg_msg "🔕 *Dog Alarm not sounded* — \`$REASON\`
Outside allowed hours ($WINDOW_START–$WINDOW_END local)."
  fi
  bump_stats alarm_blocked_window
  exit 2
fi

# ---- Guard 3: replay guard (min interval between sounds) ----
NOW_EPOCH=$(date +%s)
mkdir -p "$(dirname "$STATE_FILE")"
LAST_EPOCH=0
[ -f "$STATE_FILE" ] && LAST_EPOCH=$(tr -dc '0-9' < "$STATE_FILE" 2>/dev/null)
if [ -n "$LAST_EPOCH" ] && [ $(( NOW_EPOCH - LAST_EPOCH )) -lt "$MIN_INTERVAL" ]; then
  log "blocked: replay guard — last sounded $(date -d @$LAST_EPOCH '+%H:%M:%S'), min interval ${MIN_INTERVAL}s"
  if [ "$MANUAL" -eq 1 ]; then
    tg_msg "🔕 *Dog Alarm not sounded* — \`$REASON\`
Replay guard: last sounded at $(date -d @$LAST_EPOCH '+%H:%M:%S'), min interval ${MIN_INTERVAL}s."
  fi
  bump_stats alarm_blocked_replay
  exit 4
fi

# ---- Obtain HA bearer token ----
if [ -z "$HA_TOKEN" ] && [ -z "$HA_REFRESH_TOKEN" ]; then
  log "ERROR: no HA credential configured (alarm.ha_token or alarm.ha_refresh_token)"
  bump_stats alarm_errors
  exit 5
fi
BEARER="$HA_TOKEN"
if [ -z "$BEARER" ]; then
  BEARER=$(python3 - "$HA_URL" "$HA_REFRESH_TOKEN" <<'PYEOF' | tr -d '\n'
import json, sys, urllib.request, urllib.parse
url, rt = sys.argv[1], sys.argv[2]
body = urllib.parse.urlencode({'grant_type': 'refresh_token', 'refresh_token': rt}).encode()
req = urllib.request.Request(url.rstrip('/') + '/auth/token', data=body)
req.add_header('Content-Type', 'application/x-www-form-urlencoded')
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print(json.loads(r.read()).get('access_token', ''))
except Exception as e:
    print(f'ERROR {e}', file=sys.stderr)
    sys.exit(1)
PYEOF
) || { bump_stats alarm_errors; exit 5; }
  if [ -z "$BEARER" ] || [ "${BEARER#ERROR}" != "$BEARER" ]; then
    log "ERROR: HA token exchange failed ($BEARER)"
    bump_stats alarm_errors
    exit 5
  fi
fi

# ---- Sound the alarm ----
HA_STATUS=$(python3 - "$HA_URL" "$ENTITY_ID" "$BEARER" <<'PYEOF' | tr -d '\n'
import json, sys, urllib.request
url, entity, bearer = sys.argv[1], sys.argv[2], sys.argv[3]
payload = json.dumps({'entity_id': entity}).encode()
req = urllib.request.Request(url.rstrip('/') + '/api/services/siren/turn_on', data=payload, method='POST')
req.add_header('Content-Type', 'application/json')
req.add_header('Authorization', f'Bearer {bearer}')
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(f'HTTP {e.code}: {e.read()[:200].decode(errors="replace")}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'ERROR {e}', file=sys.stderr)
    sys.exit(1)
PYEOF
)
if [ -z "$HA_STATUS" ]; then
  log "ERROR: siren.turn_on failed (no response)"
  tg_msg "⚠️ *Dog alarm failed* — \`$REASON\` — Home Assistant returned nothing."
  bump_stats alarm_errors
  exit 5
fi
if [ "${HA_STATUS#HTTP}" != "$HA_STATUS" ] || [ "${HA_STATUS#ERROR}" != "$HA_STATUS" ]; then
  log "ERROR: siren.turn_on failed: $HA_STATUS"
  tg_msg "⚠️ *Dog alarm failed* — \`$REASON\` — Home Assistant: \`$HA_STATUS\`"
  bump_stats alarm_errors
  exit 5
fi

# Record sound time (replay guard state)
echo "$NOW_EPOCH" > "$STATE_FILE"
log "sounded at $(date '+%H:%M:%S') (reason: $REASON) — HA status $HA_STATUS"

# Raise the event back to the chat
tg_msg "🔔 *Dog Alarm sounded* — \`$REASON\`
$(date '+%H:%M:%S') local · entity \`$ENTITY_ID\`"
bump_stats alarm_sounds
exit 0

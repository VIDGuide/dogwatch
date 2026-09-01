#!/bin/bash
# dogwatch-health-sweep-cron.sh — host cron entrypoint for the hourly health sweep.
#
# Runs the headless check (pipeline/dogwatch-health-sweep.sh) and, on anomaly,
# sends ONE Telegram alert to the owner chat via the dogwatch bot token.
#
# Zero OpenClaw involvement: this runs from host crontab, so it keeps working
# even if the gateway/LLM layer is down, a model call hangs, or an isolated
# session process dies. The healthy path is fully silent and cannot time out.
#
# Usage:
#   dogwatch-health-sweep-cron.sh          # normal (silent when healthy)
#   dogwatch-health-sweep-cron.sh --test   # send a test alert to verify delivery
#
# Bot token resolution (same order as the notifier — do not regress):
#   1) botToken in dogwatch-notify.config.json   (gitignored, contains creds)
#   2) TELEGRAM_BOT_TOKEN env
set -u

REPO="${DOGWATCH_REPO:-/home/misaunders/source/dogTracker}"
SWEEP="$REPO/pipeline/dogwatch-health-sweep.sh"
CHAT_ID="${DOGWATCH_CHAT_ID:-999234597}"
LOG=/tmp/dogwatch-health-sweep.log

if [ "${1:-}" = "--test" ]; then
  # Verify the alert path without needing a real anomaly.
  out="TEST ALERT — health sweep send path verified, no anomaly."
  rc=1
else
  out="$(bash "$SWEEP" 2>&1)"
  rc=$?
fi

# Healthy — silent, nothing to log.
if [ $rc -eq 0 ]; then
  exit 0
fi

# Resolve bot token: config botToken -> env.
token=""
cfg="$REPO/dogwatch-notify.config.json"
if [ -f "$cfg" ]; then
  token="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("botToken",""))' "$cfg" 2>/dev/null)"
fi
[ -z "$token" ] && token="${TELEGRAM_BOT_TOKEN:-}"

if [ -z "$token" ]; then
  echo "$(date '+%F %T') ANOMALY but no bot token resolved; raw output:" >> "$LOG"
  echo "$out" >> "$LOG"
  exit 1
fi

msg="⚠️ dogWatch health sweep — ANOMALY
$(echo "$out" | head -c 1500)"

curl -s -X POST "https://api.telegram.org/bot${token}/sendMessage" \
  -d chat_id="$CHAT_ID" -d text="$msg" >/dev/null 2>&1

echo "$(date '+%F %T') anomaly sent (rc=$rc):" >> "$LOG"
echo "$out" >> "$LOG"
exit 0

#!/bin/bash
# dogwatch-daily-export.sh — host cron wrapper for the Daily Dog Report data.
#
# Runs 5 minutes before the n8n 08:00 report (host crontab: 55 7 * * *).
#   1) Detector container exports yesterday's events (events.db → clips/daily-events.json)
#   2) Notifier daily stats (notify_workspace/daily-stats.json) are copied into
#      the clips dir so the n8n container (/mnt/clips, read-only) can read
#      both files side by side.
#
# Non-fatal: the report degrades gracefully if either piece is missing.
set -u
REPO="${DOGWATCH_REPO:-/home/misaunders/source/dogTracker}"

docker exec dogwatch python3 /app/export-daily-events.py >/dev/null 2>&1 \
  || echo "dogwatch-daily-export: detector export failed" >&2

if [ -f "$REPO/notify_workspace/daily-stats.json" ]; then
  cp "$REPO/notify_workspace/daily-stats.json" "$REPO/clips/daily-stats.json" \
    && echo "dogwatch-daily-export: stats copied to clips/" \
    || echo "dogwatch-daily-export: stats copy failed" >&2
else
  echo "dogwatch-daily-export: no daily-stats.json yet (first run?)" >&2
fi

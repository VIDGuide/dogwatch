#!/bin/bash
# Run the notifier (long-lived MQTT subscriber) in background, and a simple
# loop that periodically runs the vision-verification check script (replacing
# the host crontab entry).

# Start the check script loop in the background (every 5 minutes, matching
# the original cron schedule).
(
  sleep 30  # wait for notifier to connect and settle
  while true; do
    /app/dogwatch-check.sh 2>&1 | head -50
    sleep 300
  done
) &

# Run the notifier as PID 1 (foreground) so Docker sees its exit.
exec python -u /app/dogwatch-notify.py

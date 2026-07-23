#!/bin/bash
# Run the notifier (long-lived MQTT subscriber) in background, and a simple
# loop that periodically runs the vision-verification check script (replacing
# the host crontab entry).

# Start the check script loop in the background (every 5 minutes, matching
# the original cron schedule).  Output goes to a persistent log file so
# API errors don't vanish into the docker log noise.
(
  sleep 30  # wait for notifier to connect and settle
  while true; do
    /app/dogwatch-check.sh >> /var/log/dogwatch-check.log 2>&1
    echo "[$(date '+%H:%M:%S')] Check cycle complete" >> /var/log/dogwatch-check.log
    sleep 300
  done
) &

# Run the notifier as PID 1 (foreground) so Docker sees its exit.
exec python -u /app/dogwatch-notify.py

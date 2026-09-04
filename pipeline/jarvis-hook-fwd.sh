#!/bin/bash
# jarvis-hook-fwd.sh — host-side forwarder for the DogWatch → jarvis announce hook.
#
# The OpenClaw gateway binds 127.0.0.1:18789 (loopback ONLY), which Docker
# containers cannot reach. This socat instance listens on the docker bridge
# gateway (172.17.0.1:18789) and forwards to the host gateway, so the
# dogwatch-notify container can POST vision-confirmed digging facts to the
# OpenClaw hooks rail (agentId=jarvis).
#
# Intended to run on the host via crontab @reboot (and safe to run manually /
# repeatedly — it exits silently if the forwarder is already listening).
set -u

BIND="${DOGWATCH_FWD_BIND:-172.17.0.1}"
PORT="${DOGWATCH_FWD_PORT:-18789}"
TARGET="${DOGWATCH_FWD_TARGET:-127.0.0.1:18789}"
LOG="${DOGWATCH_FWD_LOG:-/tmp/jarvis-hook-fwd.log}"
PIDFILE="${DOGWATCH_FWD_PIDFILE:-/tmp/jarvis-hook-fwd.pid}"

# Already listening? Nothing to do (covers repeated cron/manual runs).
if nc -z "$BIND" "$PORT" >/dev/null 2>&1; then
  exit 0
fi

# Stale pidfile guard: if recorded pid is still alive, assume it owns the port
# in a race and back off.
if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE" 2>/dev/null || echo 0)
  if kill -0 "$old" 2>/dev/null; then
    exit 0
  fi
  rm -f "$PIDFILE"
fi

nohup socat "TCP-LISTEN:${PORT},bind=${BIND},reuseaddr,fork" "TCP:${TARGET}" \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Give it a moment, then confirm it actually came up.
sleep 1
if nc -z "$BIND" "$PORT" >/dev/null 2>&1; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') forwarder up: ${BIND}:${PORT} -> ${TARGET}" >> "$LOG"
else
  echo "$(date '+%Y-%m-%d %H:%M:%S') forwarder FAILED to start" >> "$LOG"
fi

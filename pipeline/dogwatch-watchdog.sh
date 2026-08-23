#!/usr/bin/env bash
# dogwatch-watchdog.sh — keep the dogwatch stack alive.
#
# Checks the three containers the system depends on:
#   mosquitto       (MQTT broker, standalone)
#   dogwatch        (Coral TPU detector, compose service "dogwatch")
#   dogwatch-notify (alerts + vision check loop, compose service "notifier")
#
# Handles two failure modes observed after the 2026-08-23 reboot:
#   1. stopped containers that restart policies (unless-stopped/always)
#      didn't bring back after an unclean daemon shutdown
#   2. the "zombie" state: docker metadata says Running but no containerd
#      task exists (docker top fails with "no containerd Task set") —
#      metadata survived the reboot, the process didn't
#
# Runs as misaunders via dogwatch-watchdog.timer (every 2 min).
# Log: /tmp/dogwatch-watchdog.log (matches dogwatch daily-export convention).

set -u

LOG=/tmp/dogwatch-watchdog.log
COMPOSE_DIR=/home/misaunders/source/dogTracker
MAX_LOG_BYTES=102400   # ~100KB, then trim

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

trim_log() {
  if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
    tail -n 400 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  fi
}

# container_state <name> -> running | zombie | <status> | missing
container_state() {
  local name=$1 status
  status=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)
  if [ "$status" != "running" ]; then
    [ -z "$status" ] && echo missing || echo "$status"
    return
  fi
  # Metadata says running — verify a real process exists behind it.
  if docker top "$name" >/dev/null 2>&1; then
    echo running
  else
    echo zombie
  fi
}

# ensure_container <name> <compose_service_or_empty>
ensure_container() {
  local name=$1 compose_svc=$2 st
  st=$(container_state "$name")
  case "$st" in
    running) return 0 ;;
    zombie)
      if [ -n "$compose_svc" ]; then
        log "FIX: $name zombie (no task) — recreating via compose"
        docker rm -f "$name" >/dev/null 2>&1
        (cd "$COMPOSE_DIR" && docker compose up -d "$compose_svc") >>"$LOG" 2>&1 \
          && log "OK: $name recreated" || log "ERR: $name recreate failed"
      else
        log "FIX: $name zombie (no task) — docker restart"
        docker restart "$name" >/dev/null 2>&1 \
          && log "OK: $name restarted" || log "ERR: $name restart failed"
      fi
      ;;
    *)
      log "FIX: $name state=$st — docker start"
      docker start "$name" >/dev/null 2>&1 \
        && log "OK: $name started" || log "ERR: $name start failed"
      ;;
  esac
}

trim_log

# Order matters: broker first, then detector, then notifier.
ensure_container mosquitto ""
ensure_container dogwatch dogwatch
ensure_container dogwatch-notify notifier

# Summary to stdout (lands in the systemd journal too).
states=$(for n in mosquitto dogwatch dogwatch-notify; do
  printf '%s=%s ' "$n" "$(container_state "$n")"
done)
echo "dogwatch-watchdog: $states"

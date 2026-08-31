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
#
# Env overrides: DOGWATCH_COMPOSE_DIR (where docker-compose.yml lives),
# DOGWATCH_WATCHDOG_LOG, DOGWATCH_WATCHDOG_STATE_DIR,
# DOGWATCH_HEALTH_RESTART_MIN_INTERVAL.

set -u

LOG="${DOGWATCH_WATCHDOG_LOG:-/tmp/dogwatch-watchdog.log}"
# Overridable rather than a hardcoded home directory: this script is run by a
# systemd timer as a specific user, and the baked-in /home/<user> path meant
# anyone else deploying it (or the same user after a move) got a silent
# failure — every `docker compose` call runs in the wrong directory and the
# watchdog reports nothing to restart while the stack stays down.
COMPOSE_DIR="${DOGWATCH_COMPOSE_DIR:-/home/misaunders/source/dogTracker}"
MAX_LOG_BYTES=102400   # ~100KB, then trim

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

trim_log() {
  if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
    tail -n 400 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  fi
}

HEALTH_STATE_DIR="${DOGWATCH_WATCHDOG_STATE_DIR:-/tmp}"
# Don't let a health-triggered restart become a restart loop. A camera that is
# genuinely unplugged will report unhealthy indefinitely, and hammering the
# container every 2 minutes would help nobody.
HEALTH_RESTART_MIN_INTERVAL="${DOGWATCH_HEALTH_RESTART_MIN_INTERVAL:-900}"

# container_state <name> -> running | zombie | unhealthy | <status> | missing
container_state() {
  local name=$1 status health
  status=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)
  if [ "$status" != "running" ]; then
    [ -z "$status" ] && echo missing || echo "$status"
    return
  fi
  # Metadata says running — verify a real process exists behind it.
  if ! docker top "$name" >/dev/null 2>&1; then
    echo zombie
    return
  fi
  # Running with a real process, but is it doing anything?
  #
  # This is the blind spot this script had: it could only see a container that
  # had stopped or lost its task. It could not see the detector's actual failure
  # mode — process alive, frame grabber wedged, nothing being watched. The
  # container now reports that via HEALTHCHECK (see healthcheck.py), so consume
  # it. Containers without a healthcheck report an empty string and are treated
  # as running, exactly as before.
  health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' \
    "$name" 2>/dev/null)
  if [ "$health" = "unhealthy" ]; then
    echo unhealthy
    return
  fi
  echo running
}

# health_restart_allowed <name> -> 0 if enough time has passed since the last
# health-triggered restart of this container.
health_restart_allowed() {
  local name=$1 stamp last now
  stamp="$HEALTH_STATE_DIR/dogwatch-watchdog-health-$name.stamp"
  now=$(date +%s)
  last=$(cat "$stamp" 2>/dev/null | tr -dc '0-9')
  if [ -n "$last" ] && [ $((now - last)) -lt "$HEALTH_RESTART_MIN_INTERVAL" ]; then
    return 1
  fi
  echo "$now" > "$stamp" 2>/dev/null
  return 0
}

# ensure_container <name> <compose_service_or_empty>
ensure_container() {
  local name=$1 compose_svc=$2 st
  st=$(container_state "$name")
  case "$st" in
    running) return 0 ;;
    unhealthy)
      # Docker itself does not restart unhealthy containers (only Swarm does),
      # so this is the piece that closes the loop.
      if health_restart_allowed "$name"; then
        log "FIX: $name reports unhealthy — restarting"
        docker inspect -f '{{range .State.Health.Log}}{{.Output}}{{end}}' "$name" \
          2>/dev/null | tail -n 3 >> "$LOG"
        docker restart "$name" >/dev/null 2>&1 \
          && log "OK: $name restarted (unhealthy)" \
          || log "ERR: $name restart failed (unhealthy)"
      else
        log "SKIP: $name unhealthy but restarted within the last ${HEALTH_RESTART_MIN_INTERVAL}s"
      fi
      ;;
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

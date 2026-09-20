#!/usr/bin/env bash
# dogwatch-health-sweep.sh — headless health sweep for the dogwatch stack.
#
# Runs the same checks the old LLM-driven hourly sweep did, but as a plain
# shell script: the healthy path costs zero model calls and cannot time out.
# Used as the trigger script for the dogwatch-health-sweep cron job — cron
# evaluates this when the job is due and only fires the alert payload when
# this returns fire:true (i.e. when something is actually wrong).
#
# Prints HEALTHY, or ANOMALY: <detail> lines (one per problem).
# Exit 0 when healthy, non-zero when something is wrong.
#
# Log: none — intentionally silent when healthy (matches watchdog convention).

set -u

PROBLEMS=()

# 1. Containers up + detector healthy
container_state="$(docker ps --filter name=dogwatch --format '{{.Names}} | {{.Status}}' 2>&1)"
echo "$container_state" | grep -q '^dogwatch-notify | Up' || PROBLEMS+=("notifier not Up: $(echo "$container_state" | grep '^dogwatch-notify' || echo 'missing')")
echo "$container_state" | grep -q '^dogwatch | Up' || PROBLEMS+=("detector not Up: $(echo "$container_state" | grep '^dogwatch |' || echo 'missing')")
echo "$container_state" | grep -q '^dogwatch | Up.*(healthy)' || PROBLEMS+=("detector not healthy")

# 2. Healthcheck
hc="$(docker exec dogwatch python /app/healthcheck.py 2>&1)" || hc="exit $?"
echo "$hc" | grep -q '^ok:' || PROBLEMS+=("healthcheck: $(echo "$hc" | tail -1)")

# 3. Heartbeat — staleness and publishing flags
hb="$(docker exec dogwatch cat /tmp/dogwatch-heartbeat.json 2>/dev/null)"
echo "$hb" | grep -q '"stale": false' || PROBLEMS+=("heartbeat missing/stale")
echo "$hb" | grep -q '"publishing": true' || PROBLEMS+=("camera not publishing")

# 4. Check-loop liveness: state file mtime within 10 min. The state file's
#    *content* ts is the last-event watermark (advance_watermark writes max
#    seen ts on both exit paths), so it goes stale whenever no events occur —
#    that is normal. mtime advances every 300s cycle regardless.
state_mtime="$(docker exec dogwatch-notify python3 -c 'import os;print(int(os.path.getmtime("/tmp/dogwatch-check-state.json")))' 2>/dev/null)"
if [ -z "$state_mtime" ]; then
  PROBLEMS+=("check-state missing")
else
  age=$(( $(date +%s) - state_mtime ))
  [ "$age" -le 600 ] || PROBLEMS+=("check-state stale (${age}s old)")
fi

# 5. Sustained check-loop errors (transient single 429s are normal)
err_count="$(docker exec dogwatch-notify sh -c 'grep -cE "Traceback|previous check still running" /var/log/dogwatch-check.log 2>/dev/null || true' 2>/dev/null | tail -1)"
[ "${err_count:-0}" -lt 3 ] || PROBLEMS+=("check-log: $err_count error lines")

# 6. MQTT availability — main camera at dogwatch/availability, rear-east at
#    dogwatch/rear-east/availability (both retained, arrive immediately)
avail="$(timeout 8 mosquitto_sub -h localhost -t 'dogwatch/availability' -t 'dogwatch/rear-east/availability' -C 2 -v 2>&1)"
echo "$avail" | grep -q 'online' || PROBLEMS+=("MQTT availability: $(echo "$avail" | tail -1)")

# 7. Watchdog log — FIX/ERR lines only (absent file = healthy, by design)
if [ -f /tmp/dogwatch-watchdog.log ]; then
  grep -qE 'FIX|ERR' /tmp/dogwatch-watchdog.log && PROBLEMS+=("watchdog: $(tail -1 /tmp/dogwatch-watchdog.log)")
fi

# 8. Decoder health — ffmpeg HEVC errors in the detector log over the last
#    15 min. A grabber that keeps up with its stream emits ~zero of these.
#    Sustained errors mean the reader is falling behind and the decoder is
#    parsing partial access units (missing reference pictures), which degrades
#    detection *silently*: frames still arrive, so staleness, the heartbeat,
#    the watchdog and the container healthcheck all stay green. This is the
#    failure mode that hid for days in Sep 2026. Healthy is ~0/15min; the rate
#    when the intake was throttled to 6fps against a 25fps stream was ~2,250.
#    Override with DOGWATCH_DECODE_ERR_MAX.
dec_errs="$(docker logs dogwatch --since 15m 2>&1 | grep -cE 'Error constructing the frame RPS|Could not find ref with POC')"
[ "${dec_errs:-0}" -lt "${DOGWATCH_DECODE_ERR_MAX:-200}" ] || PROBLEMS+=("decoder: ${dec_errs} HEVC errors in 15m (limit ${DOGWATCH_DECODE_ERR_MAX:-200}) — grabber not keeping up with the stream")

if [ "${#PROBLEMS[@]}" -eq 0 ]; then
  echo "HEALTHY"
  exit 0
fi
printf 'ANOMALY: %s\n' "${PROBLEMS[@]}"
exit 1

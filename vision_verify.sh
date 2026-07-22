#!/usr/bin/env bash
# Vision verification throttle guard.
# Usage: ./vision_verify.sh check <camera>
#   Returns 0 (allow verify) or 1 (skip - rate limited)
#
# Usage: ./vision_verify.sh sniff <mqtt_topic>
#   Returns the snapshot/ts value or empty string
#
# Tracks calls in a state file so we don't hammer the Gemini free tier.

STATE_FILE="/home/misaunders/source/dogTracker/.vision_verify_state.json"
MIN_INTERVAL_SECONDS=30
MQTT_HOST="${MQTT_HOST:-172.17.0.1}"
MQTT_PORT="${MQTT_PORT:-1883}"

mkdir -p "$(dirname "$STATE_FILE")"

action="${1:-check}"

if [ "$action" = "check" ]; then
    camera="${2:-rear-east}"
    now=$(date +%s)

    if [ ! -f "$STATE_FILE" ]; then
        echo '{"last_verify":0,"camera":"","snapshot_ts":0}' > "$STATE_FILE"
    fi

    state=$(cat "$STATE_FILE")
    last_verify=$(echo "$state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('last_verify',0))")
    last_camera=$(echo "$state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('camera',''))")

    elapsed=$(( now - last_verify ))

    if [ "$elapsed" -lt "$MIN_INTERVAL_SECONDS" ] && [ "$camera" = "$last_camera" ]; then
        echo "THROTTLE: last verify ${elapsed}s ago (min ${MIN_INTERVAL_SECONDS}s) for camera=$camera"
        exit 1
    fi

    # Update state
    echo "{\"last_verify\":$now,\"camera\":\"$camera\",\"snapshot_ts\":0}" > "$STATE_FILE"
    echo "ALLOW: ${elapsed}s since last call for camera=$camera"
    exit 0

elif [ "$action" = "sniff" ]; then
    topic="${2:-dogwatch/rear-east/snapshot/ts}"
    # Read MQTT retained timestamp
    timeout 5 mosquitto_sub -h "$MQTT_HOST" -p "$MQTT_PORT" -t "$topic" -C 1 2>/dev/null || echo ""
    exit 0

elif [ "$action" = "reset" ]; then
    echo '{"last_verify":0,"camera":"","snapshot_ts":0}' > "$STATE_FILE"
    echo "State reset"
    exit 0
fi

echo "Usage: $0 check <camera> | sniff <topic> | reset"
exit 2

#!/usr/bin/env python3
"""find-dogs-mqtt.py — MQTT trigger listener for the find-dogs scan.

Long-lived companion to the notifier (started by entrypoint.sh). Subscribes
to <base>/find-dogs/trigger and <base>/find-dogs/trigger/+ (device-specific,
e.g. dogwatch/find-dogs/trigger/lounge_echo); on a message runs the find-dogs
scan and publishes the plain-text voice summary to <base>/find-dogs/result
(or <base>/find-dogs/result/<device>), which Home Assistant picks up and
announces on an Echo. This is the Alexa/"where are the dogs" voice path.

NOTE: uses the single-level '+' wildcard, NOT '#': '#' also matches the
parent topic, so trigger/ack/result messages on the bare topic would be
delivered twice (once per subscription/automation) and every Echo would
announce twice. '+' matches only <topic>/<device>.

The device suffix lets HA announce on the *invoking* Echo: HA publishes to
<base>/find-dogs/trigger/<device> and the ack/result come back on the
corresponding <device>-suffixed topics. A bare trigger (no suffix) keeps the
original behaviour — ack/result on the base topics, announced on the default
group of Echos.

Payloads:
  "go"                      — full scan, publish the DeepSeek ack
  "silent"                  — full scan, skip the ack (skill speaks it)
  {"ack": false}            — same as "silent" (JSON form)
  {"ack": false, "channel": N} — scan ONLY camera N, skip the ack
                                 (Alexa "check for dogs at <camera>")

Doggy door (reed switch on the inner locking panel): HA automation
publishes 'locked' / 'open' to <base>/dogdoor (retained). The listener
persists it to the workspace state file so find-dogs.py can infer
"they're inside" when a full scan finds nothing while the door is open.
Bedtime (midnight-06:00 and 22:00+) short-circuits triggers entirely — the
result is the 'in bed' line, no ack, no scan.

MQTT env (same as the notifier): MQTT_HOST / MQTT_PORT / MQTT_TOPIC.
Result summary file lives in the shared workspace volume so the HA-side
consumers (and humans) can inspect what was said.
"""

import json
import os
import subprocess
import threading
import time

import paho.mqtt.client as mqtt

# Daily stats capture (per-day counters for the Daily Dog Report). Fail
# gracefully if the module is missing (older image) — capture must never
# break the listener.
try:
    import stats as _stats
except ImportError:
    _stats = None


def bump_stats(key, amount=1):
    if _stats is None:
        return
    try:
        _stats.bump(key, amount)
    except Exception as e:
        print(f'find-dogs-mqtt: stats bump {key} failed: {e}', flush=True)

MQTT_HOST = os.environ.get('MQTT_HOST', '172.17.0.1')
MQTT_PORT = int(os.environ.get('MQTT_PORT', '1883'))
MQTT_TOPIC = os.environ.get('MQTT_TOPIC', 'dogwatch')
TRIGGER_TOPIC = f'{MQTT_TOPIC}/find-dogs/trigger'
ACK_TOPIC = f'{MQTT_TOPIC}/find-dogs/ack'
RESULT_TOPIC = f'{MQTT_TOPIC}/find-dogs/result'
DOOR_TOPIC = f'{MQTT_TOPIC}/dogdoor'
SUMMARY_FILE = '/app/workspace/find-dogs-result.txt'
RESULT_SIDECAR = '/app/workspace/find-dogs-result.json'
DOOR_STATE_FILE = '/app/workspace/dogdoor.state'

_scan_lock = threading.Lock()
_client = None


def _device_from_topic(topic):
    """Return the device suffix from a trigger topic, or '' for the base.

    dogwatch/find-dogs/trigger          -> ''
    dogwatch/find-dogs/trigger/lounge   -> 'lounge'
    """
    if topic == TRIGGER_TOPIC:
        return ''
    prefix = TRIGGER_TOPIC + '/'
    if topic.startswith(prefix):
        return topic[len(prefix):]
    return ''


def _parse_payload(payload):
    """Return (silent, channel) from a trigger payload.

    silent  — True when the ack should be skipped (the skill speaks it).
    channel — int camera id when only that camera should be scanned,
              else None (full scan).
    """
    if isinstance(payload, bytes):
        payload = payload.decode('utf-8', 'replace')
    if payload == 'silent':
        return True, None
    try:
        d = json.loads(payload)
        if isinstance(d, dict):
            silent = d.get('ack') is False
            channel = d.get('channel')
            if channel is not None:
                try:
                    channel = int(channel)
                except (TypeError, ValueError):
                    channel = None
            return silent, channel
    except (ValueError, TypeError):
        pass
    return False, None


def compose_ack():
    """DeepSeek-composed 'scanning the yard' line via find-dogs.py ack mode.

    Returns the composed line, or the canned fallback if the call fails.
    """
    try:
        r = subprocess.run(
            ['python3', '/app/find-dogs.py', 'ack'],
            capture_output=True, text=True, timeout=30,
        )
        line = r.stdout.strip()
        if line:
            return line
    except Exception as e:
        print(f'find-dogs-mqtt: ack compose error: {e}', flush=True)
    return 'On it, scanning the yard for the dogs.'


def in_bed_line():
    """Return the composed 'in bed' line if it's bedtime, else ''.

    Delegates to find-dogs.py inbed mode (single source of truth for the
    bedtime gate: midnight-06:00 and 22:00+ -> dogs are crated). The caller
    skips the camera scan entirely and publishes this as the result.
    """
    try:
        r = subprocess.run(
            ['python3', '/app/find-dogs.py', 'inbed'],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()
    except Exception as e:
        print(f'find-dogs-mqtt: inbed check error: {e}', flush=True)
        return ''


def write_door_state(payload):
    """Persist the doggy-door panel state for find-dogs.py to read.

    Payload is the plain-text MQTT message ('locked' / 'open') published by
    the HA automation that watches the Tuya reed switch on the inner
    locking panel. State file lives in the shared workspace volume so
    find-dogs.py can answer "they're inside" on an empty full scan.
    """
    state = (payload or b'').decode('utf-8', 'replace').strip().lower()
    if state not in ('locked', 'open'):
        print(f'find-dogs-mqtt: ignoring dogdoor payload {payload!r}',
              flush=True)
        return
    try:
        with open(DOOR_STATE_FILE, 'w') as f:
            json.dump({'state': state, 'ts': time.time()}, f)
        print(f'find-dogs-mqtt: dogdoor state -> {state}', flush=True)
    except OSError as e:
        print(f'find-dogs-mqtt: cannot write {DOOR_STATE_FILE}: {e}',
              flush=True)


def _classify_and_bump():
    """Bump find_dogs_found / empty / inside from the scan's sidecar JSON.

    find-dogs.py writes find-dogs-result.json next to the summary file when
    it runs a scan. Missing sidecar (older image, error path) → no bump.
    """
    try:
        with open(RESULT_SIDECAR) as f:
            r = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if r.get('found'):
        bump_stats('find_dogs_found')
    elif r.get('inside'):
        bump_stats('find_dogs_inside')
    else:
        bump_stats('find_dogs_empty')


def run_scan(device='', silent=False, channel=None):
    """Publish a varied ack (unless silent), run the scan, publish the result.

    With a device suffix the ack/result go to <topic>/<device> so HA can
    announce on the invoking Echo; without one they use the base topics
    (default group announce). With a channel id, only that camera is
    scanned (Alexa "check for dogs at <camera>").
    """
    suffix = f'/{device}' if device else ''
    ack_topic = ACK_TOPIC + suffix
    result_topic = RESULT_TOPIC + suffix
    if not _scan_lock.acquire(blocking=False):
        print('find-dogs-mqtt: scan already running, ignoring trigger',
              flush=True)
        return
    try:
        # Bedtime fast path: dogs are crated (midnight-06:00, 22:00+). No
        # ack, no scan — just answer. Keeps the 2am ask instant and honest.
        bed = in_bed_line()
        if bed:
            print(f'find-dogs-mqtt: bedtime fast path — publishing result '
                  f'({result_topic}): {bed}', flush=True)
            bump_stats('find_dogs_inbed')
            _client.publish(result_topic, bed, qos=0, retain=False)
            return
        if not silent:
            ack = compose_ack()
            print(f'find-dogs-mqtt: publishing ack ({ack_topic}): {ack}',
                  flush=True)
            _client.publish(ack_topic, ack, qos=0, retain=False)
        else:
            print('find-dogs-mqtt: silent trigger — ack suppressed '
                  '(skill speaks it)', flush=True)
        scan_args = ['python3', '/app/find-dogs.py', 'scan']
        if channel:
            scan_args.append(str(channel))
        scan_args += ['--summary-file', SUMMARY_FILE]
        scope = f'channel {channel}' if channel else 'all channels'
        print(f'find-dogs-mqtt: running scan ({scope})...', flush=True)
        subprocess.run(scan_args, timeout=600)
        try:
            with open(SUMMARY_FILE) as f:
                summary = f.read().strip()
        except OSError:
            summary = ''
        if not summary:
            summary = 'The find the dogs scan did not return a result.'
        print(f'find-dogs-mqtt: publishing result ({result_topic}): {summary}',
              flush=True)
        _client.publish(result_topic, summary, qos=0, retain=False)
        _classify_and_bump()
    except subprocess.TimeoutExpired:
        print('find-dogs-mqtt: scan timed out', flush=True)
        _client.publish(result_topic, 'The find the dogs scan timed out.',
                        qos=0, retain=False)
    except Exception as e:
        print(f'find-dogs-mqtt: scan error: {e}', flush=True)
        _client.publish(result_topic, 'The find the dogs scan failed.',
                        qos=0, retain=False)
    finally:
        _scan_lock.release()


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f'find-dogs-mqtt: connected (rc={reason_code}), '
          f'subscribing {TRIGGER_TOPIC} + {TRIGGER_TOPIC}/+ + {DOOR_TOPIC}',
          flush=True)
    client.subscribe([(TRIGGER_TOPIC, 0), (f'{TRIGGER_TOPIC}/+', 0),
                      (DOOR_TOPIC, 0)])


def on_message(client, userdata, msg):
    if msg.topic == DOOR_TOPIC:
        write_door_state(msg.payload)
        return
    device = _device_from_topic(msg.topic)
    silent, channel = _parse_payload(msg.payload)
    print(f'find-dogs-mqtt: trigger received ({msg.topic}): {msg.payload!r}'
          + (f' [device={device}]' if device else '')
          + (' [silent]' if silent else '')
          + (f' [channel={channel}]' if channel else ''), flush=True)
    bump_stats('find_dogs')
    threading.Thread(target=run_scan, args=(device, silent, channel),
                     daemon=True).start()


def main():
    global _client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    _client = client
    print(f'find-dogs-mqtt: connecting to {MQTT_HOST}:{MQTT_PORT}', flush=True)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == '__main__':
    main()

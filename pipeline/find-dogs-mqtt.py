#!/usr/bin/env python3
"""find-dogs-mqtt.py — MQTT trigger listener for the find-dogs scan.

Long-lived companion to the notifier (started by entrypoint.sh). Subscribes
to <base>/find-dogs/trigger (e.g. dogwatch/find-dogs/trigger); on a message
runs the find-dogs scan and publishes the plain-text voice summary to
<base>/find-dogs/result, which Home Assistant picks up and announces on an
Echo via tts.cloud_say. This is the Alexa/"where are the dogs" voice path.

MQTT env (same as the notifier): MQTT_HOST / MQTT_PORT / MQTT_TOPIC.
Result summary file lives in the shared workspace volume so the HA-side
consumers (and humans) can inspect what was said.
"""

import os
import subprocess
import threading

import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get('MQTT_HOST', '172.17.0.1')
MQTT_PORT = int(os.environ.get('MQTT_PORT', '1883'))
MQTT_TOPIC = os.environ.get('MQTT_TOPIC', 'dogwatch')
TRIGGER_TOPIC = f'{MQTT_TOPIC}/find-dogs/trigger'
ACK_TOPIC = f'{MQTT_TOPIC}/find-dogs/ack'
RESULT_TOPIC = f'{MQTT_TOPIC}/find-dogs/result'
SUMMARY_FILE = '/app/workspace/find-dogs-result.txt'

_scan_lock = threading.Lock()
_client = None


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


def run_scan():
    """Publish a varied ack, run the scan, publish the result."""
    if not _scan_lock.acquire(blocking=False):
        print('find-dogs-mqtt: scan already running, ignoring trigger',
              flush=True)
        return
    try:
        ack = compose_ack()
        print(f'find-dogs-mqtt: publishing ack: {ack}', flush=True)
        _client.publish(ACK_TOPIC, ack, qos=0, retain=False)
        print('find-dogs-mqtt: running scan...', flush=True)
        subprocess.run(
            ['python3', '/app/find-dogs.py', 'scan',
             '--summary-file', SUMMARY_FILE],
            timeout=600,
        )
        try:
            with open(SUMMARY_FILE) as f:
                summary = f.read().strip()
        except OSError:
            summary = ''
        if not summary:
            summary = 'The find the dogs scan did not return a result.'
        print(f'find-dogs-mqtt: publishing result: {summary}', flush=True)
        _client.publish(RESULT_TOPIC, summary, qos=0, retain=False)
    except subprocess.TimeoutExpired:
        print('find-dogs-mqtt: scan timed out', flush=True)
        _client.publish(RESULT_TOPIC, 'The find the dogs scan timed out.',
                        qos=0, retain=False)
    except Exception as e:
        print(f'find-dogs-mqtt: scan error: {e}', flush=True)
        _client.publish(RESULT_TOPIC, 'The find the dogs scan failed.',
                        qos=0, retain=False)
    finally:
        _scan_lock.release()


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f'find-dogs-mqtt: connected (rc={reason_code}), '
          f'subscribing {TRIGGER_TOPIC}', flush=True)
    client.subscribe(TRIGGER_TOPIC, qos=0)


def on_message(client, userdata, msg):
    print(f'find-dogs-mqtt: trigger received: {msg.payload!r}', flush=True)
    threading.Thread(target=run_scan, daemon=True).start()


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

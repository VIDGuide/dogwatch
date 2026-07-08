"""mqtt_publisher.py — publishes events and registers HA binary sensors.

Pinned to paho-mqtt < 2 (see Dockerfile) so the simple Client() constructor
works — consistent with the frozen-deps philosophy of the whole container.
Home Assistant MQTT discovery is on by default, so two binary_sensors
("Dog at fence", "Dog digging") appear automatically under a Dogwatch device.
"""
import json
import threading
import time
import paho.mqtt.client as mqtt


class Publisher:
    def __init__(self, host, port, base_topic, camera_name="camera", ha_discovery=True):
        self.base = base_topic
        self.camera_name = camera_name
        self.client = mqtt.Client()
        self.client.connect(host, port, 60)
        self.client.loop_start()
        if ha_discovery:
            self._publish_discovery()

    def _publish_discovery(self):
        cam = self.camera_name
        sensors = [
            ("dog_at_fence", f"{cam} Dog at fence", f"{self.base}/dog_at_fence"),
            ("dog_digging", f"{cam} Dog digging", f"{self.base}/digging"),
        ]
        for slug, display_name, state_topic in sensors:
            dev_id = f"dogwatch_{cam}"
            cfg = {
                "name": display_name,
                "state_topic": state_topic,
                "json_attributes_topic": f"{state_topic}/attributes",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "motion",
                "unique_id": f"{dev_id}_{slug}",
                "device": {"identifiers": [dev_id], "name": f"Dogwatch {cam}"},
            }
            self.client.publish(
                f"homeassistant/binary_sensor/{dev_id}_{slug}/config",
                json.dumps(cfg), retain=True)

        # Camera snapshot discovery — auto-registers an MQTT camera entity
        cam_cfg = {
            "name": f"{cam} Dogwatch",
            "topic": f"{self.base}/snapshot",
            "unique_id": f"{dev_id}_snapshot",
            "device": {"identifiers": [dev_id], "name": f"Dogwatch {cam}"},
        }
        self.client.publish(
            f"homeassistant/camera/{dev_id}_snapshot/config",
            json.dumps(cam_cfg), retain=True)

    def snapshot(self, jpeg_bytes, capture_ts=None):
        """Publish an annotated JPEG frame to the snapshot topic (retained).

        Uses a companion ``snapshot/ts`` topic as a guard: reads the current
        retained timestamp and only publishes if *capture_ts* is newer than or
        equal to it.  This prevents a slower process from overwriting a newer
        snapshot with an older frame.

        If the companion topic has never been set (no retained message) the
        publish always goes through — handles first-run and topic-space
        upgrades gracefully.
        """
        capture_ts = capture_ts or time.time()
        ts_topic = f"{self.base}/snapshot/ts"

        current = self._read_retained(ts_topic)
        if current is not None:
            try:
                if float(current) > capture_ts:
                    return  # A newer snapshot is already published
            except ValueError:
                pass

        # Publish timestamp first (retained), then the JPEG payload.
        self.client.publish(ts_topic, str(capture_ts), retain=True)
        self.client.publish(f"{self.base}/snapshot", payload=jpeg_bytes, retain=True)

    def event(self, etype, payload, auto_off=15):
        topic = f"{self.base}/{etype}"
        self.client.publish(f"{topic}/attributes", json.dumps(payload))
        self.client.publish(topic, "ON")
        if auto_off:
            threading.Timer(
                auto_off, lambda: self.client.publish(topic, "OFF")
            ).start()

    def _read_retained(self, topic, timeout=1.0):
        """Read the last retained message on *topic* via one-shot subscribe.

        Returns the payload as a str, or *None* if there is no retained
        message or the read times out.
        """
        result = [None]

        def _cb(_client, _userdata, msg):
            result[0] = msg.payload.decode()

        self.client.message_callback_add(topic, _cb)
        self.client.subscribe(topic, qos=0)

        deadline = time.time() + timeout
        while result[0] is None and time.time() < deadline:
            self.client.loop(timeout=0.05)

        self.client.unsubscribe(topic)
        self.client.message_callback_remove(topic)
        return result[0]

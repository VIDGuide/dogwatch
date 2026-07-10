"""mqtt_publisher.py — publishes events and registers HA binary sensors.

Uses paho-mqtt 2.x with the VERSION2 callback API (see Dockerfile pin).
Home Assistant MQTT discovery is on by default, so two binary_sensors
("Dog at fence", "Dog digging") appear automatically under a Dogwatch device.

Optional MQTT auth/TLS: set ``mqtt_username``/``mqtt_password`` (and/or
``mqtt_tls: true``) in the camera config to secure the broker connection.
By default the connection is plaintext and unauthenticated, which is fine
for a broker that never leaves localhost/a trusted LAN but should not be
exposed beyond that without auth+TLS.
"""
import json
import threading
import time
import paho.mqtt.client as mqtt


class Publisher:
    def __init__(self, host, port, base_topic, camera_name="camera", ha_discovery=True,
                 off_delay=180, username=None, password=None, use_tls=False):
        self.base = base_topic
        self.camera_name = camera_name
        self.off_delay = off_delay
        self._host = host
        self._port = port
        self._ha_discovery = ha_discovery
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self.client.username_pw_set(username, password)
        if use_tls:
            self.client.tls_set()
        # Survive broker restarts / dropped TCP.  Older paho-mqtt releases
        # could otherwise crash their network thread with "'NoneType' object
        # has no attribute 'recv'" and never recover — which silently kills
        # OFF publishes and leaves HA sensors stuck ON forever. Kept even on
        # 2.x as cheap insurance.
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.connect(host, port, 60)
        self.client.loop_start()
        self._start_supervisor()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        print(f"[{self.camera_name}] MQTT connected reason_code={reason_code}")
        # (Re)publish discovery on every (re)connect so HA entities survive
        # broker restarts and retained configs stay fresh.
        if self._ha_discovery:
            self._publish_discovery()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            print(f"[{self.camera_name}] MQTT disconnected reason_code={reason_code} \u2014 auto-reconnecting")

    def _start_supervisor(self):
        """Watchdog thread: force a reconnect if the network loop ever wedges.

        reconnect_delay_set handles the common case, but if paho's loop thread
        dies outright (the 'NoneType recv' bug) is_connected() stays False
        forever.  This polls and calls reconnect() to guarantee recovery.
        """
        def _watch():
            while True:
                time.sleep(30)
                try:
                    if not self.client.is_connected():
                        print(f"[{self.camera_name}] MQTT not connected \u2014 reconnecting")
                        self.client.reconnect()
                except Exception as e:
                    print(f"[{self.camera_name}] MQTT reconnect attempt failed: {e}")
        threading.Thread(target=_watch, daemon=True).start()

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
                # HA-side safety net: auto-revert to OFF this many seconds after
                # the last ON, even if our OFF message is never delivered (e.g.
                # container restart mid-event).  This is the durable fix for
                # sensors sticking ON permanently.
                "off_delay": self.off_delay,
                "unique_id": f"{dev_id}_{slug}",
                "device": {"identifiers": [dev_id], "name": f"Dogwatch {cam}"},
            }
            self.client.publish(
                f"homeassistant/binary_sensor/{dev_id}_{slug}/config",
                json.dumps(cfg), retain=True)
            # Initialise a clean retained OFF so a fresh HA never shows a stale
            # ON from a previous run.
            self.client.publish(state_topic, "OFF", retain=True)

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
        # Retain state so HA recovers the correct value after any reconnect.
        # off_delay on the HA entity is the primary auto-clear; this timer is a
        # best-effort belt-and-braces OFF for the normal (no-restart) path.
        self.client.publish(topic, "ON", retain=True)
        if auto_off:
            threading.Timer(
                auto_off, lambda: self.client.publish(topic, "OFF", retain=True)
            ).start()

    def _read_retained(self, topic, timeout=1.0):
        """Read the last retained message on *topic* via one-shot subscribe.

        Returns the payload as a str, or *None* if there is no retained
        message or the read times out.

        The client's network loop is already running in a background thread
        (``loop_start()`` in ``__init__``), so this just waits on an event set
        by the message callback rather than pumping ``client.loop()`` itself —
        calling ``loop()`` manually alongside a running ``loop_start()`` thread
        means two threads racing to read the same socket, which can silently
        drop or duplicate messages.
        """
        result = [None]
        received = threading.Event()

        def _cb(_client, _userdata, msg):
            result[0] = msg.payload.decode()
            received.set()

        self.client.message_callback_add(topic, _cb)
        self.client.subscribe(topic, qos=0)

        received.wait(timeout)

        self.client.unsubscribe(topic)
        self.client.message_callback_remove(topic)
        return result[0]

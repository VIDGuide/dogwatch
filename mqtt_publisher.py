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
        self._geometry = None  # set via publish_geometry() before first event
        # Availability ("is the detector actually watching this camera?") is
        # separate from the event state topics: a sensor sitting at OFF is
        # indistinguishable from a detector that died, which is precisely the
        # silent-failure mode this exists to surface. See set_available().
        self.availability_topic = f"{self.base}/availability"
        self._available = None  # None = not yet reported, so first call publishes
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self.client.username_pw_set(username, password)
        if use_tls:
            self.client.tls_set()
        # Last Will and Testament: if this process dies or the socket drops
        # without a clean DISCONNECT, the *broker* publishes "offline" on our
        # behalf. Without this, a hard crash leaves a retained "online" that
        # never gets corrected and HA keeps trusting a dead detector.
        self.client.will_set(self.availability_topic, "offline", retain=True)
        # Survive broker restarts / dropped TCP.  Older paho-mqtt releases
        # could otherwise crash their network thread with "'NoneType' object
        # has no attribute 'recv'" and never recover — which silently kills
        # OFF publishes and leaves HA sensors stuck ON forever. Kept even on
        # 2.x as cheap insurance.
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        # connect_async + loop_start, NOT connect(): a synchronous connect()
        # raises when the broker isn't up yet, and the caller's only sane
        # response was to give up and set pub = None — permanently, because
        # the reconnect machinery below never got armed. A broker that boots
        # a few seconds after this container therefore disabled all MQTT
        # output until someone manually restarted us, with the detector
        # continuing to run blind. connect_async never raises here; the
        # network thread owns the initial connection *and* every retry, so
        # "broker down at startup" and "broker down later" are now the same
        # (recoverable) code path.
        self.client.connect_async(host, port, 60)
        self.client.loop_start()
        self._start_supervisor()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        print(f"[{self.camera_name}] MQTT connected reason_code={reason_code}")
        # (Re)publish discovery on every (re)connect so HA entities survive
        # broker restarts and retained configs stay fresh.
        if self._ha_discovery:
            self._publish_discovery()
        # Re-assert availability after every reconnect: the broker may have
        # published our LWT "offline" while we were away, and a retained
        # "offline" would otherwise stick until the next state change.
        self._available = None
        self.set_available(True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            print(f"[{self.camera_name}] MQTT disconnected reason_code={reason_code} \u2014 auto-reconnecting")

    def _start_supervisor(self):
        """Watchdog thread: force a reconnect if the network loop ever wedges.

        reconnect_delay_set handles the common case, but if paho's loop thread
        dies outright (the 'NoneType recv' bug) is_connected() stays False
        forever.  This polls and calls reconnect() to guarantee recovery.

        This now also covers "never connected in the first place" (broker not
        yet up when we started). connect_async's own retry loop normally
        handles that, so this is belt-and-braces — but it means there is no
        longer *any* path where a startup-time broker outage is permanent.
        The log line distinguishes the two cases so a genuinely unreachable
        broker is diagnosable rather than looking like a flapping connection.
        """
        def _watch():
            ever_connected = False
            while True:
                time.sleep(30)
                try:
                    if self.client.is_connected():
                        ever_connected = True
                        continue
                    state = "reconnecting" if ever_connected else "still waiting for first connect"
                    print(f"[{self.camera_name}] MQTT not connected \u2014 {state}")
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
        dev_id = f"dogwatch_{cam}"
        for slug, display_name, state_topic in sensors:
            cfg = {
                "name": display_name,
                "state_topic": state_topic,
                "json_attributes_topic": f"{state_topic}/attributes",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "motion",
                "availability_topic": self.availability_topic,
                "payload_available": "online",
                "payload_not_available": "offline",
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
            "availability_topic": self.availability_topic,
            "payload_available": "online",
            "payload_not_available": "offline",
            "device": {"identifiers": [dev_id], "name": f"Dogwatch {cam}"},
        }
        self.client.publish(
            f"homeassistant/camera/{dev_id}_snapshot/config",
            json.dumps(cam_cfg), retain=True)

        # Publish detection geometry so the notifier (and any other consumer)
        # can read the detector's actual resolution and crop config directly
        # from MQTT rather than maintaining a separate copy. This eliminates
        # the "stale detect_w/detect_h in dogwatch-notify.config.json" class
        # of bugs — see commit 55b8619 for the incident this fixes.
        if self._geometry:
            self.client.publish(
                f"{self.base}/geometry",
                json.dumps(self._geometry), retain=True)

    def publish_geometry(self, detect_w, detect_h, crop_roi=None):
        """Publish (retained) the detector's resolution and crop config.

        Called once by CameraPipeline after init, and again on every
        reconnect (via _publish_discovery), so a consumer can map bounding
        boxes onto its own snapshot without a hand-maintained copy of the
        detector's resolution.

        **Deliberately carries no credentials.** This used to include the
        camera's ``snapshot_url``, whose documented form is
        ``http://user:pass@nvr-ip/ISAPI/...`` — so NVR credentials were
        written to a *retained* topic on a broker that is plaintext and
        unauthenticated by default. Retained means the message outlived the
        container, was replayed to every new subscriber, and surfaced in Home
        Assistant entity attributes (and therefore in HA backups and
        diagnostics downloads). Nothing ever consumed the field: the notifier
        resolves snapshot URLs from its own gitignored config file, so this
        was pure liability.

        Because a retained publish *replaces* the previous retained message
        on the same topic, the first call after this change also purges the
        credential-bearing payload from the broker. No manual cleanup needed
        beyond letting the detector start once.
        """
        self._geometry = {
            "detect_w": detect_w,
            "detect_h": detect_h,
        }
        if crop_roi:
            self._geometry["crop_roi"] = list(crop_roi)
        self.client.publish(
            f"{self.base}/geometry",
            json.dumps(self._geometry), retain=True)

    def set_available(self, available):
        """Publish (retained) detector availability, but only on a change.

        HA consumes this via ``availability_topic`` on every entity, so a
        stale/dead detector shows up as *unavailable* rather than as a
        permanently-OFF sensor. Called with False when the frame grabber goes
        stale (see CameraPipeline.tick) and True once frames resume.

        Change-gated so a per-frame caller doesn't republish on every tick.
        """
        available = bool(available)
        if self._available == available:
            return
        self._available = available
        payload = "online" if available else "offline"
        self.client.publish(self.availability_topic, payload, retain=True)
        print(f"[{self.camera_name}] availability -> {payload}")

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

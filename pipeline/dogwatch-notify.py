#!/usr/bin/env python3
"""MQTT subscriber for dogwatch events — sends Telegram notifications with
annotated snapshots (bounding boxes from Coral TPU detection) and writes
triggered events to a status file.

Supports multiple cameras via the ``camera`` field in attributes payloads.
"""
import json
import io
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error

import paho.mqtt.client as mqtt
import requests
from PIL import Image, ImageDraw
from requests.auth import HTTPDigestAuth

STATUS_FILE = "/tmp/dogwatch-events.jsonl"
MQTT_HOST = os.environ.get("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
BASE_TOPIC = os.environ.get("MQTT_TOPIC", "dogwatch")

# ---- Config (cameras + chat_id) ----
# Camera registry and chat_id live in an external, gitignored config file so
# no RTSP credentials or chat ids are baked into source (the repo is public).
_CONFIG_PATH = os.environ.get(
    "DOGWATCH_NOTIFY_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dogwatch-notify.config.json"),
)


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load notify config {_CONFIG_PATH}: {exc} \u2014 copy "
            f"dogwatch-notify.config.example.json and fill in camera URLs"
        )


_CONFIG = _load_config()
TELEGRAM_CHAT_ID = str(_CONFIG.get("chat_id", os.environ.get("TELEGRAM_CHAT_ID", "")))


# ---- MQTT topic helpers ----
# The mqtt_publisher configures each camera's snapshot topic as:
#   f"{base_topic}/snapshot"
# where base_topic = "dogwatch" for the primary fence cam, and
# "dogwatch/rear-east" for the rear-east cam.  The notifier must
# match this scheme or its published images land on the wrong topic.

def _snapshot_topic(camera: str) -> str:
    """Return the MQTT snapshot topic for *camera*."""
    if camera == "camera":
        return f"{BASE_TOPIC}/snapshot"
    return f"{BASE_TOPIC}/{camera}/snapshot"

# ---- Camera registry ----
# Loaded from the external config file (see _load_config above).  Each camera
# that publishes dogwatch events needs an entry so the notifier knows which
# stream to snapshot and at what resolution detection ran (for bbox scaling).
#
# NOTE: the fence "camera" deliberately uses the low-res SUB stream for
# snapshots.  The main stream caused constant ffmpeg timeouts (frozen HA
# image); the sub stream decodes instantly and is plenty for a still.
CAMERAS = _CONFIG["cameras"]


# ---- Secrets ----
def _load_bot_token() -> str:
    """Resolve the default Telegram bot token from the OpenClaw secrets file."""
    secrets_path = os.path.expanduser("~/.openclaw/secrets.json")
    try:
        with open(secrets_path) as f:
            sec = json.load(f)
        token = sec.get("channels", {}).get("telegram", {}).get("accounts", {}).get("default", {}).get("botToken", "")
        if token:
            return token
    except Exception as exc:
        print(f"WARN: could not load bot token from {secrets_path}: {exc}")
    # Fallback: env var
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        return token
    raise RuntimeError("No Telegram bot token available — set TELEGRAM_BOT_TOKEN env or fix secrets.json")

BOT_TOKEN = _load_bot_token()
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ---- Bounding box annotation ----

# Store the latest attributes (incl. bbox) per camera+slug.
# Key format: "{camera_name}:{slug}"  e.g. "rear-east:dog_at_fence"
_attributes: dict = {}
BBOX_COLOR = (0, 255, 0)
BBOX_WIDTH = 4


def draw_bbox_on_image(image_path: str, bbox: list, camera_name: str,
                       label: str = "", score: float = 0.0) -> None:
    """Draw a bounding box + label on the image in-place.

    The bbox comes in the detection frame's coordinate space (detect_w/detect_h
    from the camera registry).  The snapshot may be at a different resolution,
    so we scale accordingly.

    For cameras that use a ``crop_roi``, the snapshot is first cropped so it
    matches what the detector evaluated.  The bbox is then drawn in that
    cropped coordinate space.
    """
    cam = CAMERAS.get(camera_name)
    if cam is None:
        print(f"  Unknown camera '{camera_name}', using 640x480 as fallback")
        d_w, d_h = 640, 480
        roi = None
    else:
        d_w, d_h = cam["detect_w"], cam["detect_h"]
        roi = cam.get("crop_roi")

    try:
        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size

        # Crop to the region of interest if configured
        if roi:
            cx1 = int(roi[0] * iw)
            cy1 = int(roi[1] * ih)
            cx2 = int(roi[2] * iw)
            cy2 = int(roi[3] * ih)
            img = img.crop((cx1, cy1, cx2, cy2))
            iw, ih = img.size
            print(f"  Cropped snapshot to {iw}x{ih} (ROI {roi})")

        draw = ImageDraw.Draw(img)

        sx = iw / d_w
        sy = ih / d_h
        x0, y0, x1, y1 = bbox
        x0 = int(x0 * sx)
        y0 = int(y0 * sy)
        x1 = int(x1 * sx)
        y1 = int(y1 * sy)

        x0 = max(0, min(x0, iw))
        y0 = max(0, min(y0, ih))
        x1 = max(0, min(x1, iw))
        y1 = max(0, min(y1, ih))

        draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=BBOX_WIDTH)

        label_text = label + f" {score:.0%}" if score > 0 else label
        if label_text:
            bbox_txt = draw.textbbox((0, 0), label_text)
            tw = bbox_txt[2] - bbox_txt[0]
            th = bbox_txt[3] - bbox_txt[1]
            draw.rectangle([x0, y0 - th - 4, x0 + tw + 4, y0],
                           fill=BBOX_COLOR)
            draw.text((x0 + 2, y0 - th - 2), label_text, fill=(0, 0, 0))

        img.save(image_path, quality=95)
        print(f"  Bounding box drawn on {image_path}: [{x0},{y0},{x1},{y1}]")

    except Exception as exc:
        print(f"  Failed to draw bbox: {exc}")


def _attrs_key(topic: str) -> str:
    """Build a camera-prefixed attributes key from a MQTT topic.

    Topics look like:
        dogwatch/dog_at_fence/attributes        -> camera:dog_at_fence
        dogwatch/rear-east/dog_at_fence/attributes -> rear-east:dog_at_fence
    """
    parts = topic.split("/")
    if len(parts) == 3:
        return f"camera:{parts[1]}"
    elif len(parts) == 4:
        return f"{parts[1]}:{parts[2]}"
    return f"unknown:{parts[-2] if len(parts) >= 2 else '?'}"


def _bbox_from_attributes(attrs: dict) -> list | None:
    raw = attrs.get("bbox")
    if raw and isinstance(raw, (list, tuple)) and len(raw) == 4:
        return [int(v) for v in raw]
    return None


# ---- Telegram helpers ----

def send_telegram_photo(photo_path: str, caption: str) -> bool:
    url = f"{TG_API}/sendPhoto"
    boundary = "----DogWatchBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{TELEGRAM_CHAT_ID}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="dogwatch.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    try:
        with open(photo_path, "rb") as f:
            img_data = f.read()
    except OSError as exc:
        print(f"send_telegram_photo: cannot read {photo_path}: {exc}")
        return False
    body += img_data
    body += f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                print(f"Telegram photo sent: {caption}")
                return True
            else:
                print(f"Telegram API error: {result}")
                return False
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"send_telegram_photo: request failed: {exc}")
        return False


def _read_mqtt_retained(client, topic: str, timeout: float = 1.0) -> str | None:
    """Read the last retained message on *topic* via one-shot subscribe.

    Returns the payload as a str, or *None* if there is no retained
    message or the read times out.
    """
    result = [None]

    def _cb(_client, _userdata, msg):
        result[0] = msg.payload.decode()

    client.message_callback_add(topic, _cb)
    client.subscribe(topic, qos=0)

    deadline = time.time() + timeout
    while result[0] is None and time.time() < deadline:
        client.loop(timeout=0.05)

    client.unsubscribe(topic)
    client.message_callback_remove(topic)
    return result[0]


def publish_image_to_mqtt(client, image_path: str, camera: str, capture_ts: float | None = None) -> None:
    """Publish annotated snapshot to MQTT for HA camera entity.

    Uses a companion ``snapshot/ts`` guard topic to avoid overwriting
    a newer snapshot with an older frame.
    """
    capture_ts = capture_ts or time.time()
    topic = _snapshot_topic(camera)
    ts_topic = f"{topic}/ts"

    # Check guard: skip if a newer snapshot already exists
    current = _read_mqtt_retained(client, ts_topic)
    if current is not None:
        try:
            if float(current) > capture_ts:
                print(f"  Skipping MQTT publish — snapshot/ts has {current} > {capture_ts}")
                return
        except ValueError:
            pass

    # Publish timestamp first, then the JPEG
    client.publish(ts_topic, str(capture_ts), qos=0, retain=True)
    try:
        with open(image_path, "rb") as f:
            payload = f.read()
        client.publish(topic, payload, qos=0, retain=True)
        print(f"  Published snapshot to MQTT topic {topic} ({len(payload)} bytes, ts={capture_ts})")
        _schedule_clear(client, camera)
    except Exception as exc:
        print(f"  Failed to publish snapshot to MQTT: {exc}")


# ---- Snapshot clear timer (reset snapshot to blank after 10 min) ----

_CLEAR_TIMERS: dict = {}  # camera_name -> threading.Timer


def _publish_live_still(client, camera: str) -> None:
    """Capture a clean frame from the camera and publish to MQTT.

    This is used both for the periodic still loop (60s interval) and
    for the event-clear timer (5 min).  The still is published directly
    to the snapshot topic WITHOUT updating the timestamp guard, so
    event-triggered annotated snapshots always take priority.
    """
    snap_path = capture_snapshot(camera)
    if not snap_path:
        return
    try:
        topic = _snapshot_topic(camera)
        with open(snap_path, "rb") as f:
            payload = f.read()
        client.publish(topic, payload, qos=0, retain=True)
        print(f"  Live still for {camera} ({len(payload)} bytes)")
    except Exception as exc:
        print(f"  Failed to publish live still for {camera}: {exc}")
    finally:
        try:
            os.unlink(snap_path)
        except OSError:
            pass


def _schedule_clear(client, camera: str, delay: float = 300.0) -> None:
    """Schedule (or reschedule) a snapshot reset for *delay* seconds.

    After the delay, a clean live still replaces the annotated event
    snapshot.  Rapid successive events cancel and restart the timer so
    only one final clear runs.
    """
    global _CLEAR_TIMERS
    old = _CLEAR_TIMERS.get(camera)
    if old:
        old.cancel()
    t = threading.Timer(delay, _publish_live_still, args=[client, camera])
    t.daemon = True
    _CLEAR_TIMERS[camera] = t
    t.start()
    print(f"  Clear timer set for {camera} in {delay:.0f}s")


def _periodic_still_loop(client, camera: str, interval: float = 60.0) -> None:
    """Background loop: publish a clean camera still every *interval* seconds.

    Runs forever as a daemon thread.  The first tick happens immediately
    so there's always a still on startup, then every *interval* after.
    """
    while True:
        _publish_live_still(client, camera)
        time.sleep(interval)


def send_telegram_text(text: str) -> bool:
    url = f"{TG_API}/sendMessage"
    data = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            ok = result.get("ok", False)
            if ok:
                print(f"Telegram text sent: {text}")
            return ok
    except Exception as exc:
        print(f"send_telegram_text: {exc}")
        return False


# ---- Snapshot ----

def _validate_image(path: str, min_bytes: int = 100_000) -> bool:
    """Quick check that the image has real content (not grey corruption).

    HEVC decode glitches produce flat grey/static images that are substantially
    smaller than real JPEGs at the same resolution.
    """
    size = os.path.getsize(path)
    if size < min_bytes:
        print(f"  Snapshot rejected: {size} bytes < {min_bytes} min (likely corruption)")
        return False
    return True


def capture_snapshot(camera_name: str) -> str:
    """Grab a clean frame from the camera.  Returns file path or ''.

    Primary path: RTSP via ffmpeg with enough frames for the HEVC decoder to
    sync past the initial corruption window (first frame when connecting
    mid-keyframe-interval is often grey static).

    Fallback: HTTP ISAPI snapshot from the NVR (always clean JPEG but may be
    at sub-stream resolution).
    """
    cam = CAMERAS.get(camera_name)
    if cam is None:
        print(f"capture_snapshot: unknown camera '{camera_name}'")
        return ""

    snap_path = f"/tmp/dogwatch_snap_{camera_name}_{int(time.time())}.jpg"

    # Primary: RTSP via ffmpeg.  Wait for a KEYFRAME (I-frame) before writing.
    #
    # Why: these cameras use inter-frame compression (the rear-east main stream
    # is HEVC with a ~2 second GOP).  If we just grab "the next frame" we almost
    # always land mid-GOP on a P/B-frame whose reference I-frame ffmpeg never
    # received on connect — the decoder then renders a flat grey field with a
    # few motion artefacts (mean~128, near-zero variance).  That is the grey /
    # corrupted snapshot problem.
    #
    # `-skip_frame nokey` tells the decoder to discard every non-keyframe, so
    # the first frame we actually output is a self-contained I-frame.  Measured
    # 10/10 clean at ~1.8s on the 2s-GOP HEVC stream (vs ~5/6 grey for a blind
    # single-frame grab, and unreliable for the old fixed -frames:v 10 which
    # only covered 0.5s of a 2s GOP).
    url = cam.get("snapshot_rtsp_fallback", cam["snapshot_url"])
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-skip_frame", "nokey",   # decode only keyframes -> no grey P-frames
                "-i", url,
                "-frames:v", "1",
                "-q:v", "2",
                "-update", "1",
                "-y", snap_path,
            ],
            capture_output=True,
            timeout=15,
        )
        # Guard: ffmpeg may exit without writing the file (timeout / stream
        # error).  os.path.getsize on a missing file raises, so check first.
        if os.path.exists(snap_path) and os.path.getsize(snap_path) > 1000 \
                and _validate_image(snap_path, 50000):
            return snap_path
        # Corrupted or missing — discard and try HTTP fallback
        print(f"  RTSP frame corrupted/missing, trying HTTP snapshot")
        try:
            os.remove(snap_path)
        except OSError:
            pass
    except Exception as exc:
        print(f"RTSP snapshot for '{camera_name}' failed: {exc}")

    # Fallback: HTTP ISAPI snapshot (always clean JPEG)
    if cam["snapshot_url"].startswith("http://") or cam["snapshot_url"].startswith("https://"):
        try:
            parsed = requests.utils.urlparse(cam["snapshot_url"])
            user, pw = parsed.username, parsed.password
            clean_url = cam["snapshot_url"].replace(f"{user}:{pw}@", "") if user else cam["snapshot_url"]
            resp = requests.get(clean_url, auth=HTTPDigestAuth(user, pw), timeout=10)
            resp.raise_for_status()
            with open(snap_path, "wb") as f:
                f.write(resp.content)
            if os.path.getsize(snap_path) > 100:
                print(f"  HTTP fallback snapshot: {os.path.getsize(snap_path)} bytes")
                return snap_path
        except Exception as exc:
            print(f"  HTTP snapshot fallback failed: {exc}")

    return ""


# ---- MQTT ----

_last_on_ts: dict = {}  # debounce per camera — keyed by camera name


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT ({MQTT_HOST}:{MQTT_PORT}) rc={rc}")
    client.subscribe(f"{BASE_TOPIC}/#", qos=0)


def on_message(client, userdata, msg):
    global _last_on_ts, _attributes

    topic = msg.topic
    raw = msg.payload.decode("utf-8", errors="replace")

    # --- Handle attributes topics (store latest bbox per camera+slug) ---
    if "/attributes" in topic:
        akey = _attrs_key(topic)
        try:
            attrs = json.loads(raw)
            bbox = _bbox_from_attributes(attrs)
            if bbox:
                camera = attrs.get("camera", "camera")
                _attributes[akey] = {
                    "bbox": bbox,
                    "camera": camera,
                    "score": attrs.get("score", 0.0),
                    "track": attrs.get("track"),
                    "frame_w": attrs.get("frame_w"),
                    "frame_h": attrs.get("frame_h"),
                    "ts": attrs.get("ts", time.time()),
                }
                print(f"  Stored attributes for {akey}: camera={camera} bbox={bbox}")
        except json.JSONDecodeError:
            pass
        return

    if "/config" in topic:
        return

    # --- Only process ON/OFF state changes ---
    if raw not in ("ON", "OFF"):
        return

    now = time.time()
    parts = topic.split("/")
    if len(parts) == 2:
        camera = "camera"
        slug = parts[1]
    elif len(parts) == 3:
        camera = parts[1]
        slug = parts[2]
    else:
        return

    ts_str = time.strftime("%H:%M:%S", time.localtime(now))

    entry = {"ts": now, "topic": topic, "state": raw, "camera": camera}

    # Look up attributes by camera+slug key
    akey = f"{camera}:{slug}"
    attr = _attributes.get(akey)
    bbox = attr.get("bbox") if attr else None
    if bbox:
        entry["bbox"] = bbox
        if attr and attr.get("score"):
            entry["score"] = attr["score"]

    # Grab snapshot for ON events
    snap_path = ""
    if raw == "ON" and slug in ("dog_at_fence", "digging"):
        last_ts = _last_on_ts.get(camera, 0)
        if now - last_ts > 25:
            snap_path = capture_snapshot(camera)
            _last_on_ts[camera] = now
            if snap_path:
                if bbox:
                    label = slug.replace("_", " ").title()
                    score = attr.get("score", 0.0) if attr else 0.0
                    draw_bbox_on_image(snap_path, bbox, camera, label, score)

                entry["snapshot"] = snap_path

                # Publish annotated snapshot to HA via MQTT camera
                capture_ts = attr.get("ts", now) if attr else now
                publish_image_to_mqtt(client, snap_path, camera, capture_ts)

                if slug == "dog_at_fence":
                    caption = f"🐕 Dog at fence ({camera}) @ {ts_str}"
                elif slug == "digging":
                    caption = f"🕳️ Dogs digging ({camera}) @ {ts_str}"
                else:
                    caption = f"⚠️ Dog alert ({camera}) @ {ts_str}"
                send_telegram_photo(snap_path, caption)
            else:
                fallback = f"🐕 Alert: {slug} ({camera}) @ {ts_str}"
                send_telegram_text(fallback)

    # Write event to status file for the cron-based backup pipeline
    with open(STATUS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    print(f"Dogwatch notifier listening on {BASE_TOPIC}/#")

    # Start periodic still loops for each known camera.
    # Each runs as a daemon thread — first still fires immediately so
    # there's always a frame on startup, then every 60s after that.
    # Event-triggered annotated snapshots override these stills, and
    # the event-clear timer (300s) schedules a one-shot still to
    # replace the annotation back to a clean frame.
    for cam in CAMERAS:
        t = threading.Thread(
            target=_periodic_still_loop,
            args=(client, cam, 60.0),
            daemon=True,
        )
        t.start()
        print(f"  Periodic still loop started for '{cam}' (every 60s)")

    client.loop_forever()


if __name__ == "__main__":
    main()

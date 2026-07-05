"""
dogwatch.py — multi-camera main loop.

Pulls the latest frame from one or more RTSP cameras (each in a background
thread so we never lag behind real-time), runs Coral detection via a single
shared interpreter, tracks each dog, evaluates fence/digging behaviour per
camera, and emits MQTT events to Home Assistant.
"""
import os
import sys
import json
import time
import threading
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from detector import DogDetector
from tracker import CentroidTracker
from behavior import BehaviorMonitor
from mqtt_publisher import Publisher


class FrameGrabber:
    """Background reader that always holds only the newest frame.

    RTSP's classic footgun: if you read frames in your processing loop you fall
    behind the stream's buffer and end up analysing stale video. This keeps the
    latest frame and lets the main loop sample it at whatever rate it likes.
    """

    def __init__(self, url, reconnect_delay=0.5):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.ready = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:                       # stream dropped — reconnect
                time.sleep(self.reconnect_delay)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            with self.lock:
                self.frame = f
                self.ready.set()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class CameraPipeline:
    """One camera's full processing chain: grab, track, monitor, publish."""

    def __init__(self, cfg, name):
        self.name = name
        self.grab = FrameGrabber(cfg["rtsp_url"])

        # Optional HTTP snapshot URL (NVR ISAPI) for clean snapshots.
        # Hikvision format: http://user:pass@nvr-ip/ISAPI/Streaming/channels/1201/picture
        self.snapshot_url = cfg.get("snapshot_url")

        # Wait for the first frame so we know the resolution.
        frame = None
        while frame is None:
            frame = self.grab.read()
            time.sleep(0.2)
        full_h, full_w = frame.shape[:2]

        # Optional crop — zoom into a region of interest.
        # crop_roi: [x1_norm, y1_norm, x2_norm, y2_norm] in full-frame 0-1.
        crop = cfg.get("crop_roi")
        if crop:
            self.crop_norm = tuple(crop)  # normalized 0-1 fractions for snapshot use
            self.crop = (
                max(0, int(round(crop[0] * full_w))),
                max(0, int(round(crop[1] * full_h))),
                min(full_w, int(round(crop[2] * full_w))),
                min(full_h, int(round(crop[3] * full_h))),
            )
            self.h = self.crop[3] - self.crop[1]
            self.w = self.crop[2] - self.crop[0]
            print(f"[{name}] Stream up: {full_w}x{full_h}, crop to {self.w}x{self.h}")
        else:
            self.crop_norm = None
            self.crop = None
            self.h, self.w = full_h, full_w
            print(f"[{name}] Stream up: {self.w}x{self.h}")

        self.tracker = CentroidTracker()
        self.monitor = BehaviorMonitor(cfg, self.w, self.h)
        try:
            self.pub = Publisher(
                os.environ.get("MQTT_HOST", cfg["mqtt_host"]),
                int(os.environ.get("MQTT_PORT", cfg["mqtt_port"])),
                os.environ.get("MQTT_TOPIC", cfg["mqtt_base_topic"]),
                camera_name=name,
            )
        except Exception as e:
            print(f"[{name}] MQTT connection failed: {e} — running without publishing")
            self.pub = None
        self.full_w, self.full_h = full_w, full_h
        self.clip_dir = cfg.get("clip_dir", "clips")
        self.cooldown = cfg.get("event_cooldown_seconds", 30)
        os.makedirs(self.clip_dir, exist_ok=True)

    def _apply_crop(self, frame):
        if self.crop:
            x1, y1, x2, y2 = self.crop
            return frame[y1:y2, x1:x2]
        return frame

    def _fetch_snapshot_image(self):
        """Fetch a clean JPEG from the NVR HTTP snapshot API."""
        url = self.snapshot_url
        if not url:
            return None

        try:
            # Parse credentials from URL for Digest auth.
            parsed = requests.utils.urlparse(url)
            user, pw = parsed.username, parsed.password
            clean_url = url.replace(f"{user}:{pw}@", "") if user else url

            resp = requests.get(clean_url, auth=HTTPDigestAuth(user, pw),
                                timeout=5)
            resp.raise_for_status()

            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[{self.name}] HTTP snapshot decode returned None (bad JPEG)")
            return img

        except Exception as e:
            print(f"[{self.name}] HTTP snapshot fetch failed: {e}")
            return None

    def _publish_snapshot_thread(self, etype, tid, bbox):
        """Fetch snapshot, annotate, and publish (runs in background thread)."""
        img = self._fetch_snapshot_image() if self.snapshot_url else None

        if img is None:
            if self.snapshot_url:
                # HTTP snapshot configured but failed — skip rather than use a
                # potentially corrupted RTSP frame (HEVC decode glitches).
                return
            # No snapshot URL configured: use the RTSP frame directly.
            frame = self.grab.read()
            if frame is None:
                return
            roi = self._apply_crop(frame)
            annotated = roi.copy()
        else:
            # Crop the HTTP snapshot to match the detection ROI.
            # Use normalized fractions so RTSP-vs-HTTP resolution mismatches
            # (e.g. sub-stream vs full 4K) don't produce wrong regions.
            if self.crop_norm:
                snap_h, snap_w = img.shape[:2]
                x1 = max(0, int(round(self.crop_norm[0] * snap_w)))
                y1 = max(0, int(round(self.crop_norm[1] * snap_h)))
                x2 = min(snap_w, int(round(self.crop_norm[2] * snap_w)))
                y2 = min(snap_h, int(round(self.crop_norm[3] * snap_h)))
                annotated = img[y1:y2, x1:x2].copy()
            elif self.crop:
                annotated = img[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]].copy()
            else:
                annotated = img.copy()

        # Draw bounding box and label.
        # Ensure contiguous layout — HTTP snapshot decode can produce
        # non-standard strides that OpenCV drawing chokes on.
        annotated = np.ascontiguousarray(annotated)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, f"{etype} T{tid}",
                    (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ok, buf = cv2.imencode(".jpg", annotated,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok and self.pub:
            self.pub.snapshot(buf.tobytes())

    def tick(self, detector, t0):
        """Process one frame through the shared detector."""
        frame = self.grab.read()
        if frame is None:
            return

        # Crop to region of interest before detection (zoom effect).
        roi = self._apply_crop(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(roi)
        tracks = self.tracker.update([d["bbox"] for d in dets], t0)
        events = self.monitor.evaluate(tracks, gray)

        # Annotate one snapshot per tick (first event only) so we don't spam.
        snapshot_sent = False

        for etype, tid, bbox in events:
            payload = {
                "track": tid,
                "bbox": [int(v) for v in bbox],
                "camera": self.name,
                "frame_w": self.w,
                "frame_h": self.h,
                "ts": t0,
            }
            if self.pub:
                self.pub.event(etype, payload)
            stamp = time.strftime("%H:%M:%S")
            if etype == "digging":
                fn = os.path.join(self.clip_dir, f"dig_{int(t0)}_{tid}.jpg")
                cv2.imwrite(fn, frame)
                print(f"[{stamp}] {self.name}: DIGGING  track {tid} -> {fn}")
            else:
                print(f"[{stamp}] {self.name}: {etype}  track {tid}")

            # Send annotated snapshot once per tick in a background thread.
            if self.pub and not snapshot_sent:
                snapshot_sent = True
                threading.Thread(
                    target=self._publish_snapshot_thread,
                    args=(etype, tid, bbox),
                    daemon=True,
                ).start()


def load_config(path):
    with open(path) as f:
        return json.load(f)


def main():
    # Config files: either passed as CLI args, or default to config.json plus
    # any config-*.json files alongside it.
    if len(sys.argv) > 1:
        config_paths = sys.argv[1:]
    else:
        config_paths = ["config.json"]
        base = os.path.dirname(os.path.abspath("config.json")) or "."
        extras = sorted(
            os.path.join(base, f) for f in os.listdir(base)
            if f.startswith("config-") and f.endswith(".json")
        )
        config_paths.extend(extras)

    cfgs = [load_config(p) for p in config_paths]
    print(f"Loaded {len(cfgs)} camera config(s): {', '.join(config_paths)}")

    # Shared model / Coral interpreter (only one can bind the TPU).
    shared = DogDetector(
        cfgs[0]["model_path"], cfgs[0]["labels_path"],
        cfgs[0]["score_threshold"],
    )

    # Build a pipeline per camera.
    pipelines = []
    for i, cfg in enumerate(cfgs):
        name = os.path.splitext(os.path.basename(config_paths[i]))[0]
        name = name.replace("config-", "").replace("config", "camera")
        pipelines.append(CameraPipeline(cfg, name))

    # Sync all to the fastest camera's target fps.
    target_fps = min(cfg.get("target_fps", 5) for cfg in cfgs)
    interval = 1.0 / target_fps

    # Warm up frame grabbers before entering the loop.
    time.sleep(2)

    while True:
        t0 = time.time()
        for pipe in pipelines:
            pipe.tick(shared, t0)
        dt = time.time() - t0
        if dt < interval:
            time.sleep(interval - dt)


if __name__ == "__main__":
    main()

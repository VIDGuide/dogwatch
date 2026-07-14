"""camera_pipeline.py — one camera's full processing chain.

Grab -> crop -> detect -> track -> evaluate behaviour -> publish. Each
camera in the fleet gets one CameraPipeline instance; a single shared
DogDetector (the Coral interpreter) is passed into tick() so only one
process ever binds the TPU.
"""
import os
import threading
import time

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from behavior import BehaviorMonitor
from debug_capture import DebugCapture
from event_store import EventStore
from frame_grabber import FrameGrabber
from motion_gate import MotionGate
from mqtt_publisher import Publisher
from snapshot_quality import is_image_bad
from tracker import CentroidTracker


class CameraPipeline:
    """One camera's full processing chain: grab, track, monitor, publish."""

    def __init__(self, cfg, name):
        self.name = name
        self.grab = FrameGrabber(
            cfg["rtsp_url"],
            target_fps=cfg.get("target_fps", 5),
            gpu_decode=cfg.get("gpu_decode", False),
        )

        # Optional HTTP snapshot URL (NVR ISAPI) for clean snapshots.
        # Hikvision format: http://user:pass@nvr-ip/ISAPI/Streaming/channels/1201/picture
        self.snapshot_url = cfg.get("snapshot_url")

        # Wait for the first frame so we know the resolution. Bounded so a
        # dead/unreachable camera fails loudly (non-zero exit) instead of
        # hanging the container forever with no signal for the orchestrator
        # to act on.
        frame = None
        deadline = time.time() + cfg.get("startup_timeout_seconds", 60)
        while frame is None:
            frame = self.grab.read()
            if frame is not None:
                break
            if time.time() > deadline:
                raise RuntimeError(
                    f"[{name}] No frame received from {cfg['rtsp_url']} after "
                    f"{cfg.get('startup_timeout_seconds', 60)}s — check the RTSP "
                    f"URL/credentials and that the camera is reachable"
                )
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

        # Centroid-matching distance/miss tolerance scale with the resolution
        # the detector actually runs on (post-crop): a dog moving the same
        # real-world amount covers more pixels at higher resolution, so a
        # fixed default here would fragment tracks into a new ID every frame
        # on a tightly-cropped, high-res feed. Override per-camera in config
        # if the default (tuned for a ~640-720px-wide detection frame) is a
        # poor fit for a much larger or smaller crop.
        self.tracker = CentroidTracker(
            max_distance=cfg.get("tracker_max_distance", 120),
            max_misses=cfg.get("tracker_max_misses", 5),
        )
        self.monitor = BehaviorMonitor(cfg, self.w, self.h)
        try:
            self.pub = Publisher(
                os.environ.get("MQTT_HOST", cfg["mqtt_host"]),
                int(os.environ.get("MQTT_PORT", cfg["mqtt_port"])),
                os.environ.get("MQTT_TOPIC", cfg["mqtt_base_topic"]),
                camera_name=name,
                off_delay=cfg.get("off_delay_seconds", 180),
                username=os.environ.get("MQTT_USERNAME", cfg.get("mqtt_username")),
                password=os.environ.get("MQTT_PASSWORD", cfg.get("mqtt_password")),
                use_tls=cfg.get("mqtt_tls", False),
            )
        except Exception as e:
            print(f"[{name}] MQTT connection failed: {e} — running without publishing")
            self.pub = None
        self.full_w, self.full_h = full_w, full_h

        # Publish detection geometry to MQTT so the notifier (and any other
        # consumer) always knows the exact resolution and crop the detector
        # is running at — single source of truth, no manual sync needed.
        if self.pub:
            self.pub.publish_geometry(
                detect_w=self.w,
                detect_h=self.h,
                crop_roi=list(self.crop_norm) if self.crop_norm else None,
                snapshot_url=cfg.get("snapshot_url"),
            )
        self.clip_dir = cfg.get("clip_dir", "clips")
        self.cooldown = cfg.get("event_cooldown_seconds", 30)
        os.makedirs(self.clip_dir, exist_ok=True)

        # Optional rolling archive of low-res (post-crop, model input) +
        # high-res (raw frame) snapshots per event, for offline diagnosis.
        # Off by default — see debug_capture.py and README "Debug capture".
        self.debug_capture = DebugCapture(cfg, name)
        self._last_debug_cleanup = 0.0

        # Motion gate: skip the TPU entirely when nothing is moving in the
        # scene. Eliminates false positives from static structural elements
        # (beams, railings, shadows) that the model consistently mis-classifies
        # as dogs at 0.5-0.7 confidence. Enabled by default; disable per-camera
        # with "motion_gate_enabled": false.
        self.motion_gate = MotionGate(cfg)

        # SQLite event log — replaces grepping container stdout for event
        # history. Shared db across all cameras (thread-safe).
        self.event_store = EventStore(cfg, camera_name=name)

    def _apply_crop(self, frame):
        if self.crop:
            x1, y1, x2, y2 = self.crop
            return frame[y1:y2, x1:x2]
        return frame

    def _fetch_snapshot_image(self):
        """Fetch a clean JPEG from the NVR HTTP snapshot API, retrying on bad quality.

        The Hikvision ISAPI /picture endpoint can return a partially-decoded
        grey frame if it snaps mid-GOP. We retry up to 3 times with a 500ms
        delay between attempts, validating each result for image quality.
        """
        url = self.snapshot_url
        if not url:
            return None

        for attempt in range(3):
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

                if img is not None and not is_image_bad(img):
                    return img

                if img is None:
                    print(f"[{self.name}] HTTP snapshot decode returned None (bad JPEG, attempt {attempt + 1})")
                else:
                    print(f"[{self.name}] HTTP snapshot returned grey/static frame (attempt {attempt + 1}) \u2014 retrying")

            except Exception as e:
                print(f"[{self.name}] HTTP snapshot fetch failed (attempt {attempt + 1}): {e}")

            if attempt < 2:
                time.sleep(0.5)  # wait for next GOP I-frame before retrying

        print(f"[{self.name}] HTTP snapshot gave bad image after 3 attempts")
        return None

    def _publish_snapshot_thread(self, etype, tid, bbox, capture_ts=None):
        """Fetch snapshot, annotate, and publish (runs in background thread)."""
        img = self._fetch_snapshot_image() if self.snapshot_url else None

        if img is None:
            if self.snapshot_url:
                # HTTP snapshot returned bad quality after 3 retries.
                # Wait for a clean frame from RTSP — the HEVC decoder
                # may take a few frames to sync (a 1080p/HEVC stream
                # typically recovers within 1-2 GOPs, ~1-2s). Poll the
                # FrameGrabber (which keeps the latest frame) and skip
                # decode-glitched grey frames.
                good_frame = None
                for _ in range(30):  # ~3s at 100ms per poll
                    f = self.grab.read()
                    if f is not None and not is_image_bad(f):
                        good_frame = f
                        break
                    time.sleep(0.1)
                frame = good_frame
            else:
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
            self.pub.snapshot(buf.tobytes(), capture_ts=capture_ts)

    def tick(self, detector, t0):
        """Process one frame through the shared detector."""
        frame = self.grab.read()
        if frame is None:
            return

        # Crop to region of interest before detection (zoom effect).
        roi = self._apply_crop(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Motion gate: skip the expensive TPU inference when the scene is
        # completely static. This eliminates false positives from permanent
        # structural elements (beams, railings, fence posts) that never move
        # but that the model consistently scores at 0.5-0.7 confidence.
        if not self.motion_gate.should_detect(gray, t0):
            return

        # Quality check on the cropped ROI (not the full frame — much cheaper).
        # Discards grey/corrupt frames from mid-GOP connects before wasting a
        # TPU inference cycle on garbage input.
        if is_image_bad(roi):
            return

        dets = detector.detect(roi)
        tracks = self.tracker.update(
            [d["bbox"] for d in dets], t0, scores=[d["score"] for d in dets]
        )
        events = self.monitor.evaluate(tracks, gray)

        # Annotate one snapshot per tick (first event only) so we don't spam.
        snapshot_sent = False

        for etype, tid, bbox, score in events:
            payload = {
                "track": tid,
                "bbox": [int(v) for v in bbox],
                "score": round(float(score), 3),
                "camera": self.name,
                "frame_w": self.w,
                "frame_h": self.h,
                "ts": t0,
            }
            if self.pub:
                self.pub.event(etype, payload, auto_off=120)
            stamp = time.strftime("%H:%M:%S")

            # Persist to SQLite for easy post-hoc querying without log grep.
            self.event_store.log_event(
                event_type=etype, track_id=tid, score=score,
                bbox=bbox, frame_w=self.w, frame_h=self.h, ts=t0,
                metadata={"motion_fraction": self.motion_gate.motion_fraction},
            )
            if etype == "digging":
                fn = os.path.join(self.clip_dir, f"dig_{int(t0)}_{tid}.jpg")
                cv2.imwrite(fn, frame)
                print(f"[{stamp}] {self.name}: DIGGING  track {tid} score={score:.2f} -> {fn}")
            else:
                print(f"[{stamp}] {self.name}: {etype}  track {tid} score={score:.2f}")

            # Optional debug archive: low-res = exactly what the model saw
            # (post-crop ROI), high-res = the full raw frame, for offline
            # review of events like false positives. No-op unless enabled.
            self.debug_capture.save(etype, tid, t0, roi, high_res_frame=frame)

            # Send annotated snapshot once per tick in a background thread.
            if self.pub and not snapshot_sent:
                snapshot_sent = True
                threading.Thread(
                    target=self._publish_snapshot_thread,
                    args=(etype, tid, bbox, t0),
                    daemon=True,
                ).start()

        # Sweep old debug captures at most once per hour rather than on
        # every tick (a directory listing per frame at several fps would be
        # wasteful for what's a background housekeeping task).
        if t0 - self._last_debug_cleanup > 3600:
            self._last_debug_cleanup = t0
            self.debug_capture.cleanup()

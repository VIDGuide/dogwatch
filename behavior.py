"""behavior.py — the only file with real "intelligence", and the one you tune.

Two questions per dog per frame:
  1. Are its paws inside the fence zone?  (geometry, not ML)
  2. Is it digging?  (heuristic, not ML)

Digging signal = dog is in the zone AND its body is roughly stationary AND there
is a lot of pixel change *inside its bounding box*. Stationary body + busy box =
limbs working = almost certainly digging. Bounding-box aspect ratio (head-down
posture) is intentionally NOT used as a hard gate — it's too noisy across camera
angles. If you want it as an extra signal, it's a one-line add in is_digging().

All three thresholds below are starting guesses. Expect to tune them against your
own footage — that's where the real accuracy comes from.
"""
import os
import time
import cv2
import numpy as np
from shapely.geometry import Point, Polygon


DEBUG = os.environ.get("DOGWATCH_DEBUG", "").lower() in ("1", "true", "yes")


class BehaviorMonitor:
    def __init__(self, cfg, frame_w, frame_h):
        pts = [(x * frame_w, y * frame_h) for x, y in cfg["fence_zone_norm"]]
        self.zone = Polygon(pts)
        self.stationary_px = cfg["stationary_px"]
        # Digging tolerates more centroid drift than "standing still": a digging
        # dog rocks/shuffles in place, so reusing stationary_px (tuned for a
        # motionless dog) makes the digging gate almost impossible to satisfy.
        # Defaults to 2x stationary_px if not set explicitly.
        self.dig_stationary_px = cfg.get("dig_stationary_px", cfg["stationary_px"] * 2)
        self.motion_thresh = cfg["motion_energy_thresh"]
        self.dig_seconds = cfg["dig_sustain_seconds"]
        self.cooldown = cfg["event_cooldown_seconds"]
        # Require the same track to be seen this many times before firing events.
        # 2 = must survive at least one re-identification, which kills single-frame
        # false positives from shadows/reflections while still catching real dogs.
        self.min_consecutive = cfg.get("min_consecutive", 1)
        self.prev_gray = None
        self._last_event = {}                       # (event, tid) -> ts

    @staticmethod
    def paw_point(bbox):
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2.0, y1)                # bottom-centre = ground contact

    def in_zone(self, bbox):
        px, py = self.paw_point(bbox)
        return self.zone.contains(Point(px, py))

    def intra_box_motion(self, gray, bbox):
        """Fraction of pixels inside the dog's box that changed vs last frame."""
        if self.prev_gray is None:
            return 0.0
        x0, y0, x1, y1 = (int(v) for v in bbox)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        cur = gray[y0:y1, x0:x1]
        prev = self.prev_gray[y0:y1, x0:x1]
        if cur.shape != prev.shape or cur.size == 0:
            return 0.0
        diff = cv2.absdiff(cur, prev)
        return np.count_nonzero(diff > 25) / float(cur.size)

    def is_stationary(self, track, window=2.0, max_drift=None):
        now = track.history[-1][0]
        recent = [c for (tt, c, _) in track.history if tt >= now - window]
        if len(recent) < 2:
            return False
        xs = [c[0] for c in recent]
        ys = [c[1] for c in recent]
        drift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        limit = self.stationary_px if max_drift is None else max_drift
        return drift <= limit

    def evaluate(self, tracks, gray):
        """Return [(event_type, track_id, bbox, score), ...] to publish."""
        events = []
        now = time.time()
        for tid, tr in tracks.items():
            if not self.in_zone(tr.bbox):
                tr.dig_since = None
                continue

            # Skip events until the track has been seen enough times to confirm
            # it's a real, persistent detection rather than a one-frame blip.
            if len(tr.history) < self.min_consecutive:
                continue

            self._maybe_emit(events, "dog_at_fence", tid, tr.bbox, tr.score, now)

            stationary = self.is_stationary(tr, max_drift=self.dig_stationary_px)
            motion = self.intra_box_motion(gray, tr.bbox)
            digging_now = stationary and motion >= self.motion_thresh

            if DEBUG:
                held = (now - tr.dig_since) if tr.dig_since else 0.0
                print(f"[dig-debug] track {tid} in_zone stationary={stationary} "
                      f"motion={motion:.3f} (thresh {self.motion_thresh}) "
                      f"drift_limit={self.dig_stationary_px} "
                      f"dig_held={held:.1f}/{self.dig_seconds}s", flush=True)

            if digging_now:
                if tr.dig_since is None:
                    tr.dig_since = now
                elif now - tr.dig_since >= self.dig_seconds:
                    self._maybe_emit(events, "digging", tid, tr.bbox, tr.score, now)
            else:
                tr.dig_since = None

        self.prev_gray = gray
        return events

    def _maybe_emit(self, events, etype, tid, bbox, score, now):
        key = (etype, tid)
        if now - self._last_event.get(key, 0.0) >= self.cooldown:
            self._last_event[key] = now
            events.append((etype, tid, bbox, score))

"""static_suppressor.py — suppress recurring false positives from fixed scene objects.

The ssd_mobilenet_v2 model consistently scores certain static structural
elements (beams, railings, pipes) at 0.5-0.7 "dog" confidence on every
frame they appear in. The motion gate blocks most of these (nothing moved →
no inference), but any real pixel change elsewhere in the scene (shadow
shift, compression jitter, wind) opens the gate, and the model immediately
re-scores the same static spot.

This module detects that pattern: if the same bbox region fires repeatedly
at a similar score without ever having "arrived" (no significant spatial
movement between detections), it's a static object — suppress it.

A real dog that walks in and stops is NOT suppressed because:
  1. The initial motion of walking in triggers detection normally.
  2. The bbox moves significantly between the first few detections (tracking
     the dog's approach), which resets the "static" counter.
  3. Only after multiple consecutive detections at the SAME position with
     NO movement history does suppression kick in.

Config keys (per-camera, all optional):
    "static_suppression_enabled": true,       # default true
    "static_suppression_iou_threshold": 0.7,  # bbox overlap to consider "same spot"
    "static_suppression_max_hits": 3,         # consecutive same-spot hits before suppressing
    "static_suppression_decay_seconds": 300,  # forget a suppressed region after this long
                                              # without a detection (handles lighting changes
                                              # that shift the false-positive spot)
"""
import time


def _iou(box_a, box_b):
    """Compute Intersection over Union between two boxes [x0, y0, x1, y1]."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _SuppressedRegion:
    """Tracks a single region that has triggered repeatedly without movement."""

    __slots__ = ("bbox", "hit_count", "last_hit_ts", "suppressed")

    def __init__(self, bbox, ts):
        self.bbox = list(bbox)
        self.hit_count = 1
        self.last_hit_ts = ts
        self.suppressed = False

    def matches(self, bbox, iou_threshold):
        return _iou(self.bbox, bbox) >= iou_threshold

    def record_hit(self, bbox, ts):
        # Update bbox to a running average (smooths jitter)
        for i in range(4):
            self.bbox[i] = 0.7 * self.bbox[i] + 0.3 * bbox[i]
        self.hit_count += 1
        self.last_hit_ts = ts

    def is_expired(self, now, decay_seconds):
        return (now - self.last_hit_ts) > decay_seconds


class StaticSuppressor:
    """Suppress detections from static scene regions that fire repeatedly."""

    def __init__(self, cfg):
        self.enabled = cfg.get("static_suppression_enabled", True)
        self.iou_threshold = cfg.get("static_suppression_iou_threshold", 0.7)
        self.max_hits = cfg.get("static_suppression_max_hits", 3)
        self.decay_seconds = cfg.get("static_suppression_decay_seconds", 300.0)
        self._regions = []  # list of _SuppressedRegion

    def should_suppress(self, bbox, ts):
        """Return True if this bbox should be suppressed as a static false positive.

        Call this for each detection that would otherwise fire an event.
        """
        if not self.enabled:
            return False

        # Expire old regions
        self._regions = [
            r for r in self._regions if not r.is_expired(ts, self.decay_seconds)
        ]

        # Check if this bbox matches any known region
        for region in self._regions:
            if region.matches(bbox, self.iou_threshold):
                region.record_hit(bbox, ts)
                if region.hit_count >= self.max_hits:
                    region.suppressed = True
                    return True
                return False

        # New region — start tracking
        self._regions.append(_SuppressedRegion(bbox, ts))
        return False

    def record_movement(self, bbox, ts):
        """Call when a tracked object moves significantly — resets suppression.

        This is the key differentiator: a real dog that walks into frame
        causes bbox movement between frames (tracked by CentroidTracker).
        When we see that movement, we clear any suppression state for that
        region so the detection is allowed through.
        """
        for region in self._regions:
            if region.matches(bbox, self.iou_threshold * 0.5):
                # Object moved TO this location — it arrived, not static
                self._regions.remove(region)
                return

    @property
    def suppressed_count(self):
        """Number of currently suppressed regions."""
        return sum(1 for r in self._regions if r.suppressed)

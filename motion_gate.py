"""motion_gate.py — cheap frame-diff motion check to skip the TPU when nothing moves.

The ssd_mobilenet_v2 model consistently false-positives on static structural
elements (beams, railings, shadow edges) at 0.5-0.7 confidence — these never
move, so a motion gate before detection eliminates them entirely without any
model or zone-geometry changes.

Design:
  - Compares the current greyscale ROI to the previous one via cv2.absdiff.
  - If the fraction of changed pixels is below a configurable threshold, the
    frame is "still" and detection is skipped entirely.
  - A cooldown timer ensures detection still runs periodically even in a
    completely static scene, so a dog that walks in and *stops* isn't missed
    after the initial motion-triggered detection fades from the tracker's
    memory.
  - The gate is bypass-able per-camera via config ("motion_gate_enabled": false)
    for cameras where every frame should be processed regardless.

Config keys (all per-camera, all optional):
    "motion_gate_enabled": true,         # default true — opt-out to disable
    "motion_gate_threshold": 0.005,      # fraction of pixels that must change
    "motion_gate_pixel_threshold": 25,   # per-pixel abs-diff floor (noise filter)
    "motion_gate_max_idle_seconds": 10,  # force a detection pass this often even
                                         # if no motion, so a stationary dog
                                         # doesn't vanish from the tracker
"""
import time

import cv2


class MotionGate:
    """Cheap frame-diff gate: returns True if enough pixels changed to warrant detection."""

    def __init__(self, cfg):
        self.enabled = cfg.get("motion_gate_enabled", True)
        self.threshold = cfg.get("motion_gate_threshold", 0.005)
        self.pixel_threshold = cfg.get("motion_gate_pixel_threshold", 25)
        self.max_idle = cfg.get("motion_gate_max_idle_seconds", 10.0)

        self._prev_gray = None
        self._last_detect_time = 0.0
        self._motion_fraction = 0.0  # exposed for debugging/logging

    def should_detect(self, gray, t0):
        """Return True if detection should run on this frame.

        Parameters
        ----------
        gray : numpy ndarray (H, W) uint8 — greyscale of the current ROI
        t0   : float — current timestamp (time.time() or equivalent)

        Always returns True (bypassed) if the gate is disabled via config.
        """
        if not self.enabled:
            return True

        # First frame after startup: always detect (establishes baseline).
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            self._last_detect_time = t0
            return True

        # Periodic forced detection even without motion, so a dog that walks
        # in and stops isn't forgotten once the tracker's max_misses expire.
        if t0 - self._last_detect_time >= self.max_idle:
            self._prev_gray = gray.copy()
            self._last_detect_time = t0
            return True

        # Shape mismatch (resolution change, crop change) — detect and reset.
        if gray.shape != self._prev_gray.shape:
            self._prev_gray = gray.copy()
            self._last_detect_time = t0
            return True

        # Core motion check: fraction of pixels whose absolute difference
        # exceeds the noise floor.
        diff = cv2.absdiff(gray, self._prev_gray)
        changed = (diff > self.pixel_threshold).sum()
        total = gray.shape[0] * gray.shape[1]
        self._motion_fraction = changed / total if total else 0.0

        if self._motion_fraction >= self.threshold:
            self._prev_gray = gray.copy()
            self._last_detect_time = t0
            return True

        # No significant motion — skip detection this frame but do NOT
        # update _prev_gray, so motion is measured against the last frame
        # that actually triggered detection (accumulates small drifts).
        return False

    @property
    def motion_fraction(self):
        """Last computed motion fraction — useful for debug logging."""
        return self._motion_fraction

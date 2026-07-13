"""Tests for motion_gate.py — the frame-diff gate that skips detection on still scenes."""
import numpy as np
import pytest

from motion_gate import MotionGate


def _blank_gray(h=480, w=640, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _noisy_gray(base, fraction=0.01, magnitude=50):
    """Return a copy of *base* with *fraction* of pixels randomly shifted."""
    out = base.copy()
    n_pixels = int(out.size * fraction)
    indices = np.random.default_rng(42).choice(out.size, size=n_pixels, replace=False)
    flat = out.ravel()
    flat[indices] = np.clip(flat[indices].astype(int) + magnitude, 0, 255).astype(np.uint8)
    return out


class TestMotionGateBasics:
    def test_first_frame_always_detects(self):
        gate = MotionGate({})
        gray = _blank_gray()
        assert gate.should_detect(gray, 1.0) is True

    def test_identical_frames_skip_detection(self):
        gate = MotionGate({})
        gray = _blank_gray()
        gate.should_detect(gray, 1.0)  # first frame
        assert gate.should_detect(gray, 1.5) is False

    def test_significant_motion_triggers_detection(self):
        gate = MotionGate({"motion_gate_threshold": 0.005})
        gray1 = _blank_gray()
        gate.should_detect(gray1, 1.0)

        # Change 2% of pixels significantly — well above 0.5% threshold
        gray2 = _noisy_gray(gray1, fraction=0.02, magnitude=50)
        assert gate.should_detect(gray2, 1.5) is True

    def test_below_threshold_skips(self):
        gate = MotionGate({"motion_gate_threshold": 0.05})
        gray1 = _blank_gray()
        gate.should_detect(gray1, 1.0)

        # Change only 1% — below the 5% threshold
        gray2 = _noisy_gray(gray1, fraction=0.01, magnitude=50)
        assert gate.should_detect(gray2, 1.5) is False

    def test_pixel_threshold_filters_noise(self):
        gate = MotionGate({
            "motion_gate_threshold": 0.005,
            "motion_gate_pixel_threshold": 40,
        })
        gray1 = _blank_gray()
        gate.should_detect(gray1, 1.0)

        # Change 5% of pixels but only by magnitude 30 (below pixel_threshold of 40)
        gray2 = _noisy_gray(gray1, fraction=0.05, magnitude=30)
        assert gate.should_detect(gray2, 1.5) is False


class TestMotionGateIdleForce:
    def test_idle_timeout_forces_detection(self):
        gate = MotionGate({"motion_gate_max_idle_seconds": 5.0})
        gray = _blank_gray()
        gate.should_detect(gray, 1.0)  # first frame

        # Still scene at t=3 — should skip
        assert gate.should_detect(gray, 3.0) is False

        # Still scene at t=7 — exceeded 5s idle, should force detect
        assert gate.should_detect(gray, 7.0) is True

    def test_motion_resets_idle_timer(self):
        gate = MotionGate({"motion_gate_max_idle_seconds": 5.0})
        gray1 = _blank_gray()
        gate.should_detect(gray1, 1.0)

        # Motion at t=4 resets the timer
        gray2 = _noisy_gray(gray1, fraction=0.02, magnitude=50)
        gate.should_detect(gray2, 4.0)

        # At t=8 (4s after last detect, < 5s idle) — should skip
        assert gate.should_detect(gray2, 8.0) is False

        # At t=10 (6s after last detect, > 5s idle) — should force
        assert gate.should_detect(gray2, 10.0) is True


class TestMotionGateDisabled:
    def test_disabled_always_returns_true(self):
        gate = MotionGate({"motion_gate_enabled": False})
        gray = _blank_gray()
        assert gate.should_detect(gray, 1.0) is True
        assert gate.should_detect(gray, 2.0) is True
        assert gate.should_detect(gray, 3.0) is True

    def test_enabled_explicit_true(self):
        gate = MotionGate({"motion_gate_enabled": True})
        gray = _blank_gray()
        gate.should_detect(gray, 1.0)
        # Identical frame should skip when enabled
        assert gate.should_detect(gray, 1.5) is False


class TestMotionGateEdgeCases:
    def test_shape_mismatch_forces_detection(self):
        gate = MotionGate({})
        gray1 = _blank_gray(480, 640)
        gate.should_detect(gray1, 1.0)

        gray2 = _blank_gray(720, 1280)
        assert gate.should_detect(gray2, 2.0) is True

    def test_motion_fraction_property(self):
        gate = MotionGate({"motion_gate_threshold": 0.005})
        gray1 = _blank_gray()
        gate.should_detect(gray1, 1.0)

        gray2 = _noisy_gray(gray1, fraction=0.03, magnitude=50)
        gate.should_detect(gray2, 2.0)
        assert gate.motion_fraction > 0.0

    def test_prev_gray_not_updated_on_skip(self):
        """When motion is below threshold, prev_gray stays at the last detected
        frame so small gradual drifts accumulate rather than being lost."""
        gate = MotionGate({"motion_gate_threshold": 0.05})
        gray1 = _blank_gray(value=100)
        gate.should_detect(gray1, 1.0)

        # Small drift (1%) — skipped
        gray2 = _noisy_gray(gray1, fraction=0.01, magnitude=50)
        assert gate.should_detect(gray2, 2.0) is False

        # Another small drift from gray2 that together with first drift
        # exceeds threshold when measured from the original gray1
        gray3 = _noisy_gray(gray1, fraction=0.06, magnitude=50)
        assert gate.should_detect(gray3, 3.0) is True

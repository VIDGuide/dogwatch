"""Unit tests for behavior.py's fence-zone / digging heuristic."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior import BehaviorMonitor
from tracker import Track

FRAME_W, FRAME_H = 200, 200

# Fence zone covering the bottom half of the frame (normalised 0-1 corners).
FULL_BOTTOM_HALF_ZONE = [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]


def make_cfg(**overrides):
    cfg = {
        "fence_zone_norm": FULL_BOTTOM_HALF_ZONE,
        "stationary_px": 25,
        "motion_energy_thresh": 0.06,
        "dig_sustain_seconds": 2.0,
        "event_cooldown_seconds": 30,
    }
    cfg.update(overrides)
    return cfg


class TestZoneGeometry:
    def test_paw_point_is_bottom_centre(self):
        assert BehaviorMonitor.paw_point((0, 0, 10, 20)) == (5.0, 20.0)

    def test_bbox_in_zone_when_paws_inside_polygon(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        # Bottom half of a 200x200 frame is y >= 100. Paw point at y=150 is inside.
        bbox = (80, 120, 120, 150)
        assert mon.in_zone(bbox) is True

    def test_bbox_out_of_zone_when_paws_outside_polygon(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        # Paw point at y=50 is in the top half, outside the fence zone.
        bbox = (80, 20, 120, 50)
        assert mon.in_zone(bbox) is False


class TestStationary:
    def test_is_stationary_true_when_drift_within_limit(self):
        mon = BehaviorMonitor(make_cfg(stationary_px=25), FRAME_W, FRAME_H)
        tr = Track(1, (100, 100, 110, 110), t=0.0)
        tr.update((102, 101, 112, 111), t=0.5)   # small drift
        tr.update((101, 103, 111, 113), t=1.0)
        assert mon.is_stationary(tr, window=2.0) is True

    def test_is_stationary_false_when_drift_exceeds_limit(self):
        mon = BehaviorMonitor(make_cfg(stationary_px=10), FRAME_W, FRAME_H)
        tr = Track(1, (100, 100, 110, 110), t=0.0)
        tr.update((150, 150, 160, 160), t=0.5)   # big jump
        assert mon.is_stationary(tr, window=2.0) is False

    def test_is_stationary_false_with_insufficient_history(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        tr = Track(1, (100, 100, 110, 110), t=0.0)
        # Only one history point within the window -> can't judge drift.
        assert mon.is_stationary(tr, window=2.0) is False

    def test_dig_stationary_px_defaults_to_double_stationary_px(self):
        mon = BehaviorMonitor(make_cfg(stationary_px=15), FRAME_W, FRAME_H)
        assert mon.dig_stationary_px == 30

    def test_dig_stationary_px_explicit_override(self):
        mon = BehaviorMonitor(make_cfg(stationary_px=15, dig_stationary_px=99), FRAME_W, FRAME_H)
        assert mon.dig_stationary_px == 99


class TestIntraBoxMotion:
    def test_zero_motion_on_first_frame(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        # prev_gray is None on the very first call.
        assert mon.intra_box_motion(gray, (0, 0, 50, 50)) == 0.0

    def test_high_motion_detected_between_differing_frames(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        mon.prev_gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cur = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        frac = mon.intra_box_motion(cur, (0, 0, 50, 50))
        assert frac == 1.0

    def test_no_motion_for_identical_frames(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.full((FRAME_H, FRAME_W), 128, dtype=np.uint8)
        mon.prev_gray = gray.copy()
        frac = mon.intra_box_motion(gray, (0, 0, 50, 50))
        assert frac == 0.0

    def test_bbox_clamped_to_frame_bounds(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        mon.prev_gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cur = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        # bbox extends beyond frame edges — should not raise, just clamp.
        frac = mon.intra_box_motion(cur, (-10, -10, FRAME_W + 50, FRAME_H + 50))
        assert frac == 1.0


class TestEvaluate:
    def test_dog_at_fence_event_emitted_when_in_zone(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        events = mon.evaluate({1: tr}, gray)
        assert ("dog_at_fence", 1, tr.bbox, tr.score) in events

    def test_no_event_when_out_of_zone(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 20, 120, 50), t=time.time())  # top half, out of zone
        events = mon.evaluate({1: tr}, gray)
        assert events == []

    def test_out_of_zone_resets_dig_since(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        tr.dig_since = time.time() - 10  # pretend it was digging
        tr.bbox = (80, 20, 120, 50)  # now out of zone
        # Patch history to match the out-of-zone bbox so paw_point check works.
        tr.history[-1] = (tr.history[-1][0], tr.history[-1][1], tr.bbox)
        mon.evaluate({1: tr}, gray)
        assert tr.dig_since is None

    def test_min_consecutive_suppresses_single_frame_blip(self):
        mon = BehaviorMonitor(make_cfg(min_consecutive=3), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())  # only 1 history entry
        events = mon.evaluate({1: tr}, gray)
        assert events == []

    def test_min_consecutive_allows_event_once_satisfied(self):
        mon = BehaviorMonitor(make_cfg(min_consecutive=2), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        tr.update((80, 120, 120, 150), t=time.time())  # now has 2 history entries
        events = mon.evaluate({1: tr}, gray)
        assert ("dog_at_fence", 1, tr.bbox, tr.score) in events

    def test_cooldown_suppresses_repeat_event(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=100), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        events1 = mon.evaluate({1: tr}, gray)
        events2 = mon.evaluate({1: tr}, gray)
        assert len(events1) == 1
        assert len(events2) == 0  # still within cooldown

    def test_digging_fires_after_sustained_stationary_motion(self):
        # dig_since is set on the first frame where digging_now becomes True,
        # then only checked (and the event fired) on a *later* evaluate()
        # call once dig_sustain_seconds has elapsed since dig_since — so this
        # needs three evaluate() calls: seed prev_gray, set dig_since, then
        # fire once the (zero) sustain duration has elapsed.
        mon = BehaviorMonitor(
            make_cfg(dig_sustain_seconds=0.0, motion_energy_thresh=0.5,
                     stationary_px=1000, dig_stationary_px=1000),
            FRAME_W, FRAME_H,
        )
        bbox = (80, 120, 120, 150)
        t0 = time.time()
        tr = Track(1, bbox, t=t0)
        tr.update(bbox, t=t0 + 0.1)  # 2nd history point so is_stationary can judge

        gray_a = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        gray_b = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        mon.evaluate({1: tr}, gray_a)  # seeds prev_gray (motion=0, no dig_since yet)

        mon.evaluate({1: tr}, gray_b)  # diff vs gray_a is high -> sets dig_since
        assert tr.dig_since is not None

        # Alternate back to gray_a so the diff vs the now-stored gray_b is
        # still high motion on this next call, letting the sustain check fire.
        events = mon.evaluate({1: tr}, gray_a)

        types = [e[0] for e in events]
        assert "digging" in types

    def test_digging_does_not_fire_before_sustain_duration_elapsed(self):
        mon = BehaviorMonitor(
            make_cfg(dig_sustain_seconds=999, motion_energy_thresh=0.5,
                     stationary_px=1000, dig_stationary_px=1000),
            FRAME_W, FRAME_H,
        )
        bbox = (80, 120, 120, 150)
        t0 = time.time()
        tr = Track(1, bbox, t=t0)
        tr.update(bbox, t=t0 + 0.1)

        gray1 = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        mon.evaluate({1: tr}, gray1)

        gray2 = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        events = mon.evaluate({1: tr}, gray2)

        types = [e[0] for e in events]
        assert "digging" not in types

    def test_digging_resets_when_motion_drops(self):
        mon = BehaviorMonitor(
            make_cfg(dig_sustain_seconds=0.0, motion_energy_thresh=0.5,
                     stationary_px=1000, dig_stationary_px=1000),
            FRAME_W, FRAME_H,
        )
        bbox = (80, 120, 120, 150)
        t0 = time.time()
        tr = Track(1, bbox, t=t0)
        tr.update(bbox, t=t0 + 0.1)

        gray_a = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        mon.evaluate({1: tr}, gray_a)
        gray_b = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        mon.evaluate({1: tr}, gray_b)  # dig_since set here
        assert tr.dig_since is not None

        # Identical frame next -> zero motion -> digging condition breaks.
        mon.evaluate({1: tr}, gray_b)
        assert tr.dig_since is None

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



class TestZoneBoundaryInclusion:
    """`contains` excludes the polygon boundary; `covers` includes it.

    This is not hypothetical: the default zone in config.example.json is the
    full frame, which puts the polygon edge exactly at y == frame_h, and the
    paw point is the bbox *bottom* edge. Every dog detected at the bottom of
    the frame therefore sat precisely on the excluded boundary.
    """

    FULL_FRAME_ZONE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def test_paw_exactly_on_bottom_edge_is_inside_full_frame_zone(self):
        mon = BehaviorMonitor(make_cfg(fence_zone_norm=self.FULL_FRAME_ZONE),
                              FRAME_W, FRAME_H)
        # Paw point y == FRAME_H, exactly on the polygon edge.
        assert mon.in_zone((80, 100, 120, FRAME_H)) is True

    def test_paw_exactly_on_zone_top_edge_is_inside(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)  # bottom-half zone
        # Bottom-half zone starts at y == 100; a paw exactly there counts.
        assert mon.in_zone((80, 50, 120, 100)) is True

    def test_paw_on_left_edge_is_inside(self):
        mon = BehaviorMonitor(make_cfg(fence_zone_norm=self.FULL_FRAME_ZONE),
                              FRAME_W, FRAME_H)
        # paw_point x is the bbox centre, so use a zero-width box at x == 0.
        assert mon.in_zone((0, 100, 0, 150)) is True

    def test_paw_clearly_outside_is_still_outside(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        assert mon.in_zone((80, 20, 120, 50)) is False

    def test_paw_just_past_edge_is_outside(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        # One pixel above the bottom-half zone boundary.
        assert mon.in_zone((80, 50, 120, 99)) is False


class TestObserveFrame:
    """observe_frame keeps prev_gray advancing on frames the motion gate
    suppresses, so intra_box_motion always compares against the immediately
    preceding frame rather than whatever last passed the gate (up to
    motion_gate_max_idle_seconds ago)."""

    def test_observe_frame_sets_prev_gray(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.full((FRAME_H, FRAME_W), 7, dtype=np.uint8)
        mon.observe_frame(gray)
        assert mon.prev_gray is gray

    def test_motion_measured_against_observed_frame(self):
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        black = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        white = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        mon.observe_frame(black)
        assert mon.intra_box_motion(white, (0, 0, 50, 50)) == 1.0

    def test_gated_frames_keep_baseline_current(self):
        """Two gated (identical) frames then a real one: motion is measured
        against the immediately-preceding frame, so an unchanged scene reads as
        zero motion rather than accumulating a stale 10s-old difference."""
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        steady = np.full((FRAME_H, FRAME_W), 100, dtype=np.uint8)
        mon.observe_frame(steady)
        mon.observe_frame(steady.copy())
        assert mon.intra_box_motion(steady.copy(), (0, 0, 50, 50)) == 0.0


class TestMissedTracksAreSkipped:
    """Tracks the tracker is holding open without a detection must not fire.

    ``CentroidTracker`` keeps a track alive for ``max_misses`` frames so it
    survives a brief re-identification gap, and returns it in the live set with
    the ``bbox`` from the last frame it *was* detected on. ``evaluate`` used to
    treat those exactly like fresh detections, so a dog leaving the frame kept
    firing ``dog_at_fence`` on a stale box for several more frames.

    Worse for digging: both halves of the heuristic read as *more* positive the
    longer a track goes unseen. ``is_stationary`` measures drift over
    ``history``, which stops being appended to, so drift falls to zero; and
    ``intra_box_motion`` diffs the current frame against the stale box, which
    lights up precisely because the dog has moved out of it. Stationary plus
    busy box is the digging signal.
    """

    def test_no_event_for_a_track_with_misses(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=0), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        tr.misses = 1                      # not matched on this frame

        assert mon.evaluate({1: tr}, gray) == []

    def test_event_fires_again_once_the_track_is_redetected(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=0), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        bbox = (80, 120, 120, 150)
        tr = Track(1, bbox, t=time.time())

        assert [e[0] for e in mon.evaluate({1: tr}, gray)] == ["dog_at_fence"]

        tr.misses = 2
        assert mon.evaluate({1: tr}, gray) == []

        tr.update(bbox, t=time.time())     # re-detected: update resets misses
        assert [e[0] for e in mon.evaluate({1: tr}, gray)] == ["dog_at_fence"]

    def test_only_the_missed_track_is_skipped(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=0), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        seen = Track(1, (80, 120, 120, 150), t=time.time())
        missed = Track(2, (20, 120, 60, 150), t=time.time())
        missed.misses = 3

        events = mon.evaluate({1: seen, 2: missed}, gray)

        assert [e[1] for e in events] == [1]

    def test_dig_since_is_preserved_across_a_miss(self):
        # Skipping must not *reset* an in-progress dig: a real dog that the
        # model drops for one frame should resume, not start over.
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        tr.dig_since = time.time()

        tr.misses = 1
        mon.evaluate({1: tr}, gray)

        assert tr.dig_since is not None

    def test_digging_does_not_fire_from_a_stale_box(self):
        # The composite failure: a stationary-by-omission track whose box lights
        # up because the dog left it. dig_sustain_seconds=0 makes this fire on
        # the very next evaluate if the track is not skipped.
        mon = BehaviorMonitor(
            make_cfg(dig_sustain_seconds=0.0, motion_energy_thresh=0.5,
                     stationary_px=1000, dig_stationary_px=1000,
                     event_cooldown_seconds=0),
            FRAME_W, FRAME_H,
        )
        bbox = (80, 120, 120, 150)
        t0 = time.time()
        tr = Track(1, bbox, t=t0)
        tr.update(bbox, t=t0 + 0.1)

        gray_a = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        gray_b = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)

        mon.evaluate({1: tr}, gray_a)      # seed prev_gray while detected
        tr.misses = 1                      # dog gone from this box
        events = mon.evaluate({1: tr}, gray_b)

        assert [e[0] for e in events] == []

    def test_baseline_still_advances_when_every_track_is_missed(self):
        # prev_gray must keep tracking the newest frame even if no track is
        # evaluated, or intra-box motion would later be measured against a
        # stale baseline — the bug observe_frame() exists to prevent.
        mon = BehaviorMonitor(make_cfg(), FRAME_W, FRAME_H)
        gray_a = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        gray_b = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        tr.misses = 1

        mon.evaluate({1: tr}, gray_a)
        mon.evaluate({1: tr}, gray_b)

        assert mon.prev_gray is gray_b


class TestCooldownPruning:
    """_last_event is keyed by (event_type, track_id) and track ids increment
    forever, so without pruning every track that ever fired left a permanent
    entry — an unbounded dict in a process meant to run for months."""

    def test_entry_kept_while_track_is_alive(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=1), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        mon.evaluate({1: tr}, gray)
        assert ("dog_at_fence", 1) in mon._last_event

    def test_entry_pruned_once_track_gone_and_cooldown_expired(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=0), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        mon.evaluate({1: tr}, gray)
        assert mon._last_event

        # Age the recorded event, then evaluate with the track gone.
        for key in list(mon._last_event):
            mon._last_event[key] = time.time() - 100
        mon.evaluate({}, gray)
        assert mon._last_event == {}

    def test_entry_not_pruned_while_cooldown_still_active(self):
        """Pruning must never let a suppressed event through early."""
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=10_000),
                              FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        tr = Track(1, (80, 120, 120, 150), t=time.time())
        mon.evaluate({1: tr}, gray)
        mon.evaluate({}, gray)  # track gone, but cooldown is far from expired
        assert ("dog_at_fence", 1) in mon._last_event

    def test_dict_does_not_grow_without_bound_across_many_tracks(self):
        mon = BehaviorMonitor(make_cfg(event_cooldown_seconds=0), FRAME_W, FRAME_H)
        gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        for tid in range(1, 200):
            tr = Track(tid, (80, 120, 120, 150), t=time.time())
            mon.evaluate({tid: tr}, gray)
            # Age everything so the previous track's entry is prunable.
            for key in list(mon._last_event):
                if key[1] != tid:
                    mon._last_event[key] = time.time() - 100
        # Only the most recent track's entry should survive.
        assert len(mon._last_event) <= 2

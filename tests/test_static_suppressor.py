"""Tests for static_suppressor.py — static bbox false-positive suppression."""
import time

import pytest

from static_suppressor import StaticSuppressor, _iou


class TestIoU:
    def test_identical_boxes(self):
        assert _iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_no_overlap(self):
        assert _iou([0, 0, 50, 50], [60, 60, 100, 100]) == 0.0

    def test_partial_overlap(self):
        iou = _iou([0, 0, 100, 100], [50, 50, 150, 150])
        # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        assert abs(iou - 2500 / 17500) < 0.001

    def test_contained_box(self):
        iou = _iou([0, 0, 100, 100], [25, 25, 75, 75])
        # Intersection: 50x50 = 2500, Union: 10000 + 2500 - 2500 = 10000
        assert abs(iou - 0.25) < 0.001

    def test_zero_area_box(self):
        assert _iou([0, 0, 0, 0], [0, 0, 100, 100]) == 0.0


class TestStaticSuppressorBasics:
    def test_first_detection_not_suppressed(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        assert s.should_suppress([100, 200, 300, 400], 1.0) is False

    def test_second_detection_same_spot_not_suppressed(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        s.should_suppress([100, 200, 300, 400], 1.0)
        assert s.should_suppress([100, 200, 300, 400], 2.0) is False

    def test_third_detection_same_spot_suppressed(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        s.should_suppress([100, 200, 300, 400], 1.0)
        s.should_suppress([100, 200, 300, 400], 2.0)
        assert s.should_suppress([100, 200, 300, 400], 3.0) is True

    def test_subsequent_hits_remain_suppressed(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        for i in range(5):
            s.should_suppress([100, 200, 300, 400], float(i))
        assert s.should_suppress([100, 200, 300, 400], 6.0) is True

    def test_different_location_not_suppressed(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        # Suppress one region
        for i in range(4):
            s.should_suppress([100, 200, 300, 400], float(i))
        # Different location should not be affected
        assert s.should_suppress([500, 500, 700, 700], 5.0) is False

    def test_slight_jitter_still_matches(self):
        """Bbox that jitters slightly (compression noise) should still be tracked as same region."""
        s = StaticSuppressor({"static_suppression_max_hits": 3, "static_suppression_iou_threshold": 0.7})
        s.should_suppress([100, 200, 300, 400], 1.0)
        s.should_suppress([105, 195, 305, 405], 2.0)  # slight shift
        assert s.should_suppress([98, 202, 298, 398], 3.0) is True  # suppressed


class TestStaticSuppressorDecay:
    def test_region_expires_after_decay(self):
        s = StaticSuppressor({
            "static_suppression_max_hits": 3,
            "static_suppression_decay_seconds": 60.0,
        })
        # Build up hits
        s.should_suppress([100, 200, 300, 400], 1.0)
        s.should_suppress([100, 200, 300, 400], 2.0)
        s.should_suppress([100, 200, 300, 400], 3.0)  # now suppressed

        # After decay, region should be forgotten
        assert s.should_suppress([100, 200, 300, 400], 100.0) is False  # fresh start

    def test_region_does_not_expire_if_recently_hit(self):
        s = StaticSuppressor({
            "static_suppression_max_hits": 3,
            "static_suppression_decay_seconds": 60.0,
        })
        for i in range(4):
            s.should_suppress([100, 200, 300, 400], float(i))
        # Still within decay window
        assert s.should_suppress([100, 200, 300, 400], 50.0) is True


class TestStaticSuppressorMovement:
    def test_movement_resets_suppression(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        # Build up to suppression
        s.should_suppress([100, 200, 300, 400], 1.0)
        s.should_suppress([100, 200, 300, 400], 2.0)

        # Object moved to that location (a real dog arriving)
        s.record_movement([110, 210, 310, 410], 2.5)

        # Third hit should NOT be suppressed since movement was recorded
        assert s.should_suppress([100, 200, 300, 400], 3.0) is False

    def test_movement_at_different_location_doesnt_affect_other_regions(self):
        s = StaticSuppressor({"static_suppression_max_hits": 3})
        for i in range(3):
            s.should_suppress([100, 200, 300, 400], float(i))
        # Movement at a totally different spot
        s.record_movement([600, 600, 800, 800], 3.5)
        # Original region remains suppressed
        assert s.should_suppress([100, 200, 300, 400], 4.0) is True


class TestStaticSuppressorDisabled:
    def test_disabled_never_suppresses(self):
        s = StaticSuppressor({"static_suppression_enabled": False})
        for i in range(10):
            assert s.should_suppress([100, 200, 300, 400], float(i)) is False


class TestStaticSuppressorProperties:
    def test_suppressed_count(self):
        s = StaticSuppressor({"static_suppression_max_hits": 2})
        s.should_suppress([100, 200, 300, 400], 1.0)
        assert s.suppressed_count == 0
        s.should_suppress([100, 200, 300, 400], 2.0)
        assert s.suppressed_count == 1

    def test_multiple_suppressed_regions(self):
        s = StaticSuppressor({"static_suppression_max_hits": 2})
        # Region 1
        s.should_suppress([100, 200, 300, 400], 1.0)
        s.should_suppress([100, 200, 300, 400], 2.0)
        # Region 2
        s.should_suppress([500, 500, 700, 700], 3.0)
        s.should_suppress([500, 500, 700, 700], 4.0)
        assert s.suppressed_count == 2



# ---------------------------------------------------------------------------
# Protected event types
# ---------------------------------------------------------------------------

BEAM = [100, 200, 300, 400]


def suppressor(**cfg):
    cfg.setdefault("static_suppression_max_hits", 3)
    return StaticSuppressor(cfg, camera_name="rear-east", log=lambda m: None)


def saturate(s, bbox=BEAM, event_type="dog_at_fence", start=1.0, n=3):
    """Drive a region to the suppression threshold."""
    for i in range(n):
        s.should_suppress(bbox, start + i, event_type=event_type)


class TestProtectedEvents:
    """A stationary digging dog reached max_hits in ~90s (events are rate
    limited to one per 30s cooldown per track), and was then classified as a
    structural element — dropping the one event the siren depends on."""

    def test_digging_is_never_suppressed_even_when_region_is_saturated(self):
        s = suppressor()
        saturate(s, n=5)                    # region is firmly "static"
        assert s.should_suppress(BEAM, 10.0, event_type="dog_at_fence") is True
        assert s.should_suppress(BEAM, 11.0, event_type="digging") is False

    def test_digging_clears_an_existing_suppression(self):
        """A structural element does not dig, so a digging event is positive
        evidence the region is real."""
        s = suppressor()
        saturate(s, n=5)
        assert s.should_suppress(BEAM, 10.0, event_type="dog_at_fence") is True
        s.should_suppress(BEAM, 11.0, event_type="digging")
        # The region is no longer suppressed for other event types either.
        assert s.should_suppress(BEAM, 12.0, event_type="dog_at_fence") is False

    def test_digging_clearance_is_logged(self):
        logs = []
        s = StaticSuppressor({"static_suppression_max_hits": 3},
                             camera_name="rear-east", log=logs.append)
        saturate(s, n=5)
        s.should_suppress(BEAM, 10.0, event_type="dog_at_fence")
        logs.clear()
        s.should_suppress(BEAM, 11.0, event_type="digging")
        assert any("CLEARS" in m for m in logs)

    def test_digging_on_a_brand_new_region_is_not_suppressed(self):
        s = suppressor()
        for i in range(10):
            assert s.should_suppress(BEAM, float(i), event_type="digging") is False

    def test_protected_list_is_configurable(self):
        s = suppressor(static_suppression_protected_events=["dog_at_fence"])
        saturate(s, event_type="digging", n=5)
        # Now digging is the unprotected type and gets suppressed...
        assert s.should_suppress(BEAM, 10.0, event_type="digging") is True
        # ...while dog_at_fence is protected.
        assert s.should_suppress(BEAM, 11.0, event_type="dog_at_fence") is False

    def test_empty_protected_list_restores_old_behaviour(self):
        s = suppressor(static_suppression_protected_events=[])
        saturate(s, event_type="digging", n=3)
        assert s.should_suppress(BEAM, 10.0, event_type="digging") is True

    def test_default_protects_digging_only(self):
        s = suppressor()
        assert s.protected_events == frozenset({"digging"})

    def test_no_event_type_still_suppresses(self):
        """Callers that don't pass an event type keep the original behaviour."""
        s = suppressor()
        for i in range(3):
            s.should_suppress(BEAM, float(i))
        assert s.should_suppress(BEAM, 5.0) is True


class TestMovementGrace:
    """record_movement used to delete the region, which merely reset hit_count —
    so a dog that arrived and then held still re-accumulated max_hits and was
    suppressed again a couple of minutes later."""

    def test_movement_withholds_suppression_within_grace(self):
        s = suppressor(static_suppression_movement_grace_seconds=100.0)
        s.should_suppress(BEAM, 1.0, event_type="dog_at_fence")
        s.record_movement([110, 210, 310, 410], 2.0)
        # Many further hits, all inside the grace window.
        for i in range(10):
            assert s.should_suppress(BEAM, 3.0 + i, event_type="dog_at_fence") is False

    def test_suppression_resumes_after_grace_expires(self):
        s = suppressor(static_suppression_movement_grace_seconds=10.0,
                       static_suppression_decay_seconds=10_000.0)
        s.should_suppress(BEAM, 1.0, event_type="dog_at_fence")
        s.record_movement(BEAM, 2.0)
        # Past the grace window, the region can be judged static again.
        results = [s.should_suppress(BEAM, 100.0 + i, event_type="dog_at_fence")
                   for i in range(5)]
        assert True in results

    def test_grace_defaults_to_decay_seconds(self):
        s = suppressor(static_suppression_decay_seconds=42.0)
        assert s.movement_grace_seconds == 42.0

    def test_movement_stamps_all_overlapping_regions(self):
        s = suppressor()
        s.should_suppress([100, 200, 300, 400], 1.0, event_type="dog_at_fence")
        s.should_suppress([700, 700, 900, 900], 1.0, event_type="dog_at_fence")
        s.record_movement([100, 200, 300, 400], 2.0)
        stats_before = s.stats()["events_movement_grace"]
        s.should_suppress([100, 200, 300, 400], 3.0, event_type="dog_at_fence")
        assert s.stats()["events_movement_grace"] > stats_before

    def test_movement_is_a_noop_when_disabled(self):
        s = StaticSuppressor({"static_suppression_enabled": False},
                             log=lambda m: None)
        s.record_movement(BEAM, 1.0)  # must not raise


class TestSuppressionLogging:
    """Suppression previously returned True in complete silence, so a dropped
    real event was indistinguishable from 'nothing happened'."""

    def test_first_suppression_logs(self):
        logs = []
        s = StaticSuppressor({"static_suppression_max_hits": 3},
                             camera_name="rear-east", log=logs.append)
        saturate(s, n=3)
        assert logs, "suppression must not be silent"
        assert "rear-east" in logs[0]
        assert "suppressing" in logs[0]

    def test_log_mentions_the_config_knob_to_change(self):
        logs = []
        s = StaticSuppressor({"static_suppression_max_hits": 3},
                             log=logs.append)
        saturate(s, n=3)
        assert "static_suppression" in logs[0]

    def test_repeat_suppressions_are_rate_limited(self):
        logs = []
        s = StaticSuppressor({"static_suppression_max_hits": 2,
                              "static_suppression_decay_seconds": 10_000.0},
                             log=logs.append)
        for i in range(50):
            s.should_suppress(BEAM, float(i), event_type="dog_at_fence")
        # One line, not fifty.
        assert len(logs) == 1

    def test_log_repeats_after_the_rate_limit_window(self):
        logs = []
        s = StaticSuppressor({"static_suppression_max_hits": 2,
                              "static_suppression_decay_seconds": 10_000.0},
                             log=logs.append)
        s.should_suppress(BEAM, 1.0, event_type="dog_at_fence")
        s.should_suppress(BEAM, 2.0, event_type="dog_at_fence")
        s.should_suppress(BEAM, 1000.0, event_type="dog_at_fence")
        assert len(logs) == 2


class TestStats:
    def test_stats_shape(self):
        s = suppressor()
        assert set(s.stats()) == {
            "regions_tracked", "regions_suppressed", "events_suppressed",
            "events_protected", "events_movement_grace",
        }

    def test_suppressed_events_counted(self):
        s = suppressor(static_suppression_decay_seconds=10_000.0)
        for i in range(6):
            s.should_suppress(BEAM, float(i), event_type="dog_at_fence")
        assert s.stats()["events_suppressed"] > 0

    def test_protected_events_counted(self):
        s = suppressor()
        s.should_suppress(BEAM, 1.0, event_type="digging")
        assert s.stats()["events_protected"] == 1

    def test_regions_tracked_reflects_distinct_spots(self):
        s = suppressor()
        s.should_suppress([100, 200, 300, 400], 1.0, event_type="dog_at_fence")
        s.should_suppress([700, 700, 900, 900], 1.0, event_type="dog_at_fence")
        assert s.stats()["regions_tracked"] == 2


class TestRealDogScenario:
    """End-to-end shape of the bug: a dog walks to the fence, stops, and digs in
    one spot for several minutes. Every digging event must get through."""

    def test_stationary_digging_dog_is_never_suppressed(self):
        s = suppressor(static_suppression_max_hits=3,
                       static_suppression_decay_seconds=300.0)
        bbox = [400, 500, 520, 620]
        ts = 1000.0

        # Walks in — tracker reports real movement.
        s.record_movement(bbox, ts)

        emitted = []
        # Five minutes of alternating fence/digging events, 30s cooldown apart,
        # from a dog holding position.
        for i in range(10):
            now = ts + 30 * i
            for etype in ("dog_at_fence", "digging"):
                if not s.should_suppress(bbox, now, event_type=etype):
                    emitted.append(etype)

        # Every single digging event survived — that is the alarm path.
        assert emitted.count("digging") == 10

    def test_genuine_static_beam_is_still_suppressed(self):
        """The feature must keep working: a structural element only ever
        produces dog_at_fence, never digging, and never moves."""
        s = suppressor(static_suppression_max_hits=3,
                       static_suppression_decay_seconds=10_000.0)
        beam = [50, 60, 150, 160]
        suppressed = 0
        for i in range(20):
            if s.should_suppress(beam, 1000.0 + 30 * i, event_type="dog_at_fence"):
                suppressed += 1
        assert suppressed >= 17  # everything after the first few hits

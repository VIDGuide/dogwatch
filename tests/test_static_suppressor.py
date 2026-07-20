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

"""Unit tests for detector._clamp_bbox.

Model output needs clamping because _set_resized_input preserves aspect ratio
by scaling to fit and zero-padding the right/bottom. A box predicted partly
inside that padding maps back to coordinates beyond the frame, and because
BehaviorMonitor's paw point is the bbox *bottom-centre*, an out-of-frame bottom
edge put the paw point outside the fence polygon and the event was silently
missed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import _clamp_bbox

W, H = 640, 480


class TestInsideFrame:
    def test_fully_inside_box_is_unchanged(self):
        assert _clamp_bbox((10, 20, 100, 200), W, H) == (10, 20, 100, 200)

    def test_exact_frame_bounds_are_preserved(self):
        assert _clamp_bbox((0, 0, W, H), W, H) == (0, 0, W, H)

    def test_floats_are_coerced_to_ints(self):
        assert _clamp_bbox((10.7, 20.2, 100.9, 200.4), W, H) == (10, 20, 100, 200)


class TestClamping:
    def test_right_overflow_clamped_to_width(self):
        assert _clamp_bbox((600, 10, W + 200, 100), W, H) == (600, 10, W, 100)

    def test_bottom_overflow_clamped_to_height(self):
        # The padding strip lives at the bottom for a wide frame, so this is
        # the common real case.
        assert _clamp_bbox((10, 400, 100, H + 300), W, H) == (10, 400, 100, H)

    def test_negative_origin_clamped_to_zero(self):
        assert _clamp_bbox((-50, -20, 100, 200), W, H) == (0, 0, 100, 200)

    def test_all_edges_overflowing(self):
        assert _clamp_bbox((-10, -10, W + 10, H + 10), W, H) == (0, 0, W, H)

    def test_clamped_bottom_edge_lands_exactly_on_frame_edge(self):
        """This is what makes the zone check work: the paw point ends up at
        y == H, which BehaviorMonitor.in_zone now treats as inside (covers,
        not contains)."""
        _, _, _, y1 = _clamp_bbox((100, 400, 200, H + 500), W, H)
        assert y1 == H


class TestDegenerate:
    def test_box_entirely_right_of_frame_is_dropped(self):
        assert _clamp_bbox((W + 10, 10, W + 100, 100), W, H) is None

    def test_box_entirely_below_frame_is_dropped(self):
        assert _clamp_bbox((10, H + 10, 100, H + 100), W, H) is None

    def test_box_entirely_left_of_frame_is_dropped(self):
        assert _clamp_bbox((-100, 10, -10, 100), W, H) is None

    def test_zero_width_box_is_dropped(self):
        assert _clamp_bbox((100, 10, 100, 100), W, H) is None

    def test_zero_height_box_is_dropped(self):
        assert _clamp_bbox((10, 50, 100, 50), W, H) is None

    def test_inverted_box_is_dropped(self):
        assert _clamp_bbox((200, 200, 100, 100), W, H) is None


class TestNeverProducesOutOfRange:
    @pytest.mark.parametrize("bbox", [
        (-1000, -1000, 5000, 5000),
        (0, 0, 1, 1),
        (639, 479, 641, 481),
        (-1, -1, 0, 0),
        (320, 240, 320, 241),
    ])
    def test_result_is_always_within_bounds_or_none(self, bbox):
        out = _clamp_bbox(bbox, W, H)
        if out is None:
            return
        x0, y0, x1, y1 = out
        assert 0 <= x0 < x1 <= W
        assert 0 <= y0 < y1 <= H

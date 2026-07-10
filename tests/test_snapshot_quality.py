"""Unit tests for snapshot_quality.py's grey/corrupt-frame heuristics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_quality import active_tile_fraction, is_image_bad


def make_bgr(mean_v, noise_std=0.0, size=(160, 160), seed=0):
    rng = np.random.default_rng(seed)
    if noise_std > 0:
        arr = rng.normal(loc=mean_v, scale=noise_std, size=(size[0], size[1], 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.full((size[0], size[1], 3), mean_v, dtype=np.uint8)
    return arr


class TestIsImageBad:
    def test_none_image_is_bad(self):
        assert is_image_bad(None) is True

    def test_flat_grey_glitch_is_bad(self):
        img = make_bgr(128, noise_std=1.0)
        assert is_image_bad(img) is True

    def test_pure_black_frame_is_bad(self):
        img = make_bgr(0, noise_std=0.0)
        assert is_image_bad(img) is True

    def test_pure_white_frame_is_bad(self):
        img = make_bgr(255, noise_std=0.0)
        assert is_image_bad(img) is True

    def test_high_variance_real_scene_is_good(self):
        img = make_bgr(136, noise_std=55.0)
        assert is_image_bad(img) is False

    def test_overcast_scene_is_good_not_a_false_positive(self):
        # Regression test: the old std<40 gate rejected legitimate overcast
        # frames (mean~134, std~37). The current gate (std<12) must not.
        img = make_bgr(134, noise_std=37.0)
        assert is_image_bad(img) is False

    def test_partial_decode_blob_is_bad(self):
        # Flat grey field with a small localized high-variance patch —
        # mimics a partial HEVC decode with a "motion" artifact.
        img = make_bgr(128, noise_std=0.0)
        img[70:90, 70:90] = 255  # small bright blob in one corner region
        assert is_image_bad(img) is True

    def test_dark_night_frame_not_flagged_by_midgrey_gates(self):
        # Low mean (dark scene) with reasonable variance should not trip the
        # mid-grey (105-150) gates at all, regardless of tile spread.
        img = make_bgr(40, noise_std=20.0)
        assert is_image_bad(img) is False


class TestActiveTileFraction:
    def test_uniform_image_has_near_zero_active_tiles(self):
        gray = np.full((160, 160), 128, dtype=np.uint8)
        frac = active_tile_fraction(gray, tiles=8)
        assert frac == 0.0

    def test_full_noise_image_has_high_active_tiles(self):
        rng = np.random.default_rng(1)
        gray = rng.normal(loc=128, scale=50, size=(160, 160))
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        frac = active_tile_fraction(gray, tiles=8)
        assert frac > 0.8

    def test_localized_blob_only_activates_few_tiles(self):
        gray = np.full((160, 160), 128, dtype=np.uint8)
        gray[70:90, 70:90] = 255  # one tile-sized blob
        frac = active_tile_fraction(gray, tiles=8)
        assert frac < 0.20

    def test_too_small_frame_returns_one(self):
        gray = np.zeros((4, 4), dtype=np.uint8)
        assert active_tile_fraction(gray, tiles=8) == 1.0

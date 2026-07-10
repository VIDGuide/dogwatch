"""Unit tests for tracker.py's minimal nearest-centroid tracker."""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker import Track, CentroidTracker


class TestTrack:
    def test_centroid_calculation(self):
        assert Track._centroid((0, 0, 10, 20)) == (5.0, 10.0)
        assert Track._centroid((10, 10, 30, 50)) == (20.0, 30.0)

    def test_init_seeds_history(self):
        tr = Track(1, (0, 0, 10, 10), t=100.0)
        assert tr.id == 1
        assert tr.bbox == (0, 0, 10, 10)
        assert tr.last_seen == 100.0
        assert tr.misses == 0
        assert tr.dig_since is None
        assert len(tr.history) == 1
        assert tr.history[0] == (100.0, (5.0, 5.0), (0, 0, 10, 10))

    def test_update_appends_history_and_resets_misses(self):
        tr = Track(1, (0, 0, 10, 10), t=0.0)
        tr.misses = 3
        tr.update((5, 5, 15, 15), t=1.0)
        assert tr.bbox == (5, 5, 15, 15)
        assert tr.last_seen == 1.0
        assert tr.misses == 0
        assert len(tr.history) == 2

    def test_update_prunes_old_history_beyond_keep_seconds(self):
        tr = Track(1, (0, 0, 10, 10), t=0.0)
        tr.update((0, 0, 10, 10), t=5.0, keep_seconds=10.0)
        tr.update((0, 0, 10, 10), t=11.0, keep_seconds=10.0)
        # t=0.0 entry should be pruned (11.0 - 10.0 = 1.0 cutoff, 0.0 < 1.0)
        timestamps = [h[0] for h in tr.history]
        assert 0.0 not in timestamps
        assert 5.0 in timestamps
        assert 11.0 in timestamps

    def test_update_keeps_history_within_window(self):
        tr = Track(1, (0, 0, 10, 10), t=0.0)
        tr.update((0, 0, 10, 10), t=3.0, keep_seconds=10.0)
        tr.update((0, 0, 10, 10), t=6.0, keep_seconds=10.0)
        assert len(tr.history) == 3


class TestCentroidTracker:
    def test_new_detection_creates_track(self):
        ct = CentroidTracker()
        tracks = ct.update([(0, 0, 10, 10)], t=0.0)
        assert len(tracks) == 1
        assert 1 in tracks
        assert tracks[1].bbox == (0, 0, 10, 10)

    def test_ids_increment_across_calls(self):
        ct = CentroidTracker()
        ct.update([(0, 0, 10, 10)], t=0.0)
        tracks = ct.update([(0, 0, 10, 10), (100, 100, 110, 110)], t=1.0)
        # First detection near (5,5) should match existing track 1;
        # second detection is new -> gets id 2.
        assert set(tracks.keys()) == {1, 2}

    def test_same_dog_reassigned_to_same_track_when_close(self):
        ct = CentroidTracker(max_distance=50)
        ct.update([(0, 0, 10, 10)], t=0.0)          # track 1 centroid (5,5)
        tracks = ct.update([(2, 2, 12, 12)], t=1.0)  # centroid (7,7), close
        assert len(tracks) == 1
        assert 1 in tracks
        assert tracks[1].bbox == (2, 2, 12, 12)

    def test_far_detection_creates_new_track_not_reassignment(self):
        ct = CentroidTracker(max_distance=20)
        ct.update([(0, 0, 10, 10)], t=0.0)           # centroid (5,5)
        tracks = ct.update([(500, 500, 510, 510)], t=1.0)  # far away
        # Original track should have a miss recorded, not be moved.
        assert 1 in tracks
        assert tracks[1].misses == 1
        assert tracks[1].bbox == (0, 0, 10, 10)  # unchanged
        assert 2 in tracks
        assert tracks[2].bbox == (500, 500, 510, 510)

    def test_track_removed_after_max_misses_exceeded(self):
        ct = CentroidTracker(max_distance=20, max_misses=2)
        ct.update([(0, 0, 10, 10)], t=0.0)
        # No detections in subsequent frames -> misses accumulate.
        ct.update([], t=1.0)
        ct.update([], t=2.0)
        tracks = ct.update([], t=3.0)
        # misses: after 3 empty updates misses=3 > max_misses=2 -> removed
        assert 1 not in tracks
        assert len(tracks) == 0

    def test_track_survives_within_max_misses(self):
        ct = CentroidTracker(max_distance=20, max_misses=5)
        ct.update([(0, 0, 10, 10)], t=0.0)
        ct.update([], t=1.0)
        tracks = ct.update([], t=2.0)
        assert 1 in tracks
        assert tracks[1].misses == 2

    def test_multiple_detections_each_get_unique_tracks(self):
        ct = CentroidTracker(max_distance=20)
        tracks = ct.update([(0, 0, 10, 10), (200, 200, 210, 210)], t=0.0)
        assert len(tracks) == 2
        assert set(tracks.keys()) == {1, 2}

    def test_nearest_match_preferred_over_further_one(self):
        ct = CentroidTracker(max_distance=100)
        ct.update([(0, 0, 10, 10)], t=0.0)  # centroid (5,5), track 1
        # Two detections in the next frame: one very close, one further but
        # still in range. The tracker should claim the nearer one for track 1
        # and spawn a new track for the further one.
        tracks = ct.update([(1, 1, 11, 11), (50, 50, 60, 60)], t=1.0)
        assert tracks[1].bbox == (1, 1, 11, 11)
        assert 2 in tracks
        assert tracks[2].bbox == (50, 50, 60, 60)

    def test_distance_uses_euclidean_metric(self):
        # Sanity check the math.dist usage matches expected Euclidean distance.
        c1 = Track._centroid((0, 0, 10, 10))   # (5, 5)
        c2 = Track._centroid((5, 5, 15, 15))   # (10, 10)
        assert math.isclose(math.dist(c1, c2), math.dist((5, 5), (10, 10)))

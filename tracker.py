"""tracker.py — minimal nearest-centroid tracker.

Enough to keep stable IDs for one or several dogs across frames so the
behaviour layer can reason about "is this same dog still here / still still".
Not a Kalman/DeepSORT setup — deliberately simple, since a backyard scene is
low-density and frames are sampled at only a few fps.
"""
import math


class Track:
    def __init__(self, tid, bbox, t):
        self.id = tid
        self.bbox = bbox
        self.last_seen = t
        self.misses = 0
        self.dig_since = None                       # used by BehaviorMonitor
        self.history = [(t, self._centroid(bbox), bbox)]

    @staticmethod
    def _centroid(b):
        x0, y0, x1, y1 = b
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def update(self, bbox, t, keep_seconds=10.0):
        self.bbox = bbox
        self.last_seen = t
        self.misses = 0
        self.history.append((t, self._centroid(bbox), bbox))
        cutoff = t - keep_seconds
        self.history = [h for h in self.history if h[0] >= cutoff]


class CentroidTracker:
    def __init__(self, max_distance=120, max_misses=5):
        self.tracks = {}
        self.max_distance = max_distance            # px; raise for distant cams
        self.max_misses = max_misses
        self._next = 1

    def update(self, detections, t):
        """detections: list of bbox tuples. Returns {tid: Track}."""
        centroids = [Track._centroid(d) for d in detections]
        assigned = set()

        for tid, tr in list(self.tracks.items()):
            last_c = tr.history[-1][1]
            best, best_d = None, self.max_distance
            for i, c in enumerate(centroids):
                if i in assigned:
                    continue
                d = math.dist(last_c, c)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                tr.update(detections[best], t)
                assigned.add(best)
            else:
                tr.misses += 1

        for i, d in enumerate(detections):
            if i not in assigned:
                self.tracks[self._next] = Track(self._next, d, t)
                self._next += 1

        for tid in [tid for tid, tr in self.tracks.items()
                    if tr.misses > self.max_misses]:
            del self.tracks[tid]

        return self.tracks

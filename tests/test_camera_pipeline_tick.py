"""Unit tests for CameraPipeline.tick's event bookkeeping.

tick() is the heart of the detector — grab, crop, gate, detect, track, evaluate,
publish — and had no test coverage at all, because constructing a CameraPipeline
normally requires a live RTSP stream and an Edge TPU. These tests build one via
``__new__`` and wire stub collaborators around the real MotionGate,
BehaviorMonitor, CentroidTracker and StaticSuppressor, so the orchestration can
be exercised on plain Python.

They cover two specific bugs:

  * Event-clip filenames carried no camera name (``dig_<ts>_<track>.jpg``).
    ``clip_dir`` is per-camera in principle, but both shipped configs point at
    ``clips``, so two cameras firing on the same track id in the same second
    wrote the same path and one silently overwrote the other.
  * The static suppressor was told about movement from tracks the tracker was
    holding open *without* a detection on this frame. Their history is frozen,
    so the same historical movement was re-reported every frame, extending the
    suppression grace period for a track nothing was actually seeing.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camera_pipeline
from behavior import BehaviorMonitor
from camera_pipeline import CameraPipeline
from motion_gate import MotionGate
from static_suppressor import StaticSuppressor
from tracker import CentroidTracker

W = H = 256

# Zone covering the whole frame, so any detection is "at the fence".
CFG = {
    "fence_zone_norm": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    "stationary_px": 25,
    "motion_energy_thresh": 0.06,
    # 0.0 sustain: BehaviorMonitor.evaluate stamps dig_since from
    # time.time() while is_stationary reads the track's own timestamps, so a
    # non-zero sustain window can't be driven from synthetic tick times. The
    # existing behaviour tests use the same approach.
    "dig_sustain_seconds": 0.0,
    "event_cooldown_seconds": 0,
    "min_consecutive": 1,
    "motion_gate_enabled": False,      # exercise detection on every frame
    "static_suppression_enabled": True,
}


def scene(seed):
    """A frame with real spatial structure, so is_image_bad() accepts it."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)


class StubGrab:
    """Frame source with a scripted sequence and a fixed capture timestamp."""

    def __init__(self, frames, ts=1000.0):
        self.frames = list(frames)
        self.ts = ts

    def read_with_ts(self):
        frame = self.frames.pop(0) if self.frames else None
        return frame, self.ts

    def frame_age(self, now=None):
        return 0.0

    def health(self):
        return {"frame_age": 0.0, "has_frame": True, "thread_alive": True,
                "reconnects": 0, "consecutive_failures": 0, "last_error": None}

    def stop(self):
        pass


class RecordingWriter:
    """Captures submissions instead of performing them."""

    def __init__(self):
        self.calls = []

    def submit(self, fn, *args, label=None, **kwargs):
        self.calls.append({"fn": fn, "args": args, "kwargs": kwargs,
                           "label": label})
        return True

    def stats(self):
        return {}

    def stop(self, timeout=None):
        pass


class ScriptedDetector:
    """Returns a scripted list of detections per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def detect(self, frame, score_threshold=None):
        self.calls += 1
        return self.script.pop(0) if self.script else []


def make_pipeline(frames, name="rear-east", clip_dir="clips", cfg=None):
    """Build a CameraPipeline with no real I/O, bypassing __init__."""
    cfg = dict(CFG, **(cfg or {}))
    pipe = CameraPipeline.__new__(CameraPipeline)
    pipe.name = name
    pipe.grab = StubGrab(frames)
    pipe.pub = None                      # no MQTT
    pipe.writer = RecordingWriter()
    pipe.event_store = type("ES", (), {"log_event": lambda *a, **k: None,
                                       "maybe_prune": lambda *a, **k: None})()
    pipe.debug_capture = type("DC", (), {"save": lambda *a, **k: None,
                                         "cleanup": lambda *a, **k: None})()
    pipe.score_threshold = 0.4
    pipe.stale_after = 30.0
    pipe._stale = False
    pipe._last_stale_log = 0.0
    pipe.crop = None
    pipe.crop_norm = None
    pipe.w, pipe.h = W, H
    pipe.full_w, pipe.full_h = W, H
    pipe.clip_dir = clip_dir
    pipe.cooldown = 0
    pipe.clip_retention_days = 0
    pipe._last_debug_cleanup = 1e12      # suppress the hourly sweep
    pipe.motion_gate = MotionGate(cfg)
    pipe.monitor = BehaviorMonitor(cfg, W, H)
    pipe.tracker = CentroidTracker(max_distance=120, max_misses=5)
    pipe.static_suppressor = StaticSuppressor(cfg, camera_name=name,
                                             log=lambda *a: None)
    return pipe


def clip_writes(pipe):
    return [c for c in pipe.writer.calls if c["label"] == "clip_imwrite"]


def tick_at(pipe, detector, t):
    """Tick with a frame captured at *t*, i.e. a fresh (non-stale) frame."""
    pipe.grab.ts = t
    pipe.tick(detector, t)


# --------------------------------------------------------------------------
# Event clip filenames must identify the camera
# --------------------------------------------------------------------------

class TestClipFilenames:
    def _fire_digging(self, pipe, detector, frames, t0=5000.0):
        """Drive enough ticks for the digging sustain window to elapse."""
        for i in range(len(frames)):
            tick_at(pipe, detector, t0 + i * 0.4)

    def test_clip_filename_contains_the_camera_name(self):
        # Same bbox every frame (stationary) with changing pixels inside it
        # (busy box) is the digging heuristic.
        bbox = (100, 100, 160, 180)
        frames = [scene(i) for i in range(10)]
        pipe = make_pipeline(frames, name="rear-east")
        detector = ScriptedDetector([[{"bbox": bbox, "score": 0.9}]] * 10)

        self._fire_digging(pipe, detector, frames)

        writes = clip_writes(pipe)
        assert writes, "expected a digging clip to be written"
        path = writes[0]["args"][0]
        assert "rear-east" in os.path.basename(path), path

    def test_two_cameras_do_not_collide_on_the_same_path(self):
        # The actual bug: identical timestamp, identical track id, one shared
        # clip_dir — the two cameras used to produce the same filename.
        bbox = (100, 100, 160, 180)
        paths = []
        for name in ("camera", "rear-east"):
            pipe = make_pipeline([scene(i) for i in range(10)], name=name)
            detector = ScriptedDetector([[{"bbox": bbox, "score": 0.9}]] * 10)
            self._fire_digging(pipe, detector, [None] * 10, t0=5000.0)
            writes = clip_writes(pipe)
            assert writes, f"no clip written for {name}"
            paths.append(writes[0]["args"][0])

        assert paths[0] != paths[1], f"both cameras wrote {paths[0]}"

    def test_clip_goes_into_the_configured_clip_dir(self):
        bbox = (100, 100, 160, 180)
        pipe = make_pipeline([scene(i) for i in range(10)],
                             clip_dir="clips-rear-east")
        detector = ScriptedDetector([[{"bbox": bbox, "score": 0.9}]] * 10)
        self._fire_digging(pipe, detector, [None] * 10)

        writes = clip_writes(pipe)
        assert writes
        assert os.path.dirname(writes[0]["args"][0]) == "clips-rear-east"


# --------------------------------------------------------------------------
# Movement must only be reported for tracks detected on this frame
# --------------------------------------------------------------------------

class TestMovementReporting:
    def test_movement_not_reported_for_undetected_tracks(self):
        recorded = []

        pipe = make_pipeline([scene(i) for i in range(12)])
        pipe.static_suppressor.record_movement = (
            lambda bbox, ts: recorded.append(ts))

        # Two frames of real movement, then nothing detected at all. The track
        # stays alive (max_misses=5) with a frozen history.
        detector = ScriptedDetector([
            [{"bbox": (10, 10, 60, 90), "score": 0.9}],
            [{"bbox": (90, 10, 140, 90), "score": 0.9}],   # +80px: movement
            [], [], [], [],
        ])
        for i in range(6):
            tick_at(pipe, detector, 5000.0 + i * 0.4)

        # Exactly one movement report — from the frame the dog was actually
        # detected moving on, not one per frame for as long as the track lives.
        assert len(recorded) == 1, recorded

    def test_movement_still_reported_while_the_dog_is_tracked(self):
        # Guard against over-correcting: a genuinely moving, continuously
        # detected dog must still clear static suppression.
        recorded = []

        pipe = make_pipeline([scene(i) for i in range(12)])
        pipe.static_suppressor.record_movement = (
            lambda bbox, ts: recorded.append(ts))

        detector = ScriptedDetector([
            [{"bbox": (10, 10, 60, 90), "score": 0.9}],
            [{"bbox": (90, 10, 140, 90), "score": 0.9}],
            [{"bbox": (170, 10, 220, 90), "score": 0.9}],
        ])
        for i in range(3):
            tick_at(pipe, detector, 5000.0 + i * 0.4)

        assert len(recorded) == 2, recorded


# --------------------------------------------------------------------------
# Staleness gate still short-circuits before any detection work
# --------------------------------------------------------------------------

class TestStalenessShortCircuit:
    def test_stale_frame_skips_detection_entirely(self):
        pipe = make_pipeline([scene(1)])
        pipe.grab.ts = 1.0               # ancient
        detector = ScriptedDetector([[{"bbox": (10, 10, 60, 90), "score": 0.9}]])

        pipe.tick(detector, 5000.0)

        assert detector.calls == 0
        assert pipe._stale is True

    def test_missing_frame_marks_stale_without_detecting(self):
        pipe = make_pipeline([])         # grabber has nothing
        detector = ScriptedDetector([])

        pipe.tick(detector, 5000.0)

        assert detector.calls == 0
        assert pipe._stale is True

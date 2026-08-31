"""Unit tests for per-camera score_threshold handling.

The bug these lock down: dogwatch.py built ONE shared DogDetector from
``cfgs[0]`` and passed ``cfgs[0]["score_threshold"]`` into it. The threshold was
then applied inside ``DogDetector.detect()``, so every camera after the first
silently ran on camera #1's threshold — while the README documented
``score_threshold`` as a per-camera key and specifically advised raising it per
camera to suppress that camera's false positives.

The detector genuinely must be shared (one process can bind the TPU), so the fix
is to make the confidence filter a per-call argument: build the shared
interpreter at the lowest threshold in the fleet, and let each camera apply its
own to the results.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detector
import dogwatch
from detector import DEFAULT_SCORE_THRESHOLD, resolve_score_threshold


# --------------------------------------------------------------------------
# resolve_score_threshold
# --------------------------------------------------------------------------

class TestResolveScoreThreshold:
    def test_valid_value_passes_through(self):
        assert resolve_score_threshold({"score_threshold": 0.55}) == 0.55

    def test_missing_key_uses_documented_default(self):
        # Previously cfgs[0]["score_threshold"] raised KeyError on a config that
        # omitted the key, despite the README documenting a default of 0.4.
        assert resolve_score_threshold({}) == DEFAULT_SCORE_THRESHOLD

    def test_default_matches_readme_and_detector_default(self):
        assert DEFAULT_SCORE_THRESHOLD == 0.4

    def test_numeric_string_is_accepted(self):
        assert resolve_score_threshold({"score_threshold": "0.6"}) == 0.6

    def test_upper_bound_one_is_valid(self):
        assert resolve_score_threshold({"score_threshold": 1.0}) == 1.0

    @pytest.mark.parametrize("bad", [55, 100, 1.5, -0.2, 0, 0.0])
    def test_out_of_range_falls_back_to_default(self, bad):
        # score_threshold: 55 (meaning "55%") would otherwise suppress every
        # detection forever, looking exactly like "the model never sees my dog".
        assert resolve_score_threshold({"score_threshold": bad}) == DEFAULT_SCORE_THRESHOLD

    @pytest.mark.parametrize("bad", ["high", None, "", [], {}])
    def test_non_numeric_falls_back_to_default(self, bad):
        assert resolve_score_threshold({"score_threshold": bad}) == DEFAULT_SCORE_THRESHOLD

    def test_out_of_range_warns_with_camera_name(self, capsys):
        resolve_score_threshold({"score_threshold": 55}, "rear-east")
        out = capsys.readouterr().out
        assert "rear-east" in out
        assert "WARNING" in out

    def test_valid_value_is_silent(self, capsys):
        resolve_score_threshold({"score_threshold": 0.5}, "rear-east")
        assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Per-call threshold override in detect()
# --------------------------------------------------------------------------

class FakeInterp:
    """Interpreter stub returning one detection at a fixed score."""

    def __init__(self, score, class_id=17, box=(0.1, 0.1, 0.5, 0.5)):
        self._score = score
        self._class_id = class_id
        self._box = box
        self.invocations = 0

    def get_input_details(self):
        return [{"index": "input", "shape": (1, 300, 300, 3)}]

    def get_output_details(self):
        return [{"index": i} for i in range(4)]

    def tensor(self, index):
        if index == "input":
            return lambda: np.zeros((1, 300, 300, 3), dtype=np.uint8)
        ymin, xmin, ymax, xmax = self._box
        outputs = [
            np.array([[[ymin, xmin, ymax, xmax]]], dtype=np.float32),  # boxes
            np.array([[float(self._class_id)]], dtype=np.float32),     # class ids
            np.array([[self._score]], dtype=np.float32),               # scores
            np.array([1.0], dtype=np.float32),                         # count
        ]
        return lambda: outputs[index]

    def _get_full_signature_list(self):
        return {}

    def allocate_tensors(self):
        pass

    def invoke(self):
        self.invocations += 1


def make_detector(score, instance_threshold=0.4):
    """Build a DogDetector around a stub interpreter, no TPU involved."""
    det = detector.DogDetector.__new__(detector.DogDetector)
    det.interp = FakeInterp(score)
    det.score_threshold = instance_threshold
    det.labels = {17: "dog"}
    det.target_ids = {17}
    return det


FRAME = np.zeros((300, 300, 3), dtype=np.uint8)


class TestDetectThresholdOverride:
    def test_instance_threshold_used_when_no_override(self):
        det = make_detector(score=0.50, instance_threshold=0.40)
        assert len(det.detect(FRAME)) == 1

    def test_instance_threshold_rejects_below_default(self):
        det = make_detector(score=0.30, instance_threshold=0.40)
        assert det.detect(FRAME) == []

    def test_override_stricter_than_instance_rejects(self):
        """The core fix: a camera configured at 0.55 must reject a 0.50
        detection even though the shared detector's floor is 0.40."""
        det = make_detector(score=0.50, instance_threshold=0.40)
        assert det.detect(FRAME, score_threshold=0.55) == []

    def test_override_looser_than_instance_accepts(self):
        det = make_detector(score=0.30, instance_threshold=0.55)
        assert len(det.detect(FRAME, score_threshold=0.20)) == 1

    def test_override_does_not_mutate_instance_default(self):
        det = make_detector(score=0.50, instance_threshold=0.40)
        det.detect(FRAME, score_threshold=0.99)
        assert det.score_threshold == 0.40
        # ...and the next call without an override still uses 0.40.
        assert len(det.detect(FRAME)) == 1

    def test_two_cameras_get_different_verdicts_from_one_detector(self):
        """End-to-end shape of the bug: one shared detector, one frame, two
        cameras with different thresholds must disagree."""
        shared = make_detector(score=0.50, instance_threshold=0.40)  # fleet floor
        lenient = shared.detect(FRAME, score_threshold=0.40)   # camera A
        strict = shared.detect(FRAME, score_threshold=0.55)    # camera B
        assert len(lenient) == 1
        assert strict == []

    def test_score_exactly_at_threshold_is_accepted(self):
        # _get_objects filters with `scores[i] < threshold`, so equality passes.
        det = make_detector(score=0.55)
        assert len(det.detect(FRAME, score_threshold=0.55)) == 1

    def test_non_target_class_still_filtered_regardless_of_threshold(self):
        det = make_detector(score=0.99)
        det.interp = FakeInterp(0.99, class_id=1)  # person, not dog
        assert det.detect(FRAME, score_threshold=0.1) == []

    def test_inference_still_runs_once_per_detect(self):
        det = make_detector(score=0.9)
        det.detect(FRAME, score_threshold=0.95)  # rejected by threshold...
        assert det.interp.invocations == 1       # ...but inference still happened


# --------------------------------------------------------------------------
# Fleet-level wiring
# --------------------------------------------------------------------------

class TestFleetFloor:
    """The shared interpreter must be built at the LOWEST threshold in the
    fleet, or a lenient camera would be starved of detections that the strictest
    camera's threshold filtered out before it ever saw them."""

    def test_floor_is_the_minimum_across_cameras(self):
        cfgs = [{"score_threshold": 0.55}, {"score_threshold": 0.40},
                {"score_threshold": 0.70}]
        thresholds = [resolve_score_threshold(c) for c in cfgs]
        assert min(thresholds) == 0.40

    def test_floor_falls_back_to_default_for_configs_without_the_key(self):
        cfgs = [{"score_threshold": 0.55}, {}]
        thresholds = [resolve_score_threshold(c) for c in cfgs]
        assert min(thresholds) == DEFAULT_SCORE_THRESHOLD

    def test_invalid_threshold_does_not_drag_the_floor_to_zero(self):
        # A bad value must not become a 0.0 floor that disables all filtering.
        cfgs = [{"score_threshold": 0.6}, {"score_threshold": 0}]
        thresholds = [resolve_score_threshold(c) for c in cfgs]
        assert min(thresholds) == DEFAULT_SCORE_THRESHOLD
        assert min(thresholds) > 0


class TestCameraNameFor:
    @pytest.mark.parametrize("path,expected", [
        ("config.json", "camera"),
        ("config-rear-east.json", "rear-east"),
        ("/opt/dogwatch/config-rear-east.json", "rear-east"),
        ("config-side.json", "side"),
    ])
    def test_name_derivation(self, path, expected):
        assert dogwatch.camera_name_for(path) == expected


class TestSharedKeyConflictWarnings:
    """model_path/labels_path genuinely cannot be per-camera — only one
    interpreter exists. Previously a divergent value was ignored in silence."""

    def test_divergent_model_path_warns(self, capsys):
        cfgs = [{"model_path": "a.tflite"}, {"model_path": "b.tflite"}]
        dogwatch.warn_on_shared_key_conflicts(cfgs, ["camera", "rear-east"])
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "b.tflite" in out
        assert "rear-east" in out

    def test_divergent_labels_path_warns(self, capsys):
        cfgs = [{"labels_path": "x.txt"}, {"labels_path": "y.txt"}]
        dogwatch.warn_on_shared_key_conflicts(cfgs, ["camera", "rear-east"])
        assert "y.txt" in capsys.readouterr().out

    def test_identical_paths_are_silent(self, capsys):
        cfgs = [{"model_path": "a.tflite", "labels_path": "l.txt"}] * 3
        dogwatch.warn_on_shared_key_conflicts(cfgs, ["a", "b", "c"])
        assert capsys.readouterr().out == ""

    def test_absent_key_in_later_config_is_silent(self, capsys):
        # Omitting the key is fine — it just inherits the shared model.
        cfgs = [{"model_path": "a.tflite"}, {}]
        dogwatch.warn_on_shared_key_conflicts(cfgs, ["camera", "rear-east"])
        assert capsys.readouterr().out == ""

    def test_single_camera_never_warns(self, capsys):
        dogwatch.warn_on_shared_key_conflicts([{"model_path": "a.tflite"}], ["camera"])
        assert capsys.readouterr().out == ""

    def test_warning_mentions_that_score_threshold_is_different(self, capsys):
        """The warning should teach the distinction, since the whole confusion
        was 'which keys are actually per-camera?'."""
        cfgs = [{"model_path": "a.tflite"}, {"model_path": "b.tflite"}]
        dogwatch.warn_on_shared_key_conflicts(cfgs, ["camera", "rear-east"])
        assert "score_threshold" in capsys.readouterr().out


class TestExampleConfigsAreHonoured:
    """Regression guard tied to the actual shipped example configs: both set
    score_threshold: 0.55, and rear-east's used to be silently discarded."""

    def test_rear_east_example_threshold_is_resolved(self):
        import json
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "config-rear-east.example.json")) as f:
            cfg = json.load(f)
        assert resolve_score_threshold(cfg, "rear-east") == cfg["score_threshold"]

    def test_both_example_configs_are_in_valid_range(self):
        import json
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("config.example.json", "config-rear-east.example.json"):
            with open(os.path.join(repo, name)) as f:
                cfg = json.load(f)
            resolved = resolve_score_threshold(cfg, name)
            assert resolved == cfg["score_threshold"]
            assert 0.0 < resolved <= 1.0

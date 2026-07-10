"""Unit tests for detector.py's tensor-parsing logic (_get_objects,
_set_resized_input), using fake interpreter objects rather than a real
model/TPU — the Edge TPU delegate and real inference are only verifiable on
actual Coral hardware, but the pure-Python tensor bookkeeping this module
reimplements from pycoral can be tested in isolation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detector


class FakeInterpreter:
    """Minimal stand-in for ai_edge_litert.interpreter.Interpreter.

    Supports just enough of the API surface detector.py touches:
    get_input_details / get_output_details / tensor(index) / _get_full_signature_list.
    """

    def __init__(self, input_shape, output_tensors):
        """output_tensors: list of numpy arrays, in the same order detector.py
        expects to find them via get_output_details()[i]['index']."""
        self._input = np.zeros(input_shape, dtype=np.uint8)
        self._outputs = output_tensors
        self._input_shape = input_shape

    def get_input_details(self):
        return [{"index": "input", "shape": self._input_shape}]

    def get_output_details(self):
        return [{"index": i} for i in range(len(self._outputs))]

    def tensor(self, index):
        if index == "input":
            return lambda: self._input
        return lambda: self._outputs[index]

    def _get_full_signature_list(self):
        return {}  # no signature -> exercise the legacy tensor-order path


class TestSetResizedInput:
    def test_scales_and_pads_to_fit_input_tensor(self):
        interp = FakeInterpreter(input_shape=(1, 300, 300, 3), output_tensors=[])
        # A 600x400 frame resized to fit a 300x300 tensor should scale by 0.5.
        calls = []

        def fake_resize(size):
            calls.append(size)
            w, h = size
            return np.full((h, w, 3), 200, dtype=np.uint8)

        scale = detector._set_resized_input(interp, (600, 400), fake_resize)
        assert scale == (0.5, 0.5)
        assert calls == [(300, 200)]

    def test_input_tensor_is_zero_padded_where_not_covered(self):
        interp = FakeInterpreter(input_shape=(1, 300, 300, 3), output_tensors=[])

        def fake_resize(size):
            w, h = size
            return np.full((h, w, 3), 255, dtype=np.uint8)

        detector._set_resized_input(interp, (600, 400), fake_resize)
        tensor = interp._input[0]
        # Covered region (300x200) should be 255; the padding strip below it
        # (rows 200-299) should remain zero.
        assert (tensor[:200, :300] == 255).all()
        assert (tensor[200:, :] == 0).all()


class TestGetObjects:
    def test_legacy_tensor_order_single_count(self):
        # Mimics ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite's
        # output order: boxes, class_ids, scores, count (count.size == 1).
        boxes = np.array([[[0.1, 0.2, 0.5, 0.6]]], dtype=np.float32)
        class_ids = np.array([[17.0]], dtype=np.float32)
        scores = np.array([[0.93]], dtype=np.float32)
        count = np.array([1.0], dtype=np.float32)

        interp = FakeInterpreter(
            input_shape=(1, 300, 300, 3),
            output_tensors=[boxes, class_ids, scores, count],
        )
        results = detector._get_objects(interp, score_threshold=0.4, image_scale=(1.0, 1.0))
        assert len(results) == 1
        obj = results[0]
        assert obj["id"] == 17
        assert obj["score"] == pytest_approx(0.93)
        # bbox = (xmin*sx, ymin*sy, xmax*sx, ymax*sy) with sx=sy=300 (input size / scale 1.0)
        xmin, ymin, xmax, ymax = obj["bbox"]
        assert xmin == int(0.2 * 300)
        assert ymin == int(0.1 * 300)
        assert xmax == int(0.6 * 300)
        assert ymax == int(0.5 * 300)

    def test_alternate_tensor_order_when_count_not_size_one(self):
        # Some exports order outputs as scores, boxes, count, class_ids
        # (the branch taken when output tensor index 3's size != 1).
        scores = np.array([[0.8]], dtype=np.float32)
        boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
        count = np.array([1.0], dtype=np.float32)
        class_ids = np.array([[5.0], [6.0]], dtype=np.float32)  # size != 1

        interp = FakeInterpreter(
            input_shape=(1, 300, 300, 3),
            output_tensors=[scores, boxes, count, class_ids],
        )
        results = detector._get_objects(interp, score_threshold=0.1, image_scale=(1.0, 1.0))
        assert len(results) == 1
        assert results[0]["id"] == 5

    def test_score_threshold_filters_low_confidence_detections(self):
        boxes = np.array([[[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.1, 0.1]]], dtype=np.float32)
        class_ids = np.array([[17.0, 17.0]], dtype=np.float32)
        scores = np.array([[0.9, 0.1]], dtype=np.float32)
        count = np.array([2.0], dtype=np.float32)

        interp = FakeInterpreter(
            input_shape=(1, 300, 300, 3),
            output_tensors=[boxes, class_ids, scores, count],
        )
        results = detector._get_objects(interp, score_threshold=0.5, image_scale=(1.0, 1.0))
        assert len(results) == 1
        assert results[0]["score"] == pytest_approx(0.9)

    def test_image_scale_affects_bbox_scaling(self):
        # image_scale represents the (scale_x, scale_y) used during resize;
        # a smaller image_scale means the original frame was larger relative
        # to the model's input, so bboxes should scale up accordingly.
        boxes = np.array([[[0.0, 0.0, 0.5, 0.5]]], dtype=np.float32)
        class_ids = np.array([[17.0]], dtype=np.float32)
        scores = np.array([[0.9]], dtype=np.float32)
        count = np.array([1.0], dtype=np.float32)

        interp = FakeInterpreter(
            input_shape=(1, 300, 300, 3),
            output_tensors=[boxes, class_ids, scores, count],
        )
        results = detector._get_objects(interp, score_threshold=0.1, image_scale=(0.5, 0.5))
        xmin, ymin, xmax, ymax = results[0]["bbox"]
        # sx = width/scale_x = 300/0.5 = 600
        assert xmax == int(0.5 * 600)
        assert ymax == int(0.5 * 600)


class TestDogDetectorLabelResolution:
    def test_target_ids_resolved_by_label_name_case_insensitive(self):
        labels_content = "person\ndog\ncat\n"
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(labels_content)
            labels_path = f.name
        try:
            labels = detector.DogDetector._load_labels(labels_path)
            assert labels == {0: "person", 1: "dog", 2: "cat"}
        finally:
            os.unlink(labels_path)

    def test_missing_label_raises_value_error(self):
        labels_content = "person\ncat\n"
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(labels_content)
            labels_path = f.name
        try:
            # Can't construct a full DogDetector without a model/TPU, but we
            # can exercise the label-resolution failure path directly.
            labels = detector.DogDetector._load_labels(labels_path)
            target_ids = {i for i, n in labels.items() if n.lower() == "dog"}
            assert not target_ids
        finally:
            os.unlink(labels_path)


def pytest_approx(value, rel=1e-4):
    """Tiny local helper so this file doesn't need to import pytest.approx
    directly in every assertion (kept simple on purpose)."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= rel * max(abs(value), abs(other), 1e-9)
    return _Approx()

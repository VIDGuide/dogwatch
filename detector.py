"""detector.py — thin wrapper around ai-edge-litert + the Edge TPU delegate,
filtered to the 'dog' class.

Previously used pycoral (Google's convenience wrapper around tflite_runtime),
but pycoral is abandoned upstream and only ever shipped cp39 wheels, which is
what pinned this whole project to Python 3.9 and, downstream of that, to
numpy 1.x (see README "Known limitations" / GitHub issue #1 for the full
history). pycoral's actual surface area used here was small — a delegate
loader and two adapter functions, both plain Python with no C bindings — so
this reimplements them directly against `ai_edge_litert`, which:
  * ships wheels for Python 3.9 through 3.14 (no more cp39 ceiling)
  * exposes the same `Interpreter`/`load_delegate` API tflite_runtime did,
    so this is a like-for-like swap, not a rewrite of the detection logic
  * has no numpy version ceiling, which is what forced numpy/opencv's older
    pins in the Dockerfile

The output-tensor parsing in `_get_objects` mirrors pycoral's
`adapters.detect.get_objects` exactly (including the newer "signature" path
for models that expose one), so behavior is unchanged regardless of which
SSD-style detection model is loaded.
"""
import platform

import cv2
from ai_edge_litert.interpreter import Interpreter, load_delegate

_EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


def _make_interpreter(model_path):
    """Load *model_path* with the Edge TPU delegate attached."""
    delegate = load_delegate(_EDGETPU_SHARED_LIB)
    return Interpreter(model_path=model_path, experimental_delegates=[delegate])


def _input_size(interpreter):
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]
    return width, height


def _input_tensor(interpreter):
    index = interpreter.get_input_details()[0]["index"]
    return interpreter.tensor(index)()[0]


def _set_resized_input(interpreter, size, resize):
    """Copy a resized, zero-padded image into the model's input tensor.

    Mirrors pycoral's adapters.common.set_resized_input: preserves aspect
    ratio by scaling to fit, then pads the rest with zeros, so callers don't
    need to worry about non-square input tensors.
    """
    width, height = _input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = _input_tensor(interpreter)
    tensor.fill(0)
    _, _, channel = tensor.shape
    result = resize((w, h))
    tensor[:h, :w] = result.reshape((h, w, channel))
    return (scale, scale)


def _output_tensor(interpreter, i):
    index = interpreter.get_output_details()[i]["index"]
    return interpreter.tensor(index)()


def _get_objects(interpreter, score_threshold, image_scale):
    """Return [{'id', 'score', 'bbox': (xmin,ymin,xmax,ymax)}, ...].

    Output tensor layout for TFLite_Detection_PostProcess-based SSD models
    varies by export; this checks for a model signature first (newer
    exports), then falls back to the same tensor-order heuristics pycoral
    used for older exports like the stock
    ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite.
    """
    signature_list = interpreter._get_full_signature_list()  # noqa: SLF001
    if signature_list:
        if len(signature_list) > 1:
            raise ValueError("Only support model with one signature.")
        signature = signature_list[next(iter(signature_list))]
        count = int(interpreter.tensor(signature["outputs"]["output_0"])()[0])
        scores = interpreter.tensor(signature["outputs"]["output_1"])()[0]
        class_ids = interpreter.tensor(signature["outputs"]["output_2"])()[0]
        boxes = interpreter.tensor(signature["outputs"]["output_3"])()[0]
    elif _output_tensor(interpreter, 3).size == 1:
        boxes = _output_tensor(interpreter, 0)[0]
        class_ids = _output_tensor(interpreter, 1)[0]
        scores = _output_tensor(interpreter, 2)[0]
        count = int(_output_tensor(interpreter, 3)[0])
    else:
        scores = _output_tensor(interpreter, 0)[0]
        boxes = _output_tensor(interpreter, 1)[0]
        count = int(_output_tensor(interpreter, 2)[0])
        class_ids = _output_tensor(interpreter, 3)[0]

    width, height = _input_size(interpreter)
    scale_x, scale_y = image_scale
    sx, sy = width / scale_x, height / scale_y

    out = []
    for i in range(count):
        if scores[i] < score_threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        out.append({
            "id": int(class_ids[i]),
            "score": float(scores[i]),
            "bbox": (
                int(xmin * sx), int(ymin * sy), int(xmax * sx), int(ymax * sy)
            ),
        })
    return out


class DogDetector:
    def __init__(self, model_path, labels_path, score_threshold=0.4,
                 target_label="dog"):
        self.interp = _make_interpreter(model_path)
        self.interp.allocate_tensors()
        self.score_threshold = score_threshold
        self.labels = self._load_labels(labels_path)
        # COCO label files vary ("dog" may be id 17 or 18) — resolve by name.
        self.target_ids = {i for i, n in self.labels.items()
                           if n.lower() == target_label.lower()}
        if not self.target_ids:
            raise ValueError(f"'{target_label}' not found in {labels_path}")

    @staticmethod
    def _load_labels(path):
        labels = {}
        with open(path) as f:
            for idx, line in enumerate(f):
                name = line.strip()
                if name:
                    labels[idx] = name
        return labels

    def detect(self, frame):
        """Return [{'bbox': (x0,y0,x1,y1) in pixels, 'score': float}, ...]."""
        h, w = frame.shape[:2]
        scale = _set_resized_input(
            self.interp, (w, h), lambda size: cv2.resize(frame, size))
        self.interp.invoke()
        out = []
        for obj in _get_objects(self.interp, self.score_threshold, scale):
            if obj["id"] in self.target_ids:
                out.append({"bbox": obj["bbox"], "score": obj["score"]})
        return out

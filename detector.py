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

_EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


def _make_interpreter(model_path):
    """Load *model_path* with the Edge TPU delegate attached.

    ``ai_edge_litert`` is imported here rather than at module scope so the
    pure-Python parts of this module (tensor bookkeeping in ``_get_objects`` /
    ``_set_resized_input``, bbox clamping, label parsing) can be imported and
    unit tested without the ML runtime installed. Those helpers are
    reimplementations of pycoral logic and are exactly the parts worth testing
    off-hardware — but the module-level import meant ``tests/test_detector.py``
    could not even be collected, so CI skipped the whole file.
    """
    from ai_edge_litert.interpreter import Interpreter, load_delegate

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

    Only the *padding* strips are zeroed, not the whole tensor. The previous
    ``tensor.fill(0)`` wrote every byte of a 512x512x3 buffer (~800KB) on every
    single inference, then immediately overwrote most of it with the resized
    frame. The region the frame covers needs no pre-zeroing at all.
    """
    width, height = _input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = _input_tensor(interpreter)
    _, _, channel = tensor.shape
    result = resize((w, h))
    tensor[:h, :w] = result.reshape((h, w, channel))
    # Zero only what the frame does not cover (right and bottom strips).
    if h < height:
        tensor[h:, :] = 0
    if w < width:
        tensor[:h, w:] = 0
    return (scale, scale)


def _output_tensor(interpreter, i):
    index = interpreter.get_output_details()[i]["index"]
    return interpreter.tensor(index)()


def resolve_output_layout(interpreter):
    """Determine once where each output tensor lives.

    Output tensor layout for TFLite_Detection_PostProcess-based SSD models
    varies by export: newer exports expose a signature, older ones (like the
    stock ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite) are
    identified by tensor-order heuristics — the same ones pycoral used.

    This is resolved **once at load time** rather than per inference. It used to
    run inside `_get_objects`, which meant a private-API call
    (`_get_full_signature_list`) plus a re-read of every output tensor's details
    on every single frame, on the critical path. The layout cannot change for a
    loaded model, so there is nothing to re-derive.
    """
    signature_list = interpreter._get_full_signature_list()  # noqa: SLF001
    if signature_list:
        if len(signature_list) > 1:
            raise ValueError("Only support model with one signature.")
        signature = signature_list[next(iter(signature_list))]
        outputs = signature["outputs"]
        return {
            "kind": "signature",
            "count": outputs["output_0"],
            "scores": outputs["output_1"],
            "class_ids": outputs["output_2"],
            "boxes": outputs["output_3"],
        }
    if _output_tensor(interpreter, 3).size == 1:
        return {"kind": "order", "boxes": 0, "class_ids": 1, "scores": 2, "count": 3}
    return {"kind": "order", "scores": 0, "boxes": 1, "count": 2, "class_ids": 3}


def _read_outputs(interpreter, layout):
    """Return ``(boxes, class_ids, scores, count)`` for the resolved *layout*."""
    if layout["kind"] == "signature":
        get = interpreter.tensor
        return (
            get(layout["boxes"])()[0],
            get(layout["class_ids"])()[0],
            get(layout["scores"])()[0],
            int(get(layout["count"])()[0]),
        )
    return (
        _output_tensor(interpreter, layout["boxes"])[0],
        _output_tensor(interpreter, layout["class_ids"])[0],
        _output_tensor(interpreter, layout["scores"])[0],
        int(_output_tensor(interpreter, layout["count"])[0]),
    )


def _get_objects(interpreter, score_threshold, image_scale, layout=None):
    """Return [{'id', 'score', 'bbox': (xmin,ymin,xmax,ymax)}, ...].

    *layout* is the cached result of ``resolve_output_layout``; it is derived on
    demand when omitted so this stays usable standalone (and testable).
    """
    if layout is None:
        layout = resolve_output_layout(interpreter)
    boxes, class_ids, scores, count = _read_outputs(interpreter, layout)

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


def _clamp_bbox(bbox, width, height):
    """Clip *bbox* to the frame and return None if nothing is left.

    Model output needs clamping for a specific structural reason:
    ``_set_resized_input`` preserves aspect ratio by scaling to fit and
    **zero-padding the right/bottom**. A box the model predicts partly inside
    that padding region maps back to coordinates beyond ``width``/``height``
    (and rounding can push an edge slightly negative).

    Left unclamped that matters downstream, because BehaviorMonitor's paw
    point is the bbox *bottom-centre*: an out-of-frame bottom edge places the
    paw point outside the fence polygon and the event is silently missed. The
    out-of-range box was also written to SQLite and published over MQTT, where
    any consumer scaling by ``frame_w``/``frame_h`` would draw it off-image.
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width))
    y0 = max(0, min(int(y0), height))
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


#: Matches the README's documented default and DogDetector's own default, so
#: a config that omits score_threshold behaves as documented instead of raising.
DEFAULT_SCORE_THRESHOLD = 0.4


def resolve_score_threshold(cfg, name="camera"):
    """Return a camera config's detection threshold, validated.

    Lives here rather than in dogwatch.py so CameraPipeline can share it
    without a circular import (dogwatch imports camera_pipeline).

    A threshold outside (0, 1] breaks detection in a way that looks exactly
    like "the model never sees my dog": ``score_threshold: 55`` (meaning 55%)
    suppresses every detection forever, silently. Warn and fall back to the
    documented default rather than run in that state.
    """
    raw = cfg.get("score_threshold", DEFAULT_SCORE_THRESHOLD)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"[{name}] WARNING: score_threshold={raw!r} is not a number — "
              f"using default {DEFAULT_SCORE_THRESHOLD}")
        return DEFAULT_SCORE_THRESHOLD
    if not 0.0 < value <= 1.0:
        print(f"[{name}] WARNING: score_threshold={value} is outside (0, 1] — "
              f"confidences are fractions, not percentages. Using default "
              f"{DEFAULT_SCORE_THRESHOLD}")
        return DEFAULT_SCORE_THRESHOLD
    return value


class DogDetector:
    """Shared Edge TPU interpreter, filtered to the 'dog' class.

    Only one process can bind the TPU, so a multi-camera deployment shares one
    instance of this class. Note that ``score_threshold`` is a **pure
    post-inference filter** on the output tensors — it costs nothing to vary per
    call, and ``detect()`` accepts an override for exactly that reason. See
    ``detect()`` for why that matters.
    """

    def __init__(self, model_path, labels_path, score_threshold=DEFAULT_SCORE_THRESHOLD,
                 target_label="dog"):
        self.interp = _make_interpreter(model_path)
        self.interp.allocate_tensors()
        # Resolve the output tensor layout once — it cannot change for a loaded
        # model, and deriving it per inference cost a private-API call plus a
        # re-read of every output tensor's details on the critical path.
        self._output_layout = resolve_output_layout(self.interp)
        # Acts as the default/floor; callers may pass a stricter value per call.
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

    def detect(self, frame, score_threshold=None):
        """Return [{'bbox': (x0,y0,x1,y1) in pixels, 'score': float}, ...].

        *score_threshold* overrides the instance default for this call. This
        exists so a **shared** detector can still honour **per-camera**
        thresholds: one process binds the TPU, so all cameras share this
        interpreter, but confidence filtering happens in pure Python over the
        output tensors and therefore does not have to be shared.

        Without this, ``dogwatch.py`` built the shared detector from the first
        config alone and every subsequent camera silently ran on camera #1's
        threshold — while the README documented ``score_threshold`` as
        per-camera and specifically advised raising it per camera to suppress
        false positives.

        Boxes are clamped to the frame; see _clamp_bbox for why that matters.
        """
        threshold = self.score_threshold if score_threshold is None else score_threshold
        h, w = frame.shape[:2]
        scale = _set_resized_input(
            self.interp, (w, h), lambda size: cv2.resize(frame, size))
        self.interp.invoke()
        out = []
        for obj in _get_objects(self.interp, threshold, scale,
                                layout=self._output_layout):
            if obj["id"] not in self.target_ids:
                continue
            bbox = _clamp_bbox(obj["bbox"], w, h)
            if bbox is None:
                # Entirely outside the frame (pure padding artefact).
                continue
            out.append({"bbox": bbox, "score": obj["score"]})
        return out

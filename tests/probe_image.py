"""probe_image.py — ad-hoc CLI to run the real DogDetector against one or
more image files on the real Coral TPU, showing ALL detections (not just
'dog', and at a low score floor) rather than the filtered/thresholded output
DogDetector.detect() normally returns.

Useful for quickly checking "is the detector still working at all" against
a hand-picked image without waiting for a real dog, or for inspecting why a
specific image produced no detections (e.g. to see if *something* was
detected, just not confidently, or not as 'dog').

Requires the real Coral TPU device and model files, so — like
tests/hardware_smoke_test.py — this only runs on the deployment host with
the main dogwatch container stopped first (only one process can hold the
Edge TPU delegate at a time):

    docker stop dogwatch
    docker run --rm --device /dev/apex_0:/dev/apex_0 \
        -v "$(pwd)/models:/app/models:ro" \
        -v "$(pwd)/samples:/app/samples:ro" \
        -v "$(pwd)/tests/probe_image.py:/app/probe_image.py" \
        dogtracker-dogwatch python /app/probe_image.py samples/some_image.jpg
    docker start dogwatch
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import DogDetector, _get_objects, _set_resized_input

MODEL_PATH = os.environ.get(
    "DOGWATCH_MODEL_PATH", "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
)
LABELS_PATH = os.environ.get("DOGWATCH_LABELS_PATH", "models/coco_labels.txt")
SCORE_FLOOR = float(os.environ.get("DOGWATCH_PROBE_SCORE_FLOOR", "0.05"))


def main():
    if len(sys.argv) < 2:
        print("Usage: probe_image.py <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    d = DogDetector(MODEL_PATH, LABELS_PATH, score_threshold=SCORE_FLOOR)

    for path in sys.argv[1:]:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: FAILED TO DECODE")
            continue
        h, w = frame.shape[:2]
        scale = _set_resized_input(d.interp, (w, h), lambda size: cv2.resize(frame, size))
        d.interp.invoke()
        objs = _get_objects(d.interp, SCORE_FLOOR, scale)
        print(f"{path}: shape={frame.shape}")
        if not objs:
            print(f"  NO detections at all (any class) above {SCORE_FLOOR}")
        for o in objs:
            label = d.labels.get(o["id"], "?")
            print(f"  id={o['id']} label={label} score={o['score']:.2f} bbox={o['bbox']}")


if __name__ == "__main__":
    main()

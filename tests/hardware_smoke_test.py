"""hardware_smoke_test.py — on-hardware end-to-end detection smoke test.

Unlike tests/test_detector.py (which uses fake interpreters and needs no
hardware), this script exercises the *real* DogDetector against the real
Coral Edge TPU and real model file, using known-good sample images (real
past detections, saved in samples/) as ground truth.

This is NOT part of the pytest suite — it requires the actual Coral TPU
device and can only run where /dev/apex_0 exists and nothing else currently
holds the delegate (i.e. the main dogwatch container must be stopped first).
It exists specifically to answer "did a dependency/model/runtime change
silently break detection accuracy on real hardware" without waiting for a
real dog to wander by.

See samples/README.md for the ground truth on each sample image and why
some are *expected* to score weakly or produce zero detections at the full
frame (small/distant dogs in wide uncropped frames — see that doc for the
full explanation and an A/B test methodology against a prior dependency
stack, which is how the ai-edge-litert migration was cleared of suspicion
for this exact symptom).

Usage (on the server, with the main container stopped):
    docker run --rm --device /dev/apex_0:/dev/apex_0 \
        -v "$(pwd)/models:/app/models:ro" \
        -v "$(pwd)/samples:/app/samples:ro" \
        -v "$(pwd)/tests/hardware_smoke_test.py:/app/hardware_smoke_test.py" \
        dogtracker-dogwatch python /app/hardware_smoke_test.py

Exit code is non-zero only if a sample marked "hard" (expected to detect
reliably at the production threshold) fails. Samples marked "weak" (known,
pre-existing small-object cases — see samples/README.md) are reported but
don't fail the run, since they're expected to be threshold-sensitive.
"""
import glob
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import DogDetector, _get_objects, _set_resized_input

MODEL_PATH = os.environ.get(
    "DOGWATCH_MODEL_PATH", "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
)
LABELS_PATH = os.environ.get("DOGWATCH_LABELS_PATH", "models/coco_labels.txt")
SAMPLES_DIR = os.environ.get("DOGWATCH_SAMPLES_DIR", "samples")
SCORE_THRESHOLD = float(os.environ.get("DOGWATCH_SAMPLE_SCORE_THRESHOLD", "0.4"))

# Samples known (via the A/B test documented in samples/README.md) to be
# weak/zero at the full frame regardless of dependency stack — small or
# distant dogs in an uncropped wide-angle frame, a pre-existing model
# limitation rather than a bug. Failing to detect these does NOT fail the
# script; a *regression* on a sample NOT in this set is what matters.
#
# As of this writing, all 5 available real-world samples fall into this
# bucket (every one is a small/distant dog in a full uncropped frame) — see
# samples/README.md. That means plain pass/fail on detection presence isn't
# a useful regression signal for these specific images; BASELINE_SCORES
# below (measured on the current ai-edge-litert stack) is the more
# meaningful check — a large drop in best_score for a sample versus its
# baseline is a stronger regression signal than "still zero detections",
# since these were already at/near zero. Update BASELINE_SCORES whenever
# samples/ changes or a new baseline is intentionally established.
KNOWN_WEAK_SAMPLES = {
    "rear-east-yes-1.jpg",
    "rear-east-yes-2.jpg",
    "rear-east-yes-3.jpg",
    "rear-east-yes-4.jpg",
    "fence-camera-yes-1.jpg",
}

# best_score measured per sample at the full frame (score_threshold=0.05),
# from the A/B test in samples/README.md. A regression check compares
# against these with some tolerance rather than expecting an exact match
# (TPU quantization/model export differences can cause tiny score jitter).
BASELINE_SCORES = {
    "rear-east-yes-1.jpg": 0.20,
    "rear-east-yes-2.jpg": 0.00,
    "rear-east-yes-3.jpg": 0.00,
    "rear-east-yes-4.jpg": 0.00,
    "fence-camera-yes-1.jpg": 0.00,
}
SCORE_REGRESSION_TOLERANCE = 0.10


def find_samples(samples_dir):
    """Return sorted list of (path, expect_dog) pairs from *samples_dir*.

    Naming convention: files containing 'yes' in the name are expected to
    contain a dog; files containing 'no' are expected not to. Anything else
    is treated as expect_dog=True (the common case: real alert photos).
    """
    paths = sorted(
        glob.glob(os.path.join(samples_dir, "*.jpg"))
        + glob.glob(os.path.join(samples_dir, "*.jpeg"))
        + glob.glob(os.path.join(samples_dir, "*.png"))
    )
    out = []
    for p in paths:
        name = os.path.basename(p).lower()
        if "no" in name and "yes" not in name:
            out.append((p, False))
        else:
            out.append((p, True))
    return out


def main():
    print(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    detector = DogDetector(MODEL_PATH, LABELS_PATH, score_threshold=SCORE_THRESHOLD)
    print(f"  Detector ready in {time.time() - t0:.3f}s (Edge TPU delegate loaded)")

    samples = find_samples(SAMPLES_DIR)
    if not samples:
        print(f"No sample images found in {SAMPLES_DIR}/ — nothing to test.")
        sys.exit(1)

    print(f"\nRunning {len(samples)} sample(s) at score_threshold={SCORE_THRESHOLD}\n")

    hard_passed = 0
    hard_failed = 0
    weak_regressed = 0
    weak_total = 0
    for path, expect_dog in samples:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[SKIP] {path}: could not decode image")
            continue

        basename = os.path.basename(path)
        is_known_weak = basename in KNOWN_WEAK_SAMPLES

        # Run detection twice: once at the real production threshold (for
        # the pass/fail semantics below), once effectively unthresholded
        # (0.0) purely to get the true best_score for regression comparison
        # against BASELINE_SCORES — a weak sample can still regress further
        # (e.g. 0.20 -> 0.05) without crossing back over 0 detections.
        t0 = time.time()
        results = detector.detect(frame)
        elapsed_ms = (time.time() - t0) * 1000

        # Re-run the raw pipeline at an effectively-unthresholded score cutoff
        # to get the true best dog score for baseline comparison, independent
        # of the production score_threshold used for pass/fail above.
        h, w = frame.shape[:2]
        scale = _set_resized_input(detector.interp, (w, h), lambda size: cv2.resize(frame, size))
        detector.interp.invoke()
        raw_objs = _get_objects(detector.interp, 0.0, scale)
        raw_dog_scores = [o["score"] for o in raw_objs if o["id"] in detector.target_ids]

        found_dog = len(results) > 0
        best_score = max(raw_dog_scores, default=0.0)

        if is_known_weak:
            weak_total += 1
            baseline = BASELINE_SCORES.get(basename)
            if baseline is not None and best_score < baseline - SCORE_REGRESSION_TOLERANCE:
                status = "WEAK-REGRESSED"
                weak_regressed += 1
            else:
                status = "WEAK-OK(baseline)"
        else:
            ok = found_dog == expect_dog
            status = "PASS" if ok else "FAIL"
            if ok:
                hard_passed += 1
            else:
                hard_failed += 1

        baseline_note = f" baseline={BASELINE_SCORES.get(basename, 'n/a')}" if is_known_weak else ""
        print(
            f"[{status}] {basename}  "
            f"expected_dog={expect_dog} found_dog={found_dog} "
            f"detections={len(results)} best_score={best_score:.2f}{baseline_note} "
            f"inference={elapsed_ms:.1f}ms"
        )
        for r in results:
            print(f"         bbox={r['bbox']} score={r['score']:.2f}")

    print(f"\nHard cases: {hard_passed} passed, {hard_failed} failed")
    print(f"Known-weak cases: {weak_total - weak_regressed}/{weak_total} within baseline tolerance, "
          f"{weak_regressed} regressed")
    sys.exit(1 if (hard_failed or weak_regressed) else 0)


if __name__ == "__main__":
    main()

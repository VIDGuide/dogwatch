"""
dogwatch.py — multi-camera main loop / entrypoint.

Loads one config per camera, builds a shared Coral interpreter (only one
process can bind the TPU) plus one CameraPipeline per camera, and drives
the detect/track/publish loop at the slowest configured camera's target fps.

The actual per-camera work lives in camera_pipeline.py (grab/crop/detect/
track/publish), frame_grabber.py (background RTSP reads), and
snapshot_quality.py (grey/corrupt-frame heuristics) — this file just wires
them together.
"""
import json
import os
import sys
import time
import traceback

from camera_pipeline import CameraPipeline
from detector import DogDetector
from redact import redact


def load_config(path):
    with open(path) as f:
        return json.load(f)


def main():
    # Config files: either passed as CLI args, or default to config.json plus
    # any config-*.json files alongside it.
    if len(sys.argv) > 1:
        config_paths = sys.argv[1:]
    else:
        config_paths = ["config.json"]
        base = os.path.dirname(os.path.abspath("config.json")) or "."
        extras = sorted(
            os.path.join(base, f) for f in os.listdir(base)
            if f.startswith("config-") and f.endswith(".json")
        )
        config_paths.extend(extras)

    cfgs = [load_config(p) for p in config_paths]
    print(f"Loaded {len(cfgs)} camera config(s): {', '.join(config_paths)}")

    # Shared model / Coral interpreter (only one can bind the TPU).
    shared = DogDetector(
        cfgs[0]["model_path"], cfgs[0]["labels_path"],
        cfgs[0]["score_threshold"],
    )

    # Build a pipeline per camera.
    pipelines = []
    for i, cfg in enumerate(cfgs):
        name = os.path.splitext(os.path.basename(config_paths[i]))[0]
        name = name.replace("config-", "").replace("config", "camera")
        pipelines.append(CameraPipeline(cfg, name))

    # Drive the loop at the SLOWEST camera's target fps (min), so no camera is
    # sampled faster than it was configured for.
    target_fps = min(cfg.get("target_fps", 5) for cfg in cfgs)
    if target_fps <= 0:
        raise ValueError(
            f"target_fps must be > 0 (got {target_fps}); "
            f"check target_fps in your camera configs"
        )
    interval = 1.0 / target_fps

    # Warm up frame grabbers before entering the loop.
    time.sleep(2)

    # Per-camera error isolation. Without this, ANY exception from ANY
    # camera's tick() killed the process: an sqlite "database is locked" from
    # a concurrent reader, a cv2.error from imwrite on a full disk, a TPU
    # invoke() failure. With restart: unless-stopped that meant one camera's
    # transient fault restarted detection for every camera, re-paying the
    # startup wait each time. Errors are logged with a per-camera backoff so a
    # persistently failing camera can't flood the log either.
    err_counts = {}
    last_err_log = {}

    while True:
        t0 = time.time()
        for pipe in pipelines:
            try:
                pipe.tick(shared, t0)
            except Exception as exc:
                n = err_counts.get(pipe.name, 0) + 1
                err_counts[pipe.name] = n
                # Log the first error immediately, then at most once a minute.
                if n == 1 or t0 - last_err_log.get(pipe.name, 0.0) > 60:
                    last_err_log[pipe.name] = t0
                    print(f"[{pipe.name}] tick failed ({n} total): "
                          f"{type(exc).__name__}: {redact(exc)}", flush=True)
                    traceback.print_exc()
        dt = time.time() - t0
        if dt < interval:
            time.sleep(interval - dt)


if __name__ == "__main__":
    main()

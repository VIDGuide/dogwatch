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

from camera_pipeline import CameraPipeline
from detector import DogDetector


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

    # Sync all to the fastest camera's target fps.
    target_fps = min(cfg.get("target_fps", 5) for cfg in cfgs)
    interval = 1.0 / target_fps

    # Warm up frame grabbers before entering the loop.
    time.sleep(2)

    while True:
        t0 = time.time()
        for pipe in pipelines:
            pipe.tick(shared, t0)
        dt = time.time() - t0
        if dt < interval:
            time.sleep(interval - dt)


if __name__ == "__main__":
    main()

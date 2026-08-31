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
from detector import DogDetector, resolve_score_threshold
from heartbeat import Heartbeat
from redact import redact


def load_config(path):
    with open(path) as f:
        return json.load(f)


def camera_name_for(config_path):
    """Derive a camera name from its config filename.

    ``config-rear-east.json`` -> ``rear-east``; ``config.json`` -> ``camera``.
    """
    name = os.path.splitext(os.path.basename(config_path))[0]
    return name.replace("config-", "").replace("config", "camera")


def warn_on_shared_key_conflicts(cfgs, names):
    """Warn when configs disagree on a key that can only have one value.

    The Edge TPU delegate can only be held by one process, so there is a single
    interpreter built from the first config. Any other config's ``model_path``
    or ``labels_path`` is therefore ignored — previously in complete silence.
    """
    for key in ("model_path", "labels_path"):
        first = cfgs[0].get(key)
        for cfg, name in zip(cfgs[1:], names[1:]):
            value = cfg.get(key)
            if value is not None and value != first:
                print(f"[{name}] WARNING: {key}={value!r} is IGNORED — only one "
                      f"Edge TPU interpreter exists, built from "
                      f"{names[0]}'s {key}={first!r}. All cameras share one "
                      f"model. (score_threshold, by contrast, IS per-camera.)")


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
    if not cfgs:
        raise ValueError("No camera configs found — expected config.json and/or "
                         "config-<name>.json in the working directory")
    print(f"Loaded {len(cfgs)} camera config(s): {', '.join(config_paths)}")

    names = [camera_name_for(p) for p in config_paths]

    # Warn about keys that genuinely CANNOT be per-camera. Only one process can
    # bind the TPU, so there is exactly one interpreter and therefore one model.
    # Silently ignoring a second camera's model_path is the kind of thing that
    # wastes an afternoon, so say so out loud.
    warn_on_shared_key_conflicts(cfgs, names)

    # Per-camera detection thresholds.
    #
    # score_threshold, unlike model_path, does NOT have to be shared: it's a
    # pure post-inference filter over the output tensors. It used to be shared
    # by accident — the detector was built from cfgs[0] alone, so every camera
    # after the first ran on camera #1's threshold. That made the README's
    # advice to raise score_threshold per camera (to suppress that camera's
    # false positives) a no-op on every camera but the first, and silently
    # ignored the 0.55 in config-rear-east.example.json.
    #
    # The shared detector is built at the LOWEST configured threshold, so no
    # camera is starved of detections it should have seen, and each pipeline
    # then applies its own threshold to the results.
    thresholds = [resolve_score_threshold(cfg, name)
                  for cfg, name in zip(cfgs, names)]
    floor = min(thresholds)

    shared = DogDetector(
        cfgs[0]["model_path"], cfgs[0]["labels_path"],
        score_threshold=floor,
    )
    for name, thr in zip(names, thresholds):
        extra = " (also the shared inference floor)" if thr == floor else ""
        print(f"[{name}] score_threshold={thr}{extra}")

    # Build a pipeline per camera.
    pipelines = []
    for cfg, name in zip(cfgs, names):
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

    # Liveness signal for the container HEALTHCHECK and the host watchdog. The
    # process staying alive is not evidence that it is still watching anything —
    # see heartbeat.py.
    beat = Heartbeat(interval=float(os.environ.get("DOGWATCH_HEARTBEAT_INTERVAL", 5)))

    try:
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
            if beat.should_write(t0):
                beat.write(t0, {p.name: p.health(t0, tick_seconds=dt)
                                for p in pipelines})
            if dt < interval:
                time.sleep(interval - dt)
    except KeyboardInterrupt:
        print("Interrupted — shutting down", flush=True)
    finally:
        # Drain each camera's queued writes so an event that fired moments
        # before shutdown still reaches disk and the event DB, and release the
        # capture handles. Writes are asynchronous now, so without this a clean
        # stop could silently lose the last event or two.
        for pipe in pipelines:
            try:
                pipe.close()
            except Exception as exc:
                print(f"[{pipe.name}] error during shutdown: {redact(exc)}",
                      flush=True)


if __name__ == "__main__":
    main()

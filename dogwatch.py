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


def check_topic_collisions(cfgs, names, env=None):
    """Fail fast when two cameras would publish to the same MQTT base topic.

    Every camera needs its own base topic. The HA discovery payloads are keyed
    per camera (``dev_id = dogwatch_<name>``), so two cameras sharing a base
    topic register two *distinct* pairs of entities that both subscribe to the
    *same* state topic — camera A's dog turns on camera B's sensor, and their
    retained snapshots overwrite each other. Nothing about that is recoverable
    or visible from the Home Assistant side; it just looks like both cameras
    see everything.

    There are two routes into it, and neither used to be checked:

    1. ``MQTT_TOPIC`` in the environment. ``CameraPipeline`` reads it as
       ``os.environ.get("MQTT_TOPIC", cfg["mqtt_base_topic"])``, so it is a
       *single global value overriding a per-camera setting*. Setting it with
       more than one camera configured silently collapses the whole fleet onto
       one topic. (``MQTT_HOST``/``MQTT_PORT`` are genuinely global — one
       broker — so they are deliberately not checked here.)
    2. Two config files that simply both say the same ``mqtt_base_topic``.

    Raising rather than warning: this is a misconfiguration whose only symptom
    is wrong data, and it is cheaper to refuse at startup than to debug "why
    does the front camera trigger when the dog is out the back".
    """
    env = os.environ if env is None else env
    if len(cfgs) < 2:
        return

    override = env.get("MQTT_TOPIC")
    if override:
        raise ValueError(
            f"MQTT_TOPIC={override!r} is set in the environment, but "
            f"{len(cfgs)} cameras are configured ({', '.join(names)}). "
            f"MQTT_TOPIC is a single global value and would override every "
            f"camera's per-camera mqtt_base_topic, collapsing the whole fleet "
            f"onto one topic. Unset MQTT_TOPIC and set mqtt_base_topic in each "
            f"config instead (e.g. 'dogwatch' and 'dogwatch/rear-east')."
        )

    seen = {}
    for cfg, name in zip(cfgs, names):
        topic = cfg.get("mqtt_base_topic")
        if topic is None:
            continue
        if topic in seen:
            raise ValueError(
                f"Cameras {seen[topic]!r} and {name!r} both use "
                f"mqtt_base_topic={topic!r}. Each camera needs its own base "
                f"topic, or their Home Assistant entities and snapshots will "
                f"overwrite each other."
            )
        seen[topic] = name


def build_pipelines(cfgs, names, factory=None, log=print):
    """Construct one pipeline per camera, isolating per-camera startup failures.

    Returns ``(pipelines, failures)`` where *failures* is a list of
    ``(name, exception)`` for the cameras that could not be brought up.

    Why this is not a plain list comprehension: ``CameraPipeline.__init__``
    raises when a camera produces no frame within ``startup_timeout_seconds``
    (default 60). That exception used to propagate straight out of ``main()``,
    so **one** unreachable camera took down detection for **every** camera:
    the process exited non-zero, ``restart: unless-stopped`` restarted it, it
    waited out the timeout again and died again. A healthy camera sharing the
    container never ticked once, and because no heartbeat was ever written the
    healthcheck reported unhealthy and ``dogwatch-watchdog.sh`` restarted the
    container too — the two recovery mechanisms reinforced the loop instead of
    breaking it.

    That was also inconsistent with the policy this project already decided on
    for the *running* case. ``heartbeat.evaluate`` deliberately keeps the
    container healthy when a single camera among several goes stale, on the
    grounds that the others still work, a restart would disrupt them too, and a
    camera unplugged for a day should not cause a restart loop. Exactly the same
    reasoning applies at startup, so the same policy is applied here: bring up
    what can be brought up, and let a dead camera show as ``unavailable`` in
    Home Assistant rather than taking the fleet with it.

    *factory* defaults to ``CameraPipeline`` and exists so this is testable
    without a camera or a TPU. It is resolved at call time rather than captured
    as the parameter's default, so patching the module attribute works too.
    """
    factory = CameraPipeline if factory is None else factory
    pipelines = []
    failures = []
    for cfg, name in zip(cfgs, names):
        try:
            pipelines.append(factory(cfg, name))
        except Exception as exc:
            failures.append((name, exc))
            # redact(): the message from a startup timeout embeds rtsp_url, and
            # an unexpected exception type may embed it too. Format the
            # traceback rather than using traceback.print_exc() so it goes
            # through redaction as well — see redact.py's module docstring.
            log(f"[{name}] FAILED TO START: {type(exc).__name__}: "
                f"{redact(exc)} — continuing without this camera. The other "
                f"cameras will run normally; this one will not be retried "
                f"until the container restarts.")
            log(redact(traceback.format_exc()))
    return pipelines, failures


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

    # Two cameras publishing to one base topic is silently destructive, and
    # MQTT_TOPIC in the environment is a global override for a per-camera key.
    check_topic_collisions(cfgs, names)

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

    # Build a pipeline per camera. A camera that cannot produce a frame is
    # skipped rather than aborting the whole fleet — see build_pipelines.
    pipelines, failures = build_pipelines(cfgs, names)
    if not pipelines:
        # Nothing came up at all, so there is genuinely nothing to do and
        # exiting non-zero (letting the restart policy retry) is correct. This
        # is the one case where the old fail-loudly behaviour still applies.
        raise RuntimeError(
            "No camera pipelines could be started ("
            + "; ".join(f"{name}: {redact(exc)}" for name, exc in failures)
            + ") — check the RTSP URLs/credentials and that the cameras are "
              "reachable"
        )
    if failures:
        started = ", ".join(p.name for p in pipelines)
        dead = ", ".join(name for name, _ in failures)
        print(f"Started {len(pipelines)}/{len(cfgs)} camera(s): {started}. "
              f"DEGRADED — failed to start: {dead}", flush=True)

    # Drive the loop at the SLOWEST camera's target fps (min), so no camera is
    # sampled faster than it was configured for. Computed over the cameras that
    # actually STARTED, not every config: a camera that failed to come up has no
    # sampling requirement, and letting its target_fps hold the loop down would
    # mean an offline slow camera silently throttled a healthy fast one.
    cfg_by_name = dict(zip(names, cfgs))
    target_fps = min(cfg_by_name[p.name].get("target_fps", 5) for p in pipelines)
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
                        # format_exc() + redact(), not print_exc(): the
                        # traceback's final line is the exception message, and a
                        # cv2/requests error raised from the frame grabber or the
                        # snapshot fetch embeds the credential-bearing camera
                        # URL. print_exc() writes straight to stderr with no
                        # chance to redact, so the message above was masked while
                        # the identical string leaked one line below it. See
                        # redact.py's module docstring for the rule.
                        print(redact(traceback.format_exc()), flush=True)

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

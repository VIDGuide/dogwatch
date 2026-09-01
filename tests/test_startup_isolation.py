"""Unit tests for per-camera startup isolation in dogwatch.build_pipelines.

The bug these lock down: ``main()`` built pipelines in a bare loop, so the
``RuntimeError`` that ``CameraPipeline.__init__`` raises when a camera produces
no frame within ``startup_timeout_seconds`` propagated straight out of
``main()``. One unreachable camera therefore took down detection for *every*
camera: the process exited non-zero, ``restart: unless-stopped`` restarted it,
it waited out the timeout again and died again. A healthy camera in the same
container never ticked once, and since no heartbeat was ever written the
healthcheck also reported unhealthy and the watchdog restarted the container —
both recovery mechanisms reinforcing the loop instead of breaking it.

That contradicted the policy this project already chose for the running case:
``heartbeat.evaluate`` deliberately keeps the container healthy when one camera
among several goes stale, because the others still work and "a camera that is
genuinely unplugged for a day should not cause a restart loop". These tests
assert the same policy now holds at startup.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dogwatch


class FakePipeline:
    """Stand-in for CameraPipeline: records construction, never touches I/O."""

    def __init__(self, cfg, name):
        self.cfg = cfg
        self.name = name
        self.closed = False
        self.ticks = 0

    def tick(self, detector, t0):
        self.ticks += 1

    def health(self, now=None, tick_seconds=None):
        return {"stale": False}

    def close(self):
        self.closed = True


def factory_failing(*failing_names, exc=None):
    """Return a CameraPipeline-shaped factory that raises for *failing_names*."""
    def factory(cfg, name):
        if name in failing_names:
            raise (exc or RuntimeError(
                f"[{name}] No frame received from "
                f"rtsp://user:hunter2@cam.lan:554/s after 60s"))
        return FakePipeline(cfg, name)
    return factory


CFG = {"target_fps": 5}


# --------------------------------------------------------------------------
# build_pipelines
# --------------------------------------------------------------------------

class TestBuildPipelines:
    def test_all_healthy_builds_every_camera(self):
        pipes, failures = dogwatch.build_pipelines(
            [CFG, CFG], ["camera", "rear-east"],
            factory=FakePipeline, log=lambda *a: None)
        assert [p.name for p in pipes] == ["camera", "rear-east"]
        assert failures == []

    def test_one_failing_camera_does_not_stop_the_others(self):
        # The core regression: rear-east is offline, camera is fine.
        pipes, failures = dogwatch.build_pipelines(
            [CFG, CFG], ["camera", "rear-east"],
            factory=factory_failing("rear-east"), log=lambda *a: None)
        assert [p.name for p in pipes] == ["camera"]
        assert [name for name, _ in failures] == ["rear-east"]

    def test_failing_first_camera_does_not_stop_later_ones(self):
        # Order must not matter — the failure used to abort the whole loop.
        pipes, failures = dogwatch.build_pipelines(
            [CFG, CFG], ["camera", "rear-east"],
            factory=factory_failing("camera"), log=lambda *a: None)
        assert [p.name for p in pipes] == ["rear-east"]
        assert [name for name, _ in failures] == ["camera"]

    def test_all_failing_returns_no_pipelines_and_all_failures(self):
        pipes, failures = dogwatch.build_pipelines(
            [CFG, CFG], ["camera", "rear-east"],
            factory=factory_failing("camera", "rear-east"),
            log=lambda *a: None)
        assert pipes == []
        assert [name for name, _ in failures] == ["camera", "rear-east"]

    def test_failure_is_logged_with_the_camera_name(self):
        lines = []
        dogwatch.build_pipelines([CFG], ["rear-east"],
                                 factory=factory_failing("rear-east"),
                                 log=lines.append)
        joined = "\n".join(lines)
        assert "rear-east" in joined
        assert "FAILED TO START" in joined

    def test_failure_log_redacts_credentials(self):
        # The startup-timeout message embeds rtsp_url, which carries user:pass,
        # and this goes to docker logs / any log shipper. Both the message and
        # the formatted traceback must be redacted.
        lines = []
        dogwatch.build_pipelines([CFG], ["rear-east"],
                                 factory=factory_failing("rear-east"),
                                 log=lines.append)
        joined = "\n".join(lines)
        assert "hunter2" not in joined
        assert "***:***@cam.lan" in joined

    def test_unexpected_exception_type_is_also_isolated(self):
        # Not just the startup timeout: a missing config key (KeyError from
        # BehaviorMonitor's cfg["fence_zone_norm"]) or a disk error must not
        # take the fleet down either.
        pipes, failures = dogwatch.build_pipelines(
            [CFG, CFG], ["camera", "rear-east"],
            factory=factory_failing("rear-east", exc=KeyError("fence_zone_norm")),
            log=lambda *a: None)
        assert [p.name for p in pipes] == ["camera"]
        assert isinstance(failures[0][1], KeyError)

    def test_keyboardinterrupt_is_not_swallowed(self):
        # Ctrl-C during the startup wait must still stop the process rather than
        # being logged as "this camera failed" and pressing on.
        with pytest.raises(KeyboardInterrupt):
            dogwatch.build_pipelines(
                [CFG], ["camera"],
                factory=factory_failing("camera", exc=KeyboardInterrupt()),
                log=lambda *a: None)

    def test_default_factory_is_camera_pipeline(self):
        # Guards against the injectable factory drifting away from production:
        # omitting it must use the real CameraPipeline, resolved at call time.
        import camera_pipeline
        used = []

        class Sentinel:
            def __init__(self, cfg, name):
                used.append(name)

        original = dogwatch.CameraPipeline
        try:
            dogwatch.CameraPipeline = Sentinel
            dogwatch.build_pipelines([CFG], ["camera"], log=lambda *a: None)
        finally:
            dogwatch.CameraPipeline = original
        assert used == ["camera"]
        assert dogwatch.CameraPipeline is camera_pipeline.CameraPipeline


# --------------------------------------------------------------------------
# main() wiring: degraded start vs total failure
# --------------------------------------------------------------------------

def _patch_main_deps(monkeypatch, tmp_path, cfgs, factory):
    """Point main() at temp configs and stub out the TPU + pipeline factory."""
    for fname, cfg in cfgs.items():
        import json
        (tmp_path / fname).write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dogwatch.py"])

    class FakeDetector:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(dogwatch, "DogDetector", FakeDetector)
    monkeypatch.setattr(dogwatch, "CameraPipeline", factory)
    # Enter the loop once, then bail out so main() returns.
    monkeypatch.setattr(dogwatch.time, "sleep", lambda *a: None)

    def one_shot(*a, **k):
        raise KeyboardInterrupt

    return one_shot


BASE_CFG = {
    "model_path": "m.tflite", "labels_path": "l.txt",
    "score_threshold": 0.5, "target_fps": 5,
}


# --------------------------------------------------------------------------
# CameraPipeline cleanup when construction fails
# --------------------------------------------------------------------------

class TestFailedConstructionReleasesResources:
    """A camera that fails to start must not leak its frame-grabber thread.

    This only started mattering with the fix above: the process now keeps
    running instead of exiting, so an orphan FrameGrabber would sit there
    reconnecting to a dead RTSP URL forever with nobody reading from it.
    """

    def test_grabber_is_stopped_when_build_raises(self, monkeypatch):
        import camera_pipeline

        stopped = []

        class FakeGrabber:
            def __init__(self, *a, **k):
                pass

            def stop(self):
                stopped.append(True)

        monkeypatch.setattr(camera_pipeline, "FrameGrabber", FakeGrabber)

        # No "fence_zone_norm" etc., so _build raises after grab is assigned.
        with pytest.raises(Exception):
            camera_pipeline.CameraPipeline({"rtsp_url": "rtsp://x/y"},
                                           "rear-east")

        assert stopped, "FrameGrabber.stop() was not called on failed construction"

    def test_close_is_safe_before_anything_is_built(self, monkeypatch):
        import camera_pipeline

        class Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("cannot open device")

        monkeypatch.setattr(camera_pipeline, "FrameGrabber", Boom)

        # Fails on the very first resource, so close() runs with every
        # attribute still None. It must not raise a second exception and mask
        # the real one.
        with pytest.raises(RuntimeError, match="cannot open device"):
            camera_pipeline.CameraPipeline({"rtsp_url": "rtsp://x/y"}, "camera")

    def test_close_is_idempotent(self, monkeypatch):
        import camera_pipeline

        pipe = camera_pipeline.CameraPipeline.__new__(
            camera_pipeline.CameraPipeline)
        pipe.name = "camera"
        pipe.grab = pipe.pub = pipe.event_store = pipe.writer = None
        pipe.close()
        pipe.close()   # must not raise


class TestMainDegradedStartup:
    def test_main_survives_one_offline_camera(self, monkeypatch, tmp_path,
                                              capsys):
        one_shot = _patch_main_deps(
            monkeypatch, tmp_path,
            {"config.json": BASE_CFG, "config-rear-east.json": BASE_CFG},
            factory_failing("rear-east"))
        monkeypatch.setattr(dogwatch.Heartbeat, "should_write", one_shot)

        # Must NOT raise: the healthy camera has to keep running.
        dogwatch.main()

        out = capsys.readouterr().out
        assert "DEGRADED" in out
        assert "rear-east" in out

    def test_main_raises_only_when_no_camera_starts(self, monkeypatch, tmp_path):
        _patch_main_deps(
            monkeypatch, tmp_path,
            {"config.json": BASE_CFG, "config-rear-east.json": BASE_CFG},
            factory_failing("camera", "rear-east"))

        with pytest.raises(RuntimeError, match="No camera pipelines"):
            dogwatch.main()

    def test_total_failure_message_redacts_credentials(self, monkeypatch,
                                                       tmp_path):
        _patch_main_deps(
            monkeypatch, tmp_path, {"config.json": BASE_CFG},
            factory_failing("camera"))

        with pytest.raises(RuntimeError) as excinfo:
            dogwatch.main()
        assert "hunter2" not in str(excinfo.value)

    def test_target_fps_ignores_cameras_that_failed_to_start(
            self, monkeypatch, tmp_path):
        # A slow camera that is offline must not throttle the loop for a fast
        # camera that is up: min() used to be taken over every config.
        slow = dict(BASE_CFG, target_fps=1)
        fast = dict(BASE_CFG, target_fps=10)
        _patch_main_deps(
            monkeypatch, tmp_path,
            {"config.json": fast, "config-rear-east.json": slow},
            factory_failing("rear-east"))

        # Let one full loop iteration complete (so its sleep is recorded), then
        # break out on the next pass.
        calls = []

        def stop_after_one_iteration(self, now):
            calls.append(now)
            if len(calls) > 1:
                raise KeyboardInterrupt
            return False

        monkeypatch.setattr(dogwatch.Heartbeat, "should_write",
                            stop_after_one_iteration)
        recorded = []
        monkeypatch.setattr(dogwatch.time, "sleep",
                            lambda s: recorded.append(s))

        dogwatch.main()

        # The 2s warm-up sleep, then a loop sleep of ~1/10s (the fast camera
        # that started), not ~1/1s (the offline slow one).
        loop_sleeps = [s for s in recorded if s != 2]
        assert loop_sleeps, "expected at least one loop sleep"
        assert all(s <= 0.1 for s in loop_sleeps), loop_sleeps

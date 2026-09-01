"""Unit tests for dogwatch.py's fleet-level config guards and error logging.

Two bugs are locked down here:

  * ``MQTT_TOPIC`` in the environment is a *single global value overriding a
    per-camera setting*. ``CameraPipeline`` reads it as
    ``os.environ.get("MQTT_TOPIC", cfg["mqtt_base_topic"])``, so setting it with
    more than one camera configured silently collapses the whole fleet onto one
    base topic. The HA discovery payloads are keyed per camera, so two cameras
    then register distinct entity pairs that both subscribe to the *same* state
    topic — camera A's dog turns on camera B's sensor, and their retained
    snapshots overwrite each other. Nothing about that is visible from the Home
    Assistant side; it just looks like both cameras see everything.

  * The per-camera ``tick()`` error handler called ``traceback.print_exc()``,
    which writes straight to stderr with no chance to redact. The line above it
    was carefully passed through ``redact()``, but a cv2/requests error raised
    from the frame grabber or the snapshot fetch embeds the credential-bearing
    camera URL in its message — and the traceback's last line *is* that message.
    So the credentials were masked on one line and leaked on the next.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dogwatch

CREDS_URL = "rtsp://bob:hunter2@cam.lan:554/Streaming/Channels/1201"


def cfg(topic=None, **extra):
    out = {"model_path": "m.tflite", "labels_path": "l.txt",
           "score_threshold": 0.5, "target_fps": 5}
    if topic is not None:
        out["mqtt_base_topic"] = topic
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# check_topic_collisions
# --------------------------------------------------------------------------

class TestTopicCollisions:
    def test_distinct_topics_are_accepted(self):
        dogwatch.check_topic_collisions(
            [cfg("dogwatch"), cfg("dogwatch/rear-east")],
            ["camera", "rear-east"], env={})

    def test_single_camera_with_env_override_is_fine(self):
        # One camera cannot collide with anything, so MQTT_TOPIC stays usable.
        dogwatch.check_topic_collisions(
            [cfg("dogwatch")], ["camera"], env={"MQTT_TOPIC": "dogwatch"})

    def test_env_override_with_multiple_cameras_is_rejected(self):
        with pytest.raises(ValueError, match="MQTT_TOPIC"):
            dogwatch.check_topic_collisions(
                [cfg("dogwatch"), cfg("dogwatch/rear-east")],
                ["camera", "rear-east"], env={"MQTT_TOPIC": "dogwatch"})

    def test_env_override_error_names_the_cameras_and_the_fix(self):
        with pytest.raises(ValueError) as excinfo:
            dogwatch.check_topic_collisions(
                [cfg("dogwatch"), cfg("dogwatch/rear-east")],
                ["camera", "rear-east"], env={"MQTT_TOPIC": "dogwatch"})
        msg = str(excinfo.value)
        assert "camera" in msg and "rear-east" in msg
        assert "mqtt_base_topic" in msg

    def test_empty_env_override_is_ignored(self):
        # An unset-but-present empty value must not trip the guard.
        dogwatch.check_topic_collisions(
            [cfg("dogwatch"), cfg("dogwatch/rear-east")],
            ["camera", "rear-east"], env={"MQTT_TOPIC": ""})

    def test_duplicate_configured_topics_are_rejected(self):
        # The same collision reached via config rather than the environment.
        with pytest.raises(ValueError, match="mqtt_base_topic"):
            dogwatch.check_topic_collisions(
                [cfg("dogwatch"), cfg("dogwatch")],
                ["camera", "rear-east"], env={})

    def test_duplicate_topic_error_names_both_cameras(self):
        with pytest.raises(ValueError) as excinfo:
            dogwatch.check_topic_collisions(
                [cfg("dogwatch"), cfg("dogwatch")],
                ["camera", "rear-east"], env={})
        msg = str(excinfo.value)
        assert "camera" in msg and "rear-east" in msg

    def test_missing_topic_key_is_not_treated_as_a_duplicate(self):
        # Two configs that both omit mqtt_base_topic fail later, in Publisher
        # construction, with a clearer message than a bogus collision here.
        dogwatch.check_topic_collisions([cfg(), cfg()],
                                        ["camera", "rear-east"], env={})

    def test_mqtt_host_and_port_are_not_restricted(self):
        # These are genuinely global (one broker), so they must NOT be guarded.
        dogwatch.check_topic_collisions(
            [cfg("dogwatch"), cfg("dogwatch/rear-east")],
            ["camera", "rear-east"],
            env={"MQTT_HOST": "10.0.0.1", "MQTT_PORT": "1883"})

    def test_three_cameras_with_one_duplicate_pair_is_rejected(self):
        with pytest.raises(ValueError):
            dogwatch.check_topic_collisions(
                [cfg("a"), cfg("b"), cfg("a")],
                ["one", "two", "three"], env={})


# --------------------------------------------------------------------------
# main() calls the guard before doing any work
# --------------------------------------------------------------------------

class TestMainAppliesTheGuard:
    def test_main_refuses_to_start_with_a_global_topic_override(
            self, monkeypatch, tmp_path):
        for fname in ("config.json", "config-rear-east.json"):
            (tmp_path / fname).write_text(json.dumps(cfg("dogwatch")))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["dogwatch.py"])
        monkeypatch.setenv("MQTT_TOPIC", "dogwatch")

        built = []
        monkeypatch.setattr(dogwatch, "CameraPipeline",
                            lambda c, n: built.append(n))

        with pytest.raises(ValueError, match="MQTT_TOPIC"):
            dogwatch.main()

        # Must fail before binding the TPU or opening any camera.
        assert built == []


# --------------------------------------------------------------------------
# tick() error logging must not leak credentials
# --------------------------------------------------------------------------

class TestTickErrorRedaction:
    def _run_one_tick(self, monkeypatch, tmp_path, exc):
        (tmp_path / "config.json").write_text(json.dumps(cfg("dogwatch")))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["dogwatch.py"])
        monkeypatch.delenv("MQTT_TOPIC", raising=False)

        class FailingPipeline:
            name = "camera"

            def __init__(self, *a):
                pass

            def tick(self, detector, t0):
                raise exc

            def health(self, now=None, tick_seconds=None):
                return {"stale": False}

            def close(self):
                pass

        monkeypatch.setattr(dogwatch, "DogDetector",
                            lambda *a, **k: object())
        monkeypatch.setattr(dogwatch, "CameraPipeline", FailingPipeline)
        monkeypatch.setattr(dogwatch.time, "sleep", lambda *a: None)

        calls = []

        def stop_after_one(self, now):
            calls.append(now)
            if len(calls) > 1:
                raise KeyboardInterrupt
            return False

        monkeypatch.setattr(dogwatch.Heartbeat, "should_write", stop_after_one)
        dogwatch.main()

    def test_traceback_does_not_leak_credentials(self, monkeypatch, tmp_path,
                                                 capsys):
        # The realistic case: an exception whose *message* embeds the camera
        # URL, as cv2/requests errors out of the grabber or snapshot fetch do.
        self._run_one_tick(monkeypatch, tmp_path,
                           RuntimeError(f"open failed for {CREDS_URL}"))

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "hunter2" not in combined, combined
        assert "***:***@cam.lan" in combined

    def test_traceback_is_still_reported(self, monkeypatch, tmp_path, capsys):
        # Redaction must not cost us the diagnostic — the traceback body and
        # the exception type both still need to be there.
        self._run_one_tick(monkeypatch, tmp_path,
                           RuntimeError(f"open failed for {CREDS_URL}"))

        combined = capsys.readouterr().out
        assert "tick failed" in combined
        assert "RuntimeError" in combined
        assert "Traceback (most recent call last)" in combined

    def test_exception_without_a_url_is_unchanged(self, monkeypatch, tmp_path,
                                                  capsys):
        self._run_one_tick(monkeypatch, tmp_path, ValueError("plain failure"))

        combined = capsys.readouterr().out
        assert "plain failure" in combined
        assert "ValueError" in combined

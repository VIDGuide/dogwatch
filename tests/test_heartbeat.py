"""Unit tests for heartbeat.py / healthcheck.py.

What these protect: `restart: unless-stopped` only recovers a process that
*exits*. The detector's real failure mode is staying alive while going blind — a
wedged frame grabber returns the same stale frame forever. The heartbeat plus
healthcheck turn that into an observable container health status.
"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import healthcheck
import heartbeat
from heartbeat import Heartbeat, evaluate


def hb(path, **kw):
    return Heartbeat(path=str(path), log=lambda m: None, **kw)


class TestWrite:
    def test_write_creates_readable_payload(self, tmp_path):
        p = tmp_path / "hb.json"
        h = hb(p, interval=0.0)
        assert h.write(1000.0, {"camera": {"stale": False}}) is True
        data = heartbeat.read(str(p))
        assert data["ts"] == 1000.0
        assert data["cameras"] == {"camera": {"stale": False}}
        assert data["pid"] == os.getpid()

    def test_write_is_throttled(self, tmp_path):
        h = hb(tmp_path / "hb.json", interval=100.0)
        assert h.write(1000.0, {}) is True
        # A 5fps loop must not do a filesystem write every 200ms.
        assert h.write(1001.0, {}) is False
        assert h.write(1200.0, {}) is True

    def test_force_bypasses_throttle(self, tmp_path):
        h = hb(tmp_path / "hb.json", interval=100.0)
        h.write(1000.0, {})
        assert h.write(1001.0, {}, force=True) is True

    def test_should_write_matches_throttle(self, tmp_path):
        h = hb(tmp_path / "hb.json", interval=10.0)
        h.write(1000.0, {})
        assert h.should_write(1005.0) is False
        assert h.should_write(1011.0) is True

    def test_write_is_atomic_no_temp_left_behind(self, tmp_path):
        h = hb(tmp_path / "hb.json", interval=0.0)
        h.write(1000.0, {"camera": {"stale": False}})
        assert sorted(os.listdir(tmp_path)) == ["hb.json"]

    def test_unwritable_path_does_not_raise(self, tmp_path):
        # A heartbeat failure must never be able to break detection.
        h = hb(tmp_path / "no-such-dir" / "hb.json", interval=0.0)
        assert h.write(1000.0, {}) is False

    def test_unwritable_path_warns_only_once(self, tmp_path):
        logs = []
        h = Heartbeat(path=str(tmp_path / "nope" / "hb.json"), interval=0.0,
                      log=logs.append)
        for _ in range(10):
            h.write(1000.0, {})
        assert len(logs) == 1


class TestRead:
    def test_missing_file_returns_none(self, tmp_path):
        assert heartbeat.read(str(tmp_path / "nope.json")) is None

    @pytest.mark.parametrize("garbage", ["", "not json", "{trunc", "[]", "null"])
    def test_malformed_returns_none(self, tmp_path, garbage):
        p = tmp_path / "hb.json"
        p.write_text(garbage)
        assert heartbeat.read(str(p)) is None


class TestEvaluate:
    def test_no_heartbeat_is_unhealthy(self):
        healthy, reason = evaluate(None)
        assert healthy is False
        assert "no heartbeat" in reason

    def test_fresh_all_cameras_ok_is_healthy(self):
        data = {"ts": 1000.0, "interval": 5,
                "cameras": {"a": {"stale": False}, "b": {"stale": False}}}
        healthy, reason = evaluate(data, now=1001.0)
        assert healthy is True
        assert "2 camera(s) healthy" in reason

    def test_stale_heartbeat_is_unhealthy(self):
        data = {"ts": 1000.0, "interval": 5, "cameras": {"a": {"stale": False}}}
        healthy, reason = evaluate(data, now=1000.0 + 500)
        assert healthy is False
        assert "wedged" in reason

    def test_all_cameras_stale_is_unhealthy(self):
        data = {"ts": 1000.0, "interval": 5,
                "cameras": {"a": {"stale": True}, "b": {"stale": True}}}
        healthy, reason = evaluate(data, now=1001.0)
        assert healthy is False
        assert "all 2 camera(s) stale" in reason

    def test_one_of_several_stale_is_healthy_but_degraded(self):
        """Deliberate: the other cameras still work, restarting would disrupt
        them, and the stale one is already unavailable in HA. A camera unplugged
        for a day must not cause a restart loop."""
        data = {"ts": 1000.0, "interval": 5,
                "cameras": {"a": {"stale": True}, "b": {"stale": False}}}
        healthy, reason = evaluate(data, now=1001.0)
        assert healthy is True
        assert "degraded" in reason

    def test_single_camera_stale_is_unhealthy(self):
        data = {"ts": 1000.0, "interval": 5, "cameras": {"a": {"stale": True}}}
        healthy, _ = evaluate(data, now=1001.0)
        assert healthy is False

    def test_no_cameras_is_unhealthy(self):
        healthy, reason = evaluate({"ts": 1000.0, "cameras": {}}, now=1001.0)
        assert healthy is False
        assert "no cameras" in reason

    def test_missing_timestamp_is_unhealthy(self):
        healthy, reason = evaluate({"cameras": {"a": {}}})
        assert healthy is False
        assert "timestamp" in reason

    def test_tolerates_a_few_missed_heartbeats(self):
        """A single slow cycle must not flap the container to unhealthy."""
        data = {"ts": 1000.0, "interval": 5, "cameras": {"a": {"stale": False}}}
        healthy, _ = evaluate(data, now=1000.0 + 20)
        assert healthy is True

    def test_explicit_max_age_is_respected(self):
        data = {"ts": 1000.0, "interval": 5, "cameras": {"a": {"stale": False}}}
        assert evaluate(data, now=1010.0, max_age=5.0)[0] is False
        assert evaluate(data, now=1010.0, max_age=100.0)[0] is True

    def test_large_interval_widens_the_allowance(self):
        data = {"ts": 1000.0, "interval": 60, "cameras": {"a": {"stale": False}}}
        # 6 * 60 = 360s allowance
        assert evaluate(data, now=1000.0 + 300)[0] is True
        assert evaluate(data, now=1000.0 + 400)[0] is False


class TestHealthcheckEntryPoint:
    def _write(self, tmp_path, monkeypatch, payload):
        p = tmp_path / "hb.json"
        p.write_text(json.dumps(payload))
        monkeypatch.setenv("DOGWATCH_HEARTBEAT_FILE", str(p))
        return p

    def test_exit_zero_when_healthy(self, tmp_path, monkeypatch, capsys):
        self._write(tmp_path, monkeypatch,
                    {"ts": time.time(), "interval": 5,
                     "cameras": {"a": {"stale": False}}})
        assert healthcheck.main([]) == 0
        assert "ok" in capsys.readouterr().out

    def test_exit_one_when_unhealthy(self, tmp_path, monkeypatch, capsys):
        self._write(tmp_path, monkeypatch,
                    {"ts": time.time(), "interval": 5,
                     "cameras": {"a": {"stale": True}}})
        assert healthcheck.main([]) == 1
        assert "UNHEALTHY" in capsys.readouterr().out

    def test_exit_one_when_heartbeat_absent(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DOGWATCH_HEARTBEAT_FILE", str(tmp_path / "nope.json"))
        assert healthcheck.main([]) == 1
        assert "UNHEALTHY" in capsys.readouterr().out

    def test_json_mode_is_parseable(self, tmp_path, monkeypatch, capsys):
        self._write(tmp_path, monkeypatch,
                    {"ts": time.time(), "interval": 5,
                     "cameras": {"a": {"stale": False}}})
        healthcheck.main(["--json"])
        out = json.loads(capsys.readouterr().out)
        assert out["healthy"] is True
        assert "cameras" in out

    def test_json_mode_works_with_no_heartbeat(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("DOGWATCH_HEARTBEAT_FILE", str(tmp_path / "nope.json"))
        healthcheck.main(["--json"])
        out = json.loads(capsys.readouterr().out)
        assert out["healthy"] is False

    def test_healthcheck_imports_no_heavy_deps(self):
        """The healthcheck must not be able to fail because cv2/numpy is slow or
        broken to import — it runs every 60s inside the container."""
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "healthcheck.py")).read()
        for heavy in ("import cv2", "import numpy", "ai_edge_litert",
                      "import paho", "import requests"):
            assert heavy not in src


class TestHeartbeatPath:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DOGWATCH_HEARTBEAT_FILE", "/somewhere/else.json")
        assert heartbeat.heartbeat_path() == "/somewhere/else.json"

    def test_default_path(self):
        assert heartbeat.heartbeat_path({}) == heartbeat.DEFAULT_PATH

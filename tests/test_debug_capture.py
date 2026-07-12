"""Unit tests for debug_capture.py's opt-in event snapshot archiving."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_capture import DebugCapture


def make_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestDisabledByDefault:
    def test_disabled_by_default_no_directory_created(self, tmp_path):
        cfg = {"debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")
        assert dc.enabled is False
        assert not os.path.exists(dc.dir)

    def test_disabled_save_is_a_noop(self, tmp_path):
        cfg = {"debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")
        dc.save("dog_at_fence", 1, time.time(), make_frame())
        assert not os.path.exists(dc.dir)

    def test_disabled_cleanup_is_a_noop_and_does_not_raise(self, tmp_path):
        cfg = {"debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")
        dc.cleanup()  # should not raise even though nothing exists


class TestEnabledCapture:
    def test_enabled_creates_per_camera_subdirectory(self, tmp_path):
        cfg = {"debug_capture_enabled": True, "debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "rear-east")
        assert dc.enabled is True
        assert os.path.isdir(tmp_path / "captures" / "rear-east")

    def test_save_writes_low_res_file(self, tmp_path):
        cfg = {"debug_capture_enabled": True, "debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")
        ts = 1720000000.0
        dc.save("dog_at_fence", 5, ts, make_frame())
        expected = tmp_path / "captures" / "camera" / "1720000000_5_dog_at_fence_lowres.jpg"
        assert expected.exists()

    def test_save_writes_both_low_and_high_res_when_provided(self, tmp_path):
        cfg = {"debug_capture_enabled": True, "debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")
        ts = 1720000000.0
        dc.save("digging", 2, ts, make_frame(), high_res_frame=make_frame())
        cam_dir = tmp_path / "captures" / "camera"
        assert (cam_dir / "1720000000_2_digging_lowres.jpg").exists()
        assert (cam_dir / "1720000000_2_digging_highres.jpg").exists()

    def test_save_skips_high_res_file_when_not_provided(self, tmp_path):
        cfg = {"debug_capture_enabled": True, "debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")
        ts = 1720000000.0
        dc.save("dog_at_fence", 3, ts, make_frame())
        cam_dir = tmp_path / "captures" / "camera"
        assert not (cam_dir / "1720000000_3_dog_at_fence_highres.jpg").exists()

    def test_save_failure_is_caught_not_raised(self, tmp_path, monkeypatch):
        cfg = {"debug_capture_enabled": True, "debug_capture_dir": str(tmp_path / "captures")}
        dc = DebugCapture(cfg, "camera")

        import cv2
        def boom(*a, **k):
            raise RuntimeError("disk full (simulated)")
        monkeypatch.setattr(cv2, "imwrite", boom)

        dc.save("dog_at_fence", 1, time.time(), make_frame())  # must not raise


class TestCleanup:
    def test_retention_zero_keeps_everything(self, tmp_path):
        cfg = {
            "debug_capture_enabled": True,
            "debug_capture_dir": str(tmp_path / "captures"),
            "debug_capture_retention_days": 0,
        }
        dc = DebugCapture(cfg, "camera")
        dc.save("dog_at_fence", 1, time.time(), make_frame())
        dc.cleanup()
        assert len(os.listdir(dc.dir)) == 1

    def test_cleanup_removes_files_older_than_retention(self, tmp_path):
        cfg = {
            "debug_capture_enabled": True,
            "debug_capture_dir": str(tmp_path / "captures"),
            "debug_capture_retention_days": 7,
        }
        dc = DebugCapture(cfg, "camera")
        dc.save("dog_at_fence", 1, time.time(), make_frame())

        old_file = os.path.join(dc.dir, "old_file_lowres.jpg")
        with open(old_file, "wb") as f:
            f.write(b"fake old jpeg")
        old_time = time.time() - (10 * 86400)  # 10 days ago
        os.utime(old_file, (old_time, old_time))

        dc.cleanup()

        remaining = os.listdir(dc.dir)
        assert "old_file_lowres.jpg" not in remaining
        assert len(remaining) == 1  # only the fresh one from save() above

    def test_cleanup_keeps_files_within_retention_window(self, tmp_path):
        cfg = {
            "debug_capture_enabled": True,
            "debug_capture_dir": str(tmp_path / "captures"),
            "debug_capture_retention_days": 7,
        }
        dc = DebugCapture(cfg, "camera")
        dc.save("dog_at_fence", 1, time.time(), make_frame())
        dc.cleanup()
        assert len(os.listdir(dc.dir)) == 1

    def test_cleanup_on_missing_directory_does_not_raise(self, tmp_path):
        cfg = {
            "debug_capture_enabled": True,
            "debug_capture_dir": str(tmp_path / "does_not_exist_yet"),
            "debug_capture_retention_days": 7,
        }
        dc = DebugCapture(cfg, "camera")
        import shutil
        shutil.rmtree(dc.dir)  # remove the dir __init__ just created
        dc.cleanup()  # should not raise

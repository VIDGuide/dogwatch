"""Unit tests for async_writer.py.

Purpose of the module under test: a full-resolution `cv2.imwrite` plus a SQLite
commit used to run inline in `CameraPipeline.tick()`, inside a loop budget of
`1/target_fps` shared across the whole camera fleet (200ms at the default 5fps).
So firing an event stalled detection exactly when frames mattered most.

These tests pin down the properties that make offloading safe: never block the
caller, never raise at the caller, bound memory, and drain on shutdown.
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from async_writer import AsyncWriter


@pytest.fixture
def writer():
    w = AsyncWriter(name="test", maxsize=8, log=lambda msg: None)
    yield w
    w.stop(timeout=2.0)


class TestBasicExecution:
    def test_task_runs_on_worker_thread(self, writer):
        seen = {}

        def task():
            seen["thread"] = threading.current_thread().name

        writer.submit(task)
        assert writer.flush(timeout=2.0)
        assert seen["thread"] != threading.current_thread().name
        assert "async-writer-test" in seen["thread"]

    def test_args_and_kwargs_are_passed(self, writer):
        got = {}
        writer.submit(lambda a, b=None: got.update(a=a, b=b), 1, b=2)
        assert writer.flush(timeout=2.0)
        assert got == {"a": 1, "b": 2}

    def test_tasks_run_in_submission_order(self):
        """Single worker: ordering is preserved, so an event's DB row and its
        clip file stay consistent.

        Uses its own generously sized queue — the shared fixture's maxsize=8
        exists to exercise overflow, and would drop items here.
        """
        order = []
        w = AsyncWriter(name="order", maxsize=256, log=lambda m: None)
        try:
            for i in range(20):
                assert w.submit(order.append, i) is True
            assert w.flush(timeout=2.0)
            assert order == list(range(20))
        finally:
            w.stop(timeout=2.0)

    def test_completed_counter_tracks_work(self, writer):
        for _ in range(5):
            writer.submit(lambda: None)
        assert writer.flush(timeout=2.0)
        assert writer.stats()["completed"] == 5
        assert writer.stats()["submitted"] == 5


class TestNonBlocking:
    def test_submit_returns_immediately_while_task_is_slow(self, writer):
        release = threading.Event()
        writer.submit(release.wait)  # occupies the worker

        t0 = time.time()
        writer.submit(lambda: None)
        elapsed = time.time() - t0

        release.set()
        # The caller is the detection loop; submission must be ~free.
        assert elapsed < 0.1

    def test_submit_never_raises_when_queue_is_full(self, writer):
        release = threading.Event()
        writer.submit(release.wait)
        # Overfill well past maxsize=8.
        results = [writer.submit(lambda: None) for _ in range(50)]
        release.set()
        assert False in results          # some were dropped
        assert all(r in (True, False) for r in results)  # but nothing raised


class TestBoundedness:
    def test_queue_depth_never_exceeds_maxsize(self, writer):
        release = threading.Event()
        writer.submit(release.wait)
        for _ in range(100):
            writer.submit(lambda: None)
        assert writer.depth <= 8
        release.set()

    def test_drops_are_counted(self, writer):
        release = threading.Event()
        writer.submit(release.wait)
        for _ in range(100):
            writer.submit(lambda: None)
        release.set()
        assert writer.stats()["dropped"] > 0

    def test_drop_is_logged_but_rate_limited(self):
        logs = []
        w = AsyncWriter(name="t", maxsize=2, log=logs.append)
        try:
            release = threading.Event()
            w.submit(release.wait)
            for _ in range(50):
                w.submit(lambda: None)
            release.set()
            # Loud enough to notice, quiet enough not to flood.
            assert len(logs) == 1
            assert "FULL" in logs[0]
        finally:
            w.stop(timeout=2.0)


class TestErrorIsolation:
    def test_failing_task_does_not_kill_the_worker(self, writer):
        def boom():
            raise RuntimeError("disk on fire")

        after = []
        writer.submit(boom)
        writer.submit(after.append, "still alive")
        assert writer.flush(timeout=2.0)
        assert after == ["still alive"]
        assert writer.stats()["errors"] == 1

    def test_failing_task_does_not_raise_at_submitter(self, writer):
        writer.submit(lambda: 1 / 0)
        assert writer.flush(timeout=2.0)  # no exception escapes to the caller
        assert writer.stats()["errors"] == 1

    def test_errors_are_logged_rate_limited(self):
        logs = []
        w = AsyncWriter(name="t", maxsize=64, log=logs.append)
        try:
            for _ in range(20):
                w.submit(lambda: 1 / 0)
            w.flush(timeout=2.0)
            assert len(logs) == 1
            assert "failed" in logs[0]
            assert w.stats()["errors"] == 20
        finally:
            w.stop(timeout=2.0)

    def test_label_appears_in_error_log(self):
        logs = []
        w = AsyncWriter(name="t", log=logs.append)
        try:
            w.submit(lambda: 1 / 0, label="clip_imwrite")
            w.flush(timeout=2.0)
            assert "clip_imwrite" in logs[0]
        finally:
            w.stop(timeout=2.0)


class TestShutdown:
    def test_stop_drains_queued_work(self):
        """An event firing moments before shutdown must still land on disk."""
        done = []
        w = AsyncWriter(name="t", maxsize=64, log=lambda m: None)
        for i in range(20):
            w.submit(done.append, i)
        w.stop(timeout=5.0)
        assert done == list(range(20))

    def test_submit_after_stop_is_rejected_not_raised(self):
        w = AsyncWriter(name="t", log=lambda m: None)
        w.stop(timeout=2.0)
        assert w.submit(lambda: None) is False

    def test_stop_is_idempotent(self):
        w = AsyncWriter(name="t", log=lambda m: None)
        w.stop(timeout=2.0)
        w.stop(timeout=2.0)  # must not hang or raise

    def test_stop_completes_even_with_a_saturated_queue(self):
        w = AsyncWriter(name="t", maxsize=4, log=lambda m: None)
        release = threading.Event()
        w.submit(release.wait)
        for _ in range(50):
            w.submit(lambda: None)
        release.set()
        t0 = time.time()
        w.stop(timeout=5.0)
        assert time.time() - t0 < 5.0


class TestStats:
    def test_stats_shape(self, writer):
        s = writer.stats()
        assert set(s) == {"submitted", "completed", "dropped", "errors", "depth"}

    def test_flush_timeout_returns_false_when_work_outstanding(self, writer):
        release = threading.Event()
        writer.submit(release.wait)
        writer.submit(lambda: None)
        assert writer.flush(timeout=0.05) is False
        release.set()

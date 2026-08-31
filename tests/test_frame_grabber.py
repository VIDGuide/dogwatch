"""Unit tests for frame_grabber.py's staleness / health signalling.

The point of these tests: a frozen frame is byte-identical to a static scene,
so without a capture timestamp a dead reader is indistinguishable from a quiet
yard. That was silent, and combined with the motion gate's forced periodic
detection it made a frozen frame containing a dog re-fire forever.

No real cv2.VideoCapture is ever opened: the backend constructor and the decode
thread are both stubbed so the bookkeeping can be exercised deterministically.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_grabber import FrameGrabber

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)
CREDS_URL = "rtsp://bob:hunter2@cam.lan:554/stream"


class FakeCapture:
    """Stand-in for cv2.VideoCapture with a scripted read() sequence.

    Once the script is exhausted it keeps returning a good frame, so a test can
    never spin forever on repeated failures.
    """

    def __init__(self, results=None, then_ok=True):
        self._results = list(results or [])
        self._then_ok = then_ok
        self.released = False
        self.reads = 0

    def read(self):
        self.reads += 1
        if self._results:
            return self._results.pop(0)
        return (True, FRAME.copy()) if self._then_ok else (False, None)

    def release(self):
        self.released = True


class StubGrabber(FrameGrabber):
    """FrameGrabber with no real video I/O and no background thread."""

    def __init__(self, initial=None, reopens=None, **kw):
        self._initial = initial if initial is not None else FakeCapture()
        self._reopens = list(reopens or [])
        self.opened = 0
        self._first_open = True
        kw.setdefault("name", "test-cam")
        super().__init__(CREDS_URL, **kw)

    def _open_capture(self):
        self.opened += 1
        if self._first_open:
            self._first_open = False
            return self._initial
        if self._reopens:
            return self._reopens.pop(0)
        return FakeCapture()

    def _start_thread(self):
        # Tests drive the loop explicitly; no background thread.
        self._thread = None


def make(**kw):
    return StubGrabber(**kw)


def run_loop_until(grabber, stores=1):
    """Run _loop_cpu, stopping after *stores* successful frame stores."""
    grabber.min_interval = 0.0
    grabber.reconnect_delay = 0.0
    original = grabber._store
    seen = {"n": 0}

    def counting(frame):
        original(frame)
        seen["n"] += 1
        if seen["n"] >= stores:
            grabber.running = False

    grabber._store = counting
    grabber._loop_cpu()
    return seen["n"]


class TestFrameAge:
    def test_infinite_age_before_any_frame(self):
        g = make()
        assert g.frame_age() == float("inf")
        assert g.read() is None
        assert g.read_with_ts() == (None, None)

    def test_never_started_counts_as_stale(self):
        # inf > any limit, so "never produced a frame" and "long dead" collapse
        # into one comparison for callers.
        assert make().is_stale(30) is True

    def test_fresh_frame_is_not_stale(self):
        g = make()
        g._store(FRAME)
        assert g.frame_age() < 1.0
        assert g.is_stale(30) is False

    def test_old_frame_is_stale(self):
        g = make()
        g._store(FRAME)
        with g.lock:
            g.frame_ts = time.time() - 120
        assert g.is_stale(30) is True
        assert g.frame_age() >= 120

    @pytest.mark.parametrize("limit", [0, None, -1])
    def test_falsy_or_negative_limit_disables_staleness(self, limit):
        g = make()
        g._store(FRAME)
        with g.lock:
            g.frame_ts = time.time() - 9999
        assert g.is_stale(limit) is False

    def test_explicit_now_is_used(self):
        g = make()
        g._store(FRAME)
        with g.lock:
            g.frame_ts = 1000.0
        assert g.frame_age(now=1050.0) == pytest.approx(50.0)

    def test_age_never_negative_on_clock_skew(self):
        g = make()
        g._store(FRAME)
        with g.lock:
            g.frame_ts = 2000.0
        assert g.frame_age(now=1000.0) == 0.0


class TestReadWithTs:
    def test_returns_frame_and_matching_ts(self):
        g = make()
        g._store(FRAME)
        frame, ts = g.read_with_ts()
        assert frame is not None
        assert ts == g.frame_ts

    def test_returns_a_copy_not_the_stored_buffer(self):
        g = make()
        g._store(FRAME)
        frame, _ = g.read_with_ts()
        frame[0, 0, 0] = 99
        assert g.frame[0, 0, 0] == 0


class TestHealth:
    def test_reports_no_frame_initially(self):
        h = make().health()
        assert h["has_frame"] is False
        assert h["reconnects"] == 0
        assert h["consecutive_failures"] == 0

    def test_reports_frame_after_store(self):
        g = make()
        g._store(FRAME)
        assert g.health()["has_frame"] is True

    def test_failure_increments_counters(self):
        g = make()
        g._note_failure(0.0, "boom")
        h = g.health()
        assert h["reconnects"] == 1
        assert h["consecutive_failures"] == 1
        assert h["last_error"] == "boom"

    def test_successful_store_resets_consecutive_failures(self):
        g = make()
        g._note_failure(0.0, "boom")
        g._store(FRAME)
        h = g.health()
        assert h["consecutive_failures"] == 0
        assert h["last_error"] is None
        # Cumulative reconnect count is kept for diagnostics.
        assert h["reconnects"] == 1


class TestBackoff:
    """_next_delay is a pure function precisely so the backoff curve can be
    tested without _note_failure's real time.sleep()."""

    def test_delay_doubles(self):
        assert FrameGrabber._next_delay(0.5) == pytest.approx(1.0)
        assert FrameGrabber._next_delay(1.0) == pytest.approx(2.0)
        assert FrameGrabber._next_delay(2.0) == pytest.approx(4.0)

    def test_delay_is_capped(self):
        cap = FrameGrabber.MAX_RECONNECT_DELAY
        assert FrameGrabber._next_delay(cap) == cap
        assert FrameGrabber._next_delay(10_000) == cap

    def test_zero_delay_jumps_to_cap_rather_than_staying_zero(self):
        # Doubling zero stays zero forever, which would be a hot retry loop.
        assert FrameGrabber._next_delay(0.0) == FrameGrabber.MAX_RECONNECT_DELAY

    def test_reaches_cap_in_a_bounded_number_of_steps(self):
        d = 0.5
        for _ in range(20):
            d = FrameGrabber._next_delay(d)
        assert d == FrameGrabber.MAX_RECONNECT_DELAY

    def test_first_failure_is_logged(self, capsys):
        g = make()
        g._note_failure(0.0, "connection refused")
        out = capsys.readouterr().out
        assert "connection refused" in out
        assert "test-cam" in out

    def test_repeat_failures_are_not_logged_every_time(self, capsys):
        """A camera that is simply switched off must not flood the log at the
        retry rate — but it must not go completely silent either."""
        g = make()
        for _ in range(10):
            g._note_failure(0.0, "down")
        out = capsys.readouterr().out
        assert out.count("down") == 1

    def test_recovery_is_logged(self, capsys):
        g = make()
        g._note_failure(0.0, "down")
        capsys.readouterr()
        g._store(FRAME)
        assert "recovered" in capsys.readouterr().out


class TestLoopCpu:
    def test_stores_frames_with_timestamps(self):
        g = make(initial=FakeCapture([(True, FRAME.copy()), (True, FRAME.copy())]))
        assert run_loop_until(g, stores=2) == 2
        assert g.frame_ts is not None

    def test_read_failure_releases_and_reopens(self):
        bad = FakeCapture([(False, None)], then_ok=False)
        good = FakeCapture([(True, FRAME.copy())])
        g = make(initial=bad, reopens=[good])
        run_loop_until(g, stores=1)
        assert bad.released is True
        # 1 open in __init__ + 1 reopen after the failure.
        assert g.opened == 2

    def test_cv2_error_does_not_kill_the_loop(self):
        import cv2

        class ExplodingCapture(FakeCapture):
            def __init__(self):
                super().__init__()
                self.first = True

            def read(self):
                if self.first:
                    self.first = False
                    raise cv2.error("malformed stream")
                return True, FRAME.copy()

        g = make(initial=ExplodingCapture(), reopens=[FakeCapture()])
        # Previously a cv2.error propagated and killed the daemon thread
        # outright, after which read() returned the same stale frame forever.
        assert run_loop_until(g, stores=1) == 1
        assert g.health()["reconnects"] >= 1


class TestLoopWrapper:
    def test_thread_death_is_flagged_and_logged(self, capsys):
        g = make()

        def boom():
            raise RuntimeError("unexpected")

        g._loop_cpu = boom
        g._loop()
        assert g.thread_alive is False
        out = capsys.readouterr().out
        assert "FATAL" in out

    def test_thread_death_message_redacts_credentials(self, capsys):
        g = make()

        def boom():
            raise RuntimeError(f"failed on {CREDS_URL}")

        g._loop_cpu = boom
        g._loop()
        out = capsys.readouterr().out
        assert "hunter2" not in out
        assert "***:***" in out


class TestStop:
    def test_stop_clears_running_and_releases(self):
        cap = FakeCapture()
        g = make(initial=cap)
        g.stop()
        assert g.running is False
        assert cap.released is True

    def test_stop_is_safe_to_call_twice(self):
        g = make()
        g.stop()
        g.stop()
        assert g.running is False

"""Tests for event_store.py — SQLite event logging."""
import os
import tempfile
import time

import pytest

from event_store import EventStore


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestEventStoreBasics:
    def test_log_and_query(self, tmp_db):
        store = EventStore({"event_store_path": tmp_db}, "camera")
        store.log_event("dog_at_fence", 1, 0.85, [100, 200, 300, 400],
                        640, 480, time.time())
        events = store.query_recent()
        assert len(events) == 1
        assert events[0]["event_type"] == "dog_at_fence"
        assert events[0]["score"] == 0.85
        assert events[0]["camera"] == "camera"
        store.close()

    def test_multiple_events(self, tmp_db):
        store = EventStore({"event_store_path": tmp_db}, "rear-east")
        t = time.time()
        store.log_event("dog_at_fence", 1, 0.7, [10, 20, 30, 40], 864, 727, t)
        store.log_event("digging", 1, 0.7, [10, 20, 30, 40], 864, 727, t + 5)
        store.log_event("dog_at_fence", 2, 0.9, [50, 60, 70, 80], 864, 727, t + 10)

        events = store.query_recent(limit=2)
        assert len(events) == 2
        # Most recent first
        assert events[0]["track_id"] == 2
        store.close()

    def test_query_filter_camera(self, tmp_db):
        store = EventStore({"event_store_path": tmp_db}, "camera")
        t = time.time()
        store.log_event("dog_at_fence", 1, 0.5, [0, 0, 1, 1], 640, 480, t)

        store2 = EventStore({"event_store_path": tmp_db}, "rear-east")
        store2.log_event("dog_at_fence", 1, 0.6, [0, 0, 1, 1], 864, 727, t + 1)

        events = store.query_recent(camera="camera")
        assert len(events) == 1
        assert events[0]["camera"] == "camera"
        store.close()
        store2.close()

    def test_query_filter_since_ts(self, tmp_db):
        store = EventStore({"event_store_path": tmp_db}, "camera")
        t = time.time()
        store.log_event("dog_at_fence", 1, 0.5, [0, 0, 1, 1], 640, 480, t - 100)
        store.log_event("dog_at_fence", 2, 0.6, [0, 0, 1, 1], 640, 480, t)

        events = store.query_recent(since_ts=t - 10)
        assert len(events) == 1
        assert events[0]["track_id"] == 2
        store.close()

    def test_metadata_stored_as_json(self, tmp_db):
        store = EventStore({"event_store_path": tmp_db}, "camera")
        meta = {"motion_fraction": 0.03, "note": "test"}
        store.log_event("dog_at_fence", 1, 0.8, [0, 0, 1, 1], 640, 480,
                        time.time(), metadata=meta)
        events = store.query_recent()
        import json
        assert json.loads(events[0]["metadata"]) == meta
        store.close()


class TestEventStoreDisabled:
    def test_disabled_noop(self, tmp_db):
        store = EventStore({"event_store_enabled": False, "event_store_path": tmp_db})
        store.log_event("dog_at_fence", 1, 0.5, [0, 0, 1, 1], 640, 480, time.time())
        events = store.query_recent()
        assert events == []
        store.close()

    def test_disabled_no_file_created(self):
        path = "/tmp/should_not_exist_dogwatch_test.db"
        try:
            os.unlink(path)
        except OSError:
            pass
        store = EventStore({"event_store_enabled": False, "event_store_path": path})
        store.log_event("dog_at_fence", 1, 0.5, [0, 0, 1, 1], 640, 480, time.time())
        assert not os.path.exists(path)
        store.close()


class TestEventStoreThreadSafety:
    def test_concurrent_writes(self, tmp_db):
        import threading
        store = EventStore({"event_store_path": tmp_db}, "camera")
        t = time.time()

        def writer(offset):
            for i in range(20):
                store.log_event("dog_at_fence", offset + i, 0.5,
                                [0, 0, 1, 1], 640, 480, t + offset + i)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        events = store.query_recent(limit=200)
        assert len(events) == 100  # 5 threads * 20 writes
        store.close()



# ---------------------------------------------------------------------------
# Retention / locking
# ---------------------------------------------------------------------------

class TestRetention:
    """The events table previously had no pruning available at all — one
    append-only row per event, growing for the life of the deployment."""

    def _store(self, tmp_path, **cfg):
        cfg.setdefault("event_store_path", str(tmp_path / "events.db"))
        return EventStore(cfg, camera_name="rear-east")

    def _add(self, store, ts, etype="dog_at_fence"):
        store.log_event(event_type=etype, track_id=1, score=0.9,
                        bbox=(0, 0, 10, 10), frame_w=100, frame_h=100, ts=ts)

    def test_retention_defaults_to_keep_forever(self, tmp_path):
        store = self._store(tmp_path)
        try:
            assert store.retention_days == 0
            self._add(store, 0.0)  # epoch — as old as it gets
            assert store.prune(now=1e9) == 0
            assert store.count() == 1
        finally:
            store.close()

    def test_prune_removes_only_old_rows(self, tmp_path):
        now = 1_000_000.0
        store = self._store(tmp_path, event_store_retention_days=7)
        try:
            self._add(store, now - 10 * 86400)   # 10 days old
            self._add(store, now - 8 * 86400)    # 8 days old
            self._add(store, now - 1 * 86400)    # 1 day old
            self._add(store, now)                # now
            assert store.count() == 4
            assert store.prune(now=now) == 2
            assert store.count() == 2
        finally:
            store.close()

    def test_prune_is_idempotent(self, tmp_path):
        now = 1_000_000.0
        store = self._store(tmp_path, event_store_retention_days=1)
        try:
            self._add(store, now - 5 * 86400)
            assert store.prune(now=now) == 1
            assert store.prune(now=now) == 0
        finally:
            store.close()

    def test_fractional_retention_days(self, tmp_path):
        now = 1_000_000.0
        store = self._store(tmp_path, event_store_retention_days=0.5)
        try:
            self._add(store, now - 3600 * 20)  # 20h old
            self._add(store, now - 3600 * 2)   # 2h old
            assert store.prune(now=now) == 1
        finally:
            store.close()

    def test_prune_is_noop_when_disabled(self, tmp_path):
        store = EventStore({"event_store_enabled": False,
                            "event_store_retention_days": 1})
        assert store.prune(now=1e9) == 0
        assert store.count() == 0

    def test_maybe_prune_is_rate_limited(self, tmp_path):
        now = 1_000_000.0
        store = self._store(tmp_path, event_store_retention_days=1)
        try:
            self._add(store, now - 5 * 86400)
            assert store.maybe_prune(now=now, interval=3600) == 1
            self._add(store, now - 5 * 86400)
            # Within the interval — skipped, so the new old row survives.
            assert store.maybe_prune(now=now + 10, interval=3600) == 0
            assert store.count() == 1
            # Past the interval — pruned.
            assert store.maybe_prune(now=now + 4000, interval=3600) == 1
        finally:
            store.close()

    def test_pruning_logs_when_it_removes_rows(self, tmp_path, capsys):
        now = 1_000_000.0
        store = self._store(tmp_path, event_store_retention_days=1)
        try:
            self._add(store, now - 5 * 86400)
            store.prune(now=now)
            out = capsys.readouterr().out
            assert "pruned 1 event" in out
            assert "rear-east" in out
        finally:
            store.close()

    def test_recent_events_still_queryable_after_prune(self, tmp_path):
        now = 1_000_000.0
        store = self._store(tmp_path, event_store_retention_days=7)
        try:
            self._add(store, now - 30 * 86400, etype="digging")
            self._add(store, now, etype="digging")
            store.prune(now=now)
            rows = store.query_recent(limit=10)
            assert len(rows) == 1
            assert rows[0]["ts"] == now
        finally:
            store.close()


class TestBusyTimeout:
    """A concurrent writer could previously raise 'database is locked' straight
    out of the detection path."""

    def test_busy_timeout_is_configured(self, tmp_path):
        store = EventStore({"event_store_path": str(tmp_path / "e.db"),
                            "event_store_busy_timeout_ms": 1234})
        try:
            got = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert got == 1234
        finally:
            store.close()

    def test_default_busy_timeout_is_nonzero(self, tmp_path):
        store = EventStore({"event_store_path": str(tmp_path / "e.db")})
        try:
            assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0
        finally:
            store.close()

    def test_two_stores_on_one_db_can_both_write(self, tmp_path):
        """Each camera constructs its own EventStore against the same file."""
        path = str(tmp_path / "shared.db")
        a = EventStore({"event_store_path": path}, camera_name="camera")
        b = EventStore({"event_store_path": path}, camera_name="rear-east")
        try:
            for store in (a, b):
                store.log_event(event_type="dog_at_fence", track_id=1, score=0.5,
                                bbox=(0, 0, 1, 1), frame_w=10, frame_h=10,
                                ts=1000.0)
            assert a.count() == 2
        finally:
            a.close()
            b.close()

    def test_count_on_disabled_store(self, tmp_path):
        store = EventStore({"event_store_enabled": False})
        assert store.count() == 0

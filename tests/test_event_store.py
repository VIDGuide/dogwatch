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

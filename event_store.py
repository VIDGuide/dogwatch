"""event_store.py — lightweight SQLite event log for detection events.

Replaces the pattern of grepping container stdout to answer questions like
"when did the last detection fire?" / "how many false positives this week?"
/ "what was the score on that event at 16:46?" — all of which previously
required SSH + log archaeology.

The database is a single file (default: ``events.db`` in the working dir,
overridable via config) with one table. No ORM, no dependencies beyond
stdlib sqlite3. Designed to be append-only and cheap enough to call on
every event without measurable overhead.

Config keys (per-camera or global, all optional):
    "event_store_enabled": true,          # default true
    "event_store_path": "data/events.db",      # path to SQLite file
"""
import json
import os
import sqlite3
import threading


class EventStore:
    """Append-only SQLite event log, thread-safe."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        camera TEXT NOT NULL,
        event_type TEXT NOT NULL,
        track_id INTEGER,
        score REAL,
        bbox TEXT,
        frame_w INTEGER,
        frame_h INTEGER,
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
    CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);
    """

    def __init__(self, cfg, camera_name="camera"):
        self.enabled = cfg.get("event_store_enabled", True)
        self.camera_name = camera_name
        self._db_path = cfg.get("event_store_path", "data/events.db")
        self._lock = threading.Lock()
        self._conn = None

        if self.enabled:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.executescript(self.SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

    def log_event(self, event_type, track_id, score, bbox, frame_w, frame_h,
                  ts, metadata=None):
        """Record a detection event. No-op if store is disabled.

        Parameters
        ----------
        event_type : str — "dog_at_fence" or "digging"
        track_id : int
        score : float — detection confidence
        bbox : tuple/list of 4 ints (x0, y0, x1, y1)
        frame_w, frame_h : int — detection frame dimensions
        ts : float — event timestamp
        metadata : dict or None — extra context (JSON-serialized)
        """
        if not self.enabled or self._conn is None:
            return

        bbox_json = json.dumps([int(v) for v in bbox]) if bbox else None
        meta_json = json.dumps(metadata) if metadata else None

        with self._lock:
            self._conn.execute(
                """INSERT INTO events
                   (ts, camera, event_type, track_id, score, bbox, frame_w, frame_h, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, self.camera_name, event_type, track_id, score,
                 bbox_json, frame_w, frame_h, meta_json),
            )
            self._conn.commit()

    def query_recent(self, limit=50, camera=None, since_ts=None):
        """Retrieve recent events. Returns list of dicts.

        Useful for diagnostic scripts, CLI tools, and the notifier's
        event-history display.
        """
        if not self.enabled or self._conn is None:
            return []

        sql = "SELECT * FROM events WHERE 1=1"
        params = []
        if camera:
            sql += " AND camera = ?"
            params.append(camera)
        if since_ts:
            sql += " AND ts >= ?"
            params.append(since_ts)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

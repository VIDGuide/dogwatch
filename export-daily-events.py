#!/usr/bin/env python3
"""Export yesterday's detection events with scores to a JSON file.

Writes to /app/clips/daily-events.json so the n8n daily report script
(at /mnt/clips/daily-events.json) can include confidence percentages.

Run via host cron (e.g. 5 min before the report):
    docker exec dogwatch python3 /app/export-daily-events.py
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta

DB_PATH = "/app/data/events.db"
OUTPUT = "/app/clips/daily-events.json"

AEST = timezone(timedelta(hours=10))

def main():
    now = datetime.now(AEST)
    yesterday = now - timedelta(days=1)
    day_start = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=AEST)
    day_end = datetime(now.year, now.month, now.day, tzinfo=AEST)

    ts_start = day_start.timestamp()
    ts_end = day_end.timestamp()

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            """SELECT event_type, track_id, score, ts, bbox, frame_w, frame_h, metadata
               FROM events
               WHERE ts >= ? AND ts < ?
               ORDER BY ts ASC""",
            (ts_start, ts_end),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        # Write an error marker so the report script knows what happened
        result = {"error": str(e), "events": []}
        with open(OUTPUT, "w") as f:
            json.dump(result, f)
        print(f"export-daily-events: failed to query DB: {e}")
        return

    events = []
    for row in rows:
        event_type, track_id, score, ts, bbox_json, fw, fh, meta_json = row
        events.append({
            "type": event_type,
            "track": track_id,
            "score": round(score, 4) if score else None,
            "ts": ts,
            "time": datetime.fromtimestamp(ts, AEST).strftime("%H:%M:%S"),
        })

    summary = {
        "date": day_start.strftime("%Y-%m-%d"),
        "event_count": len(events),
        "dog_at_fence_count": sum(1 for e in events if e["type"] == "dog_at_fence"),
        "digging_count": sum(1 for e in events if e["type"] == "digging"),
        "events": events,
    }

    with open(OUTPUT, "w") as f:
        json.dump(summary, f)

    print(f"export-daily-events: wrote {len(events)} events for {summary['date']} to {OUTPUT}")

if __name__ == "__main__":
    main()

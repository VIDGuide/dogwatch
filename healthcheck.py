#!/usr/bin/env python3
"""healthcheck.py — Docker HEALTHCHECK entry point for the detector container.

Reads the heartbeat dogwatch.py writes and exits 0 (healthy) or 1 (unhealthy).
See heartbeat.py for what counts as unhealthy and why a single stale camera
among several deliberately does not.

Kept dependency-free (stdlib only, no cv2/numpy import) so the healthcheck stays
fast and cannot itself fail because of a heavy import.

Usage:
    python /app/healthcheck.py          # exit 0/1, one line of output
    python /app/healthcheck.py --json   # machine-readable, for the watchdog
"""
import json
import sys

import heartbeat


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    as_json = "--json" in argv

    data = heartbeat.read()
    healthy, reason = heartbeat.evaluate(data)

    if as_json:
        out = {"healthy": healthy, "reason": reason}
        if isinstance(data, dict):
            out["cameras"] = data.get("cameras", {})
            out["ts"] = data.get("ts")
        print(json.dumps(out))
    else:
        print(f"{'ok' if healthy else 'UNHEALTHY'}: {reason}")

    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())

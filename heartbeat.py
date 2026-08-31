"""heartbeat.py — publish "is the detector actually still watching?" to disk.

## Why this exists

`restart: unless-stopped` only recovers a process that *exits*. It cannot see
the failure mode this system is actually prone to: the process stays alive and
goes blind.

  * A frame grabber thread dies or wedges, and `read()` keeps returning the same
    stale frame forever. Before frames carried timestamps this was
    indistinguishable from a perfectly static scene.
  * The MQTT publisher fails to construct, so the camera detects but publishes
    nothing.
  * The main loop overruns its interval indefinitely and real fps collapses.

None of those change the exit code, so nothing noticed. The heartbeat records
the loop's last iteration time plus each camera's frame age; `healthcheck.py`
turns that into a container health status, and `pipeline/dogwatch-watchdog.sh`
acts on it.

Written atomically (tmp + replace) so a reader never sees a torn file, and
throttled so a 5fps loop isn't doing a filesystem write every 200ms. Every
failure is swallowed: a heartbeat that could break detection would be worse than
no heartbeat at all.
"""
import json
import os
import tempfile
import time

#: Default location. /tmp is normally tmpfs, so this costs no real disk I/O, and
#: it resets on container recreate — which is correct, since a fresh container
#: has no history worth keeping.
DEFAULT_PATH = "/tmp/dogwatch-heartbeat.json"


def heartbeat_path(env=None):
    env = env if env is not None else os.environ
    return env.get("DOGWATCH_HEARTBEAT_FILE", DEFAULT_PATH)


class Heartbeat:
    """Throttled, atomic writer for detector liveness state."""

    def __init__(self, path=None, interval=5.0, log=print):
        self.path = path or heartbeat_path()
        self.interval = float(interval)
        self._log = log
        self._last_write = 0.0
        self._warned = False

    def should_write(self, now):
        return (now - self._last_write) >= self.interval

    def write(self, now, cameras, force=False):
        """Record *cameras* status. Returns True if a write happened.

        *cameras* maps camera name -> dict (frame_age, stale, publishing, ...).
        """
        if not force and not self.should_write(now):
            return False
        self._last_write = now
        payload = {
            "ts": now,
            "pid": os.getpid(),
            "interval": self.interval,
            "cameras": cameras,
        }
        try:
            directory = os.path.dirname(self.path) or "."
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".heartbeat-")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            return True
        except Exception as exc:
            # Warn once, then stay quiet — this must never become the reason
            # detection is noisy or broken.
            if not self._warned:
                self._warned = True
                self._log(f"heartbeat: cannot write {self.path}: {exc} "
                          f"(healthcheck will report unhealthy)")
            return False


def read(path=None):
    """Load the heartbeat, or None if missing/unreadable/malformed."""
    path = path or heartbeat_path()
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def evaluate(data, now=None, max_age=None):
    """Return ``(healthy: bool, reason: str)`` for a heartbeat payload.

    Unhealthy when:
      * there is no heartbeat at all (never started, or cannot write), or
      * the main loop has not ticked within *max_age*, or
      * **every** camera's frames are stale.

    A *single* stale camera among several is deliberately NOT unhealthy: the
    other cameras are still working, restarting would disrupt them too, and that
    camera is already surfaced as unavailable in Home Assistant. A camera that is
    genuinely unplugged for a day should not cause a restart loop.
    """
    if data is None:
        return False, "no heartbeat file (detector never started, or cannot write it)"

    now = time.time() if now is None else now
    try:
        ts = float(data.get("ts"))
    except (TypeError, ValueError):
        return False, "heartbeat has no usable timestamp"

    if max_age is None:
        # Tolerate a few missed heartbeats before declaring failure.
        interval = float(data.get("interval") or 5.0)
        max_age = max(60.0, interval * 6)

    age = now - ts
    if age > max_age:
        return False, (f"main loop has not ticked for {age:.0f}s "
                       f"(limit {max_age:.0f}s) — detector is wedged")

    cameras = data.get("cameras") or {}
    if not cameras:
        return False, "heartbeat lists no cameras"

    stale = [name for name, info in cameras.items()
             if isinstance(info, dict) and info.get("stale")]
    if len(stale) == len(cameras):
        return False, (f"all {len(cameras)} camera(s) stale: "
                       f"{', '.join(sorted(stale))} — no usable frames")

    if stale:
        return True, (f"degraded: {len(stale)}/{len(cameras)} camera(s) stale "
                      f"({', '.join(sorted(stale))}), others healthy")
    return True, f"{len(cameras)} camera(s) healthy"

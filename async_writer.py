"""async_writer.py — move blocking disk/DB writes off the detection thread.

## The problem this solves

`dogwatch.py` drives every camera serially from one thread, and the loop budget
is `1 / target_fps` shared across the whole fleet — 200ms total at the default
5fps. Inside that budget, each fired event used to perform, synchronously:

  * `cv2.imwrite()` of the **full-resolution** frame for a digging event (a JPEG
    encode plus a disk write; tens of milliseconds on a 4K frame),
  * two more `cv2.imwrite()` calls when debug capture is enabled,
  * a SQLite `INSERT` + `commit()`.

So the moment something interesting happened — precisely when frames matter most
— the detection loop stalled and started dropping frames. The snapshot publish
was already correctly deferred to a thread; this extends the same treatment to
the rest.

## Design notes

*Bounded* queue, because an unbounded one just converts a stall into unbounded
memory growth (each queued full-res frame holds a reference to its numpy array).
On overflow the newest item is dropped and counted rather than blocking the
detection loop — these writes are history and diagnostics, so a gap in them is
strictly better than dropped frames or a wedged detector. Drops are logged
loudly because a sustained drop rate means the disk cannot keep up.

One worker thread, not a pool: the work is I/O to a single disk and a single
SQLite connection, so concurrency would add contention (and `EventStore`'s lock)
without adding throughput. Serial execution also preserves write ordering, which
keeps the SQLite row and the clip file for one event consistent.

Exceptions are caught, counted, and rate-limit logged. A failing disk (full, read
only, unmounted) must not take the detector down — and since these submissions
are fire-and-forget, an exception here has nowhere to propagate to anyway.
"""
import queue
import threading
import time


class AsyncWriter:
    """Single background thread + bounded queue for fire-and-forget writes."""

    #: Log at most one drop message and one error message per this interval.
    LOG_INTERVAL_SECONDS = 60.0

    def __init__(self, name="writer", maxsize=256, log=print):
        self.name = name
        self._queue = queue.Queue(maxsize=maxsize)
        self._log = log
        self._running = True

        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.errors = 0
        self._last_drop_log = 0.0
        self._last_error_log = 0.0

        self._thread = threading.Thread(
            target=self._run, name=f"async-writer-{name}", daemon=True)
        self._thread.start()

    def submit(self, fn, *args, label=None, **kwargs):
        """Queue *fn(\\*args, \\*\\*kwargs)*. Returns False if it was dropped.

        Never blocks and never raises — the caller is the detection loop.
        """
        if not self._running:
            return False
        try:
            self._queue.put_nowait((fn, args, kwargs, label or getattr(fn, "__name__", "task")))
            self.submitted += 1
            return True
        except queue.Full:
            self.dropped += 1
            now = time.time()
            if now - self._last_drop_log > self.LOG_INTERVAL_SECONDS:
                self._last_drop_log = now
                self._log(
                    f"[{self.name}] write queue FULL (depth {self._queue.maxsize}) — "
                    f"dropped {label or 'task'}; {self.dropped} dropped so far. "
                    f"The disk is not keeping up with the event rate.")
            return False

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:  # shutdown sentinel
                self._queue.task_done()
                return
            fn, args, kwargs, label = item
            try:
                fn(*args, **kwargs)
                self.completed += 1
            except Exception as exc:
                self.errors += 1
                now = time.time()
                if now - self._last_error_log > self.LOG_INTERVAL_SECONDS:
                    self._last_error_log = now
                    self._log(f"[{self.name}] write task {label!r} failed: "
                              f"{type(exc).__name__}: {exc} "
                              f"({self.errors} errors so far)")
            finally:
                self._queue.task_done()

    @property
    def depth(self):
        return self._queue.qsize()

    def stats(self):
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "dropped": self.dropped,
            "errors": self.errors,
            "depth": self.depth,
        }

    def flush(self, timeout=None):
        """Block until the queue drains. Intended for tests and shutdown."""
        if timeout is None:
            self._queue.join()
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    def stop(self, timeout=5.0):
        """Stop accepting work, drain what's queued, and join the thread."""
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Queue is saturated; drain it so the sentinel fits.
            try:
                while True:
                    self._queue.get_nowait()
                    self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=timeout)

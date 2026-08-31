"""frame_grabber.py — background RTSP reader that always holds only the newest frame.

RTSP's classic footgun: if you read frames in your processing loop you fall
behind the stream's buffer and end up analysing stale video. FrameGrabber
keeps the latest frame in a background thread and lets the main loop sample
it at whatever rate it likes.

Every stored frame is **timestamped at capture**. That timestamp is the only
way for the consumer to tell "the scene is perfectly static" apart from "this
reader died and I'm re-analysing the same pixels forever" — the two are
otherwise byte-identical, and the latter used to be completely silent. See
``frame_age()`` / ``is_stale()`` and CameraPipeline.tick's staleness gate.

Supports two decode backends:
  - CPU (default): uses cv2.VideoCapture with the FFmpeg backend.
  - GPU (opt-in via config "gpu_decode": true): uses cv2.cudacodec.VideoReader
    which offloads H.264/HEVC decode to the GPU's NVDEC hardware engine.
    Requires the CUDA-enabled OpenCV build (see Dockerfile.gpu).
"""
import threading
import time

import cv2

from redact import redact


def _has_cudacodec():
    """Check if cv2.cudacodec is available at runtime."""
    return hasattr(cv2, "cudacodec")


class FrameGrabber:
    """Background reader that always holds only the newest frame."""

    # Reconnect backoff: start at reconnect_delay, double up to this ceiling.
    # A camera that is off (unplugged, rebooting, decommissioned) previously
    # got hammered with a fresh VideoCapture every 0.5s forever.
    MAX_RECONNECT_DELAY = 30.0

    def __init__(self, url, reconnect_delay=0.5, target_fps=5, gpu_decode=False,
                 name="camera"):
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.min_interval = 1.0 / max(1.0, target_fps * 2)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_ts = None          # wall-clock time the frame was decoded
        self.running = True
        self.ready = threading.Event()

        # Diagnostics — read by CameraPipeline for logging/metrics. Guarded by
        # the same lock as the frame itself.
        self.reconnects = 0
        self.consecutive_failures = 0
        self.thread_alive = True
        self.last_error = None

        self._gpu_decode = gpu_decode and _has_cudacodec()
        if gpu_decode and not _has_cudacodec():
            print(f"[{self.name}] gpu_decode requested but cv2.cudacodec not available — falling back to CPU")

        self.cap = None
        self._reader = None
        if self._gpu_decode:
            self._reader = self._open_gpu_reader()
        else:
            self.cap = self._open_capture()

        self._thread = None
        self._start_thread()

    def _start_thread(self):
        """Start the decode thread. Overridable so tests can suppress it."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # --- backend construction (separate methods so tests can stub them) ---

    def _open_capture(self):
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _open_gpu_reader(self):
        params = cv2.cudacodec.VideoReaderInitParams()
        params.udpSource = False  # force TCP
        reader = cv2.cudacodec.createVideoReader(self.url, params=params)
        # Request BGR output to avoid per-frame BGRA→BGR CPU conversion.
        # This uses the set() API which is available in OpenCV >= 4.8.
        try:
            reader.set(cv2.cudacodec.COLOR_FORMAT_BGR)
        except (AttributeError, TypeError, cv2.error):
            try:
                # Alternative API form
                reader.setColorFormat(cv2.cudacodec.ColorFormat_BGR)
            except (AttributeError, TypeError, cv2.error):
                pass  # fallback: BGRA→BGR conversion in loop
        return reader

    def _release_gpu_reader(self):
        """Drop the current NVDEC reader before replacing it.

        cudacodec readers hold a hardware decode session; reassigning
        ``self._reader`` without releasing leaked one session per reconnect,
        so a flapping camera would eventually exhaust NVDEC.
        """
        reader, self._reader = self._reader, None
        if reader is None:
            return
        for meth in ("release", "close"):
            fn = getattr(reader, meth, None)
            if fn is None:
                continue
            try:
                fn()
                return
            except Exception:
                pass
        # No explicit release API in this OpenCV build — drop the reference
        # and let the destructor reclaim the session.
        del reader

    # --- reconnect helper shared by both loops ---

    @classmethod
    def _next_delay(cls, delay):
        """Exponential backoff, capped. Pure function — no sleeping."""
        return min(max(delay, 0.0) * 2 or cls.MAX_RECONNECT_DELAY,
                   cls.MAX_RECONNECT_DELAY)

    def _note_failure(self, delay, reason):
        """Log + sleep *delay* + return the next (backed-off) delay.

        RTSP read failures were previously entirely silent: no log line, a
        fixed 0.5s retry, and no counter. Combined with OPENCV_LOG_LEVEL=FATAL
        and av_log_set_level(AV_LOG_ERROR) suppressing FFmpeg's own
        complaints, an hour-long camera outage produced zero output anywhere.
        """
        self.consecutive_failures += 1
        self.reconnects += 1
        self.last_error = reason
        # Log the first failure, then back off logarithmically so a camera
        # that is simply off doesn't flood the log at the retry rate.
        n = self.consecutive_failures
        if n == 1 or n % 20 == 0:
            print(f"[{self.name}] stream read failed ({reason}); "
                  f"reconnecting in {delay:.1f}s "
                  f"(attempt {n}, total reconnects {self.reconnects})",
                  flush=True)
        time.sleep(delay)
        return self._next_delay(delay)

    def _store(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_ts = time.time()
            self.ready.set()
        if self.consecutive_failures:
            print(f"[{self.name}] stream recovered after "
                  f"{self.consecutive_failures} failed attempt(s)", flush=True)
            self.consecutive_failures = 0
            self.last_error = None

    # --- decode loops ---

    def _loop(self):
        """Backend dispatch, wrapped so a crash is loud instead of silent.

        Previously an unexpected exception (e.g. a cv2.error out of
        ``cap.read()`` on a malformed stream) killed this daemon thread
        outright. read() then returned the same stale frame forever, which the
        consumer could not distinguish from a static scene — so the detector
        kept "successfully" analysing a frozen image indefinitely. Now the
        thread death is logged and flagged, and the timestamp on the last
        frame lets the consumer detect it regardless.
        """
        try:
            if self._gpu_decode:
                self._loop_gpu()
            else:
                self._loop_cpu()
        except Exception as exc:
            print(f"[{self.name}] FATAL: frame grabber thread died: "
                  f"{redact(exc)}", flush=True)
        finally:
            self.thread_alive = False
            print(f"[{self.name}] frame grabber thread exiting", flush=True)

    def _loop_cpu(self):
        delay = self.reconnect_delay
        while self.running:
            t0 = time.time()
            try:
                ok, f = self.cap.read()
            except cv2.error as exc:
                # Treat a decoder-level error the same as a read failure
                # rather than letting it kill the thread.
                ok, f = False, None
                self.last_error = redact(exc)
            if not ok:
                delay = self._note_failure(delay, self.last_error or "read returned not-ok")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = self._open_capture()
                continue
            delay = self.reconnect_delay
            self._store(f)
            dt = time.time() - t0
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)

    def _loop_gpu(self):
        delay = self.reconnect_delay
        while self.running:
            t0 = time.time()
            try:
                ok, gpu_mat = self._reader.nextFrame()
                if not ok:
                    delay = self._note_failure(delay, "nextFrame returned not-ok")
                    self._release_gpu_reader()
                    self._reader = self._open_gpu_reader()
                    continue
                # cudacodec outputs BGRA from NVDEC unless the BGR colour
                # format was accepted above; drop alpha if it's still there.
                f = gpu_mat.download()
                if len(f.shape) == 3 and f.shape[2] == 4:
                    f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
                elif len(f.shape) == 2 or (len(f.shape) == 3 and f.shape[2] == 1):
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            except Exception as exc:
                delay = self._note_failure(delay, redact(exc))
                try:
                    self._release_gpu_reader()
                    self._reader = self._open_gpu_reader()
                except Exception:
                    pass
                continue

            delay = self.reconnect_delay
            self._store(f)
            dt = time.time() - t0
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)

    # --- consumer API ---

    def read(self):
        """Return a copy of the newest frame, or None if none yet."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def read_with_ts(self):
        """Return ``(frame_copy_or_None, capture_ts_or_None)`` atomically.

        Preferred over read() when the caller cares about staleness, since it
        guarantees the frame and its timestamp come from the same store.
        """
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.frame_ts

    def frame_age(self, now=None):
        """Seconds since the newest frame was decoded.

        Returns ``float('inf')`` when no frame has ever arrived, so callers
        can treat "never started" and "long dead" with the same comparison.
        """
        with self.lock:
            ts = self.frame_ts
        if ts is None:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - ts)

    def is_stale(self, max_age, now=None):
        """True if the newest frame is older than *max_age* seconds."""
        if not max_age or max_age <= 0:
            return False
        return self.frame_age(now) > max_age

    def health(self):
        """Small dict of diagnostics for logging."""
        with self.lock:
            ts = self.frame_ts
        return {
            "frame_age": self.frame_age(),
            "has_frame": ts is not None,
            "thread_alive": self.thread_alive,
            "reconnects": self.reconnects,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }

    def stop(self):
        """Stop the reader thread and release the decode backend.

        ``running`` was previously set True and never cleared — there was no
        way to shut a grabber down and no release on exit.
        """
        self.running = False
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._gpu_decode:
            self._release_gpu_reader()
        elif self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

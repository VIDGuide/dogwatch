"""frame_grabber.py — background RTSP reader that always holds only the newest frame.

RTSP's classic footgun: if you read frames in your processing loop you fall
behind the stream's buffer and end up analysing stale video. FrameGrabber
keeps the latest frame in a background thread and lets the main loop sample
it at whatever rate it likes.

Frame quality: the background loop validates each decoded frame against
snapshot_quality.is_image_bad() — grey/corrupt frames from mid-GOP connects
or HEVC decode glitches are silently discarded rather than stored as the
"latest" frame. This means the main loop's .read() always returns either
None (no good frame yet) or a frame that passed the same quality checks
the snapshot pipeline uses, eliminating the inconsistency where snapshots
used -skip_frame nokey while detection used a naive .read().
"""
import threading
import time

import cv2

from snapshot_quality import is_image_bad


class FrameGrabber:
    """Background reader that always holds only the newest *valid* frame."""

    def __init__(self, url, reconnect_delay=0.5, target_fps=5):
        self.url = url
        self.reconnect_delay = reconnect_delay
        # Throttle decode to ~2x the detection fps.  Without this the grabber
        # decodes every camera at its full native frame rate 24/7 (the main
        # CPU hog), even though the detection loop only samples a few fps.
        self.min_interval = 1.0 / max(1.0, target_fps * 2)
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.ready = threading.Event()
        # Stats for monitoring/debugging
        self._good_count = 0
        self._bad_count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            t0 = time.time()
            ok, f = self.cap.read()
            if not ok:                       # stream dropped — reconnect
                time.sleep(self.reconnect_delay)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue

            # Discard grey/corrupt frames (mid-GOP connects, HEVC decode
            # glitches).  This matches the quality gate used in the snapshot
            # pipeline (pipeline/dogwatch-notify.py) so detection never runs
            # on frames that would be rejected anyway if they appeared as a
            # snapshot — a mismatch that previously caused the detection loop
            # to process degraded/blurry/grey frames that the snapshot path
            # would have retried or discarded.
            if is_image_bad(f):
                self._bad_count += 1
                # Don't sleep full interval on a bad frame — try again quickly
                # to find the next good one (similar to the snapshot pipeline's
                # keyframe retry strategy).
                time.sleep(0.05)
                continue

            self._good_count += 1
            with self.lock:
                self.frame = f
                self.ready.set()
            # Sleep out the remainder of the frame budget so we don't spin the
            # CPU decoding frames the detector will never look at.
            dt = time.time() - t0
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    @property
    def stats(self):
        """Return (good_frames, bad_frames) counts since startup."""
        return self._good_count, self._bad_count

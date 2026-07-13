"""frame_grabber.py — background RTSP reader that always holds only the newest frame.

RTSP's classic footgun: if you read frames in your processing loop you fall
behind the stream's buffer and end up analysing stale video. FrameGrabber
keeps the latest frame in a background thread and lets the main loop sample
it at whatever rate it likes.

Frame quality validation is NOT done here — it was briefly tried but the
per-frame is_image_bad() call on high-resolution streams (2592x1944) burned
excessive CPU in the tight decode loop. Quality checks belong in tick()
(which runs at detection fps, not stream fps) where the motion gate already
provides the first-pass gating.
"""
import threading
import time

import cv2


class FrameGrabber:
    """Background reader that always holds only the newest frame."""

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

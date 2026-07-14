"""frame_grabber.py — background RTSP reader that always holds only the newest frame.

RTSP's classic footgun: if you read frames in your processing loop you fall
behind the stream's buffer and end up analysing stale video. FrameGrabber
keeps the latest frame in a background thread and lets the main loop sample
it at whatever rate it likes.

Supports two decode backends:
  - CPU (default): uses cv2.VideoCapture with the FFmpeg backend.
  - GPU (opt-in via config "gpu_decode": true): uses cv2.cudacodec.VideoReader
    which offloads H.264/HEVC decode to the GPU's NVDEC hardware engine.
    Requires the CUDA-enabled OpenCV build (see Dockerfile.gpu).
"""
import threading
import time

import cv2


def _has_cudacodec():
    """Check if cv2.cudacodec is available at runtime."""
    return hasattr(cv2, "cudacodec")


class FrameGrabber:
    """Background reader that always holds only the newest frame."""

    def __init__(self, url, reconnect_delay=0.5, target_fps=5, gpu_decode=False):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.min_interval = 1.0 / max(1.0, target_fps * 2)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.ready = threading.Event()

        self._gpu_decode = gpu_decode and _has_cudacodec()
        if gpu_decode and not _has_cudacodec():
            print("[FrameGrabber] gpu_decode requested but cv2.cudacodec not available — falling back to CPU")

        if self._gpu_decode:
            # cudacodec params for RTSP: use TCP transport
            params = cv2.cudacodec.VideoReaderInitParams()
            params.udpSource = False  # force TCP
            self._reader = cv2.cudacodec.createVideoReader(url, params=params)
        else:
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        if self._gpu_decode:
            self._loop_gpu()
        else:
            self._loop_cpu()

    def _loop_cpu(self):
        while self.running:
            t0 = time.time()
            ok, f = self.cap.read()
            if not ok:
                time.sleep(self.reconnect_delay)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            with self.lock:
                self.frame = f
                self.ready.set()
            dt = time.time() - t0
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)

    def _loop_gpu(self):
        while self.running:
            t0 = time.time()
            try:
                ok, gpu_mat = self._reader.nextFrame()
                if not ok:
                    time.sleep(self.reconnect_delay)
                    # Recreate the reader on stream drop
                    params = cv2.cudacodec.VideoReaderInitParams()
                    params.udpSource = False
                    self._reader = cv2.cudacodec.createVideoReader(self.url, params=params)
                    continue
                # Download from GPU memory to CPU numpy array
                f = gpu_mat.download()
            except Exception:
                time.sleep(self.reconnect_delay)
                try:
                    params = cv2.cudacodec.VideoReaderInitParams()
                    params.udpSource = False
                    self._reader = cv2.cudacodec.createVideoReader(self.url, params=params)
                except Exception:
                    pass
                continue

            with self.lock:
                self.frame = f
                self.ready.set()
            dt = time.time() - t0
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

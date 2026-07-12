"""debug_capture.py — optional event snapshot archiving for offline diagnosis.

Separate from the "clips" digging screenshots (clip_dir, always-on, no
cleanup — those are treated as a permanent record) and separate from HA's
live MQTT snapshot (always current, no history). This module is for keeping
a *rolling window* of both the low-res detection input (what the model
actually saw, post-crop) and the high-res raw frame (what a human would want
to review) for every fired event, with automatic age-based cleanup so it
doesn't grow unbounded.

This exists because of a real investigation gap: a false-positive
dog_at_fence event needed manual SSH access to grab the retained MQTT
snapshot before it got overwritten by the next periodic still, and there
was no separate high-resolution copy to check what the detector actually
saw at full quality. See README "Debug capture" section.

Config (per-camera, all optional):
    "debug_capture_enabled": true,       # default false — opt-in
    "debug_capture_dir": "debug_captures",
    "debug_capture_retention_days": 7,   # 0/omitted = keep forever

Layout on disk (per camera, one subfolder each so multi-camera setups
don't collide or need filename disambiguation):
    <debug_capture_dir>/<camera_name>/<epoch_ts>_<track_id>_<event_type>_lowres.jpg
    <debug_capture_dir>/<camera_name>/<epoch_ts>_<track_id>_<event_type>_highres.jpg
"""
import os
import time

import cv2


class DebugCapture:
    def __init__(self, cfg, camera_name):
        self.enabled = bool(cfg.get("debug_capture_enabled", False))
        self.retention_days = cfg.get("debug_capture_retention_days", 0)
        base_dir = cfg.get("debug_capture_dir", "debug_captures")
        self.camera_name = camera_name
        self.dir = os.path.join(base_dir, camera_name)
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def save(self, etype, tid, ts, low_res_frame, high_res_frame=None):
        """Save low-res (post-crop, what the model saw) and optionally
        high-res (raw, pre-crop) copies of the frame that triggered *etype*
        for track *tid*. No-op entirely if debug capture is disabled.

        Failures are logged but never raised — this is a diagnostic aid,
        not something that should ever interrupt detection.
        """
        if not self.enabled:
            return

        stamp = int(ts)
        base = f"{stamp}_{tid}_{etype}"
        try:
            low_path = os.path.join(self.dir, f"{base}_lowres.jpg")
            cv2.imwrite(low_path, low_res_frame)
            if high_res_frame is not None:
                high_path = os.path.join(self.dir, f"{base}_highres.jpg")
                cv2.imwrite(high_path, high_res_frame)
        except Exception as exc:
            print(f"[{self.camera_name}] debug_capture: failed to save: {exc}")

    def cleanup(self):
        """Delete files older than debug_capture_retention_days.

        A retention of 0 (or unset) means keep forever \u2014 cleanup is a
        no-op in that case. Safe to call even if disabled (no-op) or if the
        directory doesn't exist yet.
        """
        if not self.enabled or not self.retention_days:
            return
        cutoff = time.time() - (self.retention_days * 86400)
        try:
            for fname in os.listdir(self.dir):
                fpath = os.path.join(self.dir, fname)
                try:
                    if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                except OSError:
                    continue
        except FileNotFoundError:
            pass

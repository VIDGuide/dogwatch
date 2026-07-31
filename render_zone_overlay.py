#!/usr/bin/env python3
"""Render the current dogwatch detection zone + detector crop on a given frame.

Zone coordinates in configs are normalized to the CROPPED detection frame
(BehaviorMonitor multiplies by post-crop w/h), so we map them back to
full-frame pixels through the crop transform:
    x_full = crop_x1 + nx * crop_w
    y_full = crop_y1 + ny * crop_h

Draws:
  - detector crop rect  (what the TPU actually sees) — cyan
  - fence zone polygon  (event gate)                 — red, semi-transparent

Usage: render_zone_overlay.py <config.json> <input.jpg> <output.jpg>
"""
import json
import sys

import cv2
import numpy as np


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def overlay(cfg, frame, name):
    h, w = frame.shape[:2]
    crop = cfg.get("crop_roi")
    if crop:
        cx1 = int(round(crop[0] * w))
        cy1 = int(round(crop[1] * h))
        cx2 = int(round(crop[2] * w))
        cy2 = int(round(crop[3] * h))
    else:
        cx1, cy1, cx2, cy2 = 0, 0, w, h

    # Map normalized zone (crop space) -> full-frame pixels
    cw, ch = cx2 - cx1, cy2 - cy1
    pts = []
    for nx, ny in cfg["fence_zone_norm"]:
        pts.append((int(cx1 + nx * cw), int(cy1 + ny * ch)))
    poly = np.array([pts], dtype=np.int32)

    # Crop rect (cyan)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

    # Zone fill (semi-transparent red) + outline
    overlay_img = frame.copy()
    cv2.fillPoly(overlay_img, [poly], (0, 0, 255))
    cv2.addWeighted(overlay_img, 0.35, frame, 0.65, 0, frame)
    cv2.polylines(frame, [poly], True, (0, 0, 255), 3)

    # Labels
    cv2.putText(frame, "detector crop", (cx1 + 8, cy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    mx = int(np.mean([p[0] for p in pts]))
    my = int(np.mean([p[1] for p in pts]))
    cv2.putText(frame, "fence zone", (mx - 40, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"{name}  ({w}x{h})", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def main():
    cfg_path, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    name = "fence" if "config.json" in cfg_path else "rear-east"
    cfg = load_cfg(cfg_path)
    frame = cv2.imread(in_path)
    if frame is None:
        print(f"cannot read {in_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[{name}] frame {frame.shape[1]}x{frame.shape[0]}", flush=True)
    out = overlay(cfg, frame, name)
    cv2.imwrite(out_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

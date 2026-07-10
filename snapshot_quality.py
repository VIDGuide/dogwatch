"""snapshot_quality.py — detect grey/static/partially-decoded corrupt frames.

Hikvision NVRs sometimes return a JPEG from the /picture endpoint mid-GOP,
before the decoder has an I-frame reference. The result is a grey-tinted
frame (mean ~128, very low variance) that looks like decoder noise with
faint dog fragments. These heuristics catch both a fully flat glitch and a
*partial* decode (grey field + a localized pixelated "motion" blob).
"""
import cv2


def active_tile_fraction(gray, tiles=8, tile_std_thresh=15.0):
    """Fraction of NxN tiles that contain real spatial structure.

    A valid scene has detail spread across almost all tiles; a flat grey
    glitch has ~none; a *partial* decode (grey field + a localized
    pixelated 'motion' blob) lights up only one or two tiles.  This is the
    signal a global std/edge metric misses — a sharp localized blob barely
    moves a whole-frame number, but it can only ever occupy a couple of
    tiles, so it can never look like a real frame here.

    Measured: pure grey ~0.00, grey+blob ~0.06, real scene ~0.95.
    """
    h, w = gray.shape[:2]
    th, tw = h // tiles, w // tiles
    if th == 0 or tw == 0:
        return 1.0  # too small to tile — don't reject on this basis
    active = 0
    total = 0
    for ty in range(tiles):
        for tx in range(tiles):
            tile = gray[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
            if tile.size and tile.std() >= tile_std_thresh:
                active += 1
            total += 1
    return active / total if total else 1.0


def is_image_bad(img):
    """Check if a decoded image is a grey/static/corrupted frame.

    Empirically measured on these cameras:
      * genuine grey decode glitch:  mean~128, std ~1-3
      * grey + localized motion blob: mean~128, std ~8 (partial decode)
      * valid overcast/cloudy scene: mean~134, std ~37
      * valid keyframe (normal):     mean~136, std ~55
    The old `std < 40` gate rejected perfectly good overcast frames
    (a common false positive that forced needless RTSP fallbacks).
    A real glitch has almost no spatial variance, so gate on std < 12
    — which cleanly separates the glitch (<=8) from real scenes (>=37).
    """
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    mean_v, std_v = mean[0][0], stddev[0][0]
    # Pure black/white / dead frames are always suspect.
    if std_v < 8:
        return True
    # Grey/static decode glitch: mid-grey with almost no variation.
    if 105 < mean_v < 150 and std_v < 12:
        return True
    # Spatial-spread backstop for *partial* decodes: a grey frame with a
    # localized pixelated blob can lift global std past the gate above yet
    # still have structure in only a couple of tiles.  Only applied in the
    # mid-grey mean band where these glitches live, so it cannot false-
    # positive on legitimately dark (low-mean) night frames.
    if 105 < mean_v < 150 and active_tile_fraction(gray) < 0.20:
        return True
    return False

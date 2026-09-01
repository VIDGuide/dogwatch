"""image_quality.py — grey / partial-decode frame rejection (PIL, no numpy).

Host-side twin of the container's ``snapshot_quality.py``. Same three-layer
design and the same measured thresholds, implemented with PIL because the
notifier image ships Pillow rather than OpenCV.

This module exists because the logic was previously private to
``dogwatch-notify.py`` (as ``_validate_image``) while ``dogwatch-check.sh``
had its own capture path that only checked ``size > 1000`` and skipped the
pixel-content checks entirely. That gap had teeth: a grey/partial-decode
frame — exactly what these heuristics exist to catch — was fed to the vision
model, which correctly answered NO_DOG, producing a "❌ False alarm" message
for a *real* dog and suppressing the siren. Both callers now share this.
"""
import warnings

from PIL import Image, ImageStat

__all__ = ["active_tile_fraction", "is_image_bad", "validate_image_file",
           "MAX_IMAGE_PIXELS"]

# Explicit decompression-bomb ceiling.
#
# Pillow's own default (~179M pixels) only emits a *warning* — it does not
# raise until twice that — and a warning goes nowhere useful here, because
# every caller wraps Image.open in `except Exception` and just reports "cannot
# decode". So a hostile or malfunctioning JPEG could have us allocate
# gigabytes decoding a frame we were about to reject anyway.
#
# 40M pixels is ~8000x5000: an order of magnitude above the largest stream
# this project reads (a 4K frame is ~8.3M), so it constrains nothing real.
# Pillow raises DecompressionBombError above 2x this value and
# DecompressionBombWarning above it; we want the hard failure, so callers that
# open untrusted files should treat the warning as an error too — see
# validate_image_file.
MAX_IMAGE_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Layer 1: size floor.
#
# Deliberately low: it only needs to separate genuinely truncated/corrupt
# files (observed as low as 35 bytes) from *any* legitimate JPEG, not from a
# specific camera's typical size. A floor tuned to one camera (previously
# 50KB, sized for the rear-east main stream's ~300KB frames) silently
# rejected every capture from the lower-res fence sub-stream (~28KB) as
# "corruption", and the pixel checks below never even ran.
MIN_BYTES = 15_000

# Layer 2/3 thresholds — see snapshot_quality.py for the measurements.
FLAT_STD = 8.0
GREY_MEAN_LO, GREY_MEAN_HI = 105.0, 150.0
GREY_STD = 12.0
MIN_ACTIVE_TILE_FRACTION = 0.20
TILE_STD_THRESH = 15.0


def active_tile_fraction(gray_img, tiles=8, tile_std_thresh=TILE_STD_THRESH):
    """Fraction of NxN tiles containing real spatial structure.

    A valid scene has detail in almost all tiles (~0.95); a flat grey glitch
    ~0.00; a partial decode with a localized pixelated blob only ~0.06.
    Catches the "grey + a bit of dog" case a whole-frame std metric misses.
    """
    w, h = gray_img.size
    tw, th = w // tiles, h // tiles
    if tw == 0 or th == 0:
        return 1.0  # too small to tile — don't reject on this basis
    active = 0
    total = 0
    for ty in range(tiles):
        for tx in range(tiles):
            tile = gray_img.crop((tx * tw, ty * th, (tx + 1) * tw, (ty + 1) * th))
            if ImageStat.Stat(tile).stddev[0] >= tile_std_thresh:
                active += 1
            total += 1
    return active / total if total else 1.0


def is_image_bad(gray_img):
    """Return ``(bad, reason)`` for an already-greyscale PIL image."""
    st = ImageStat.Stat(gray_img)
    mean_v, std_v = st.mean[0], st.stddev[0]
    if std_v < FLAT_STD:
        return True, f"flat frame (std={std_v:.1f})"
    if GREY_MEAN_LO < mean_v < GREY_MEAN_HI:
        if std_v < GREY_STD:
            return True, f"grey glitch (mean={mean_v:.0f} std={std_v:.1f})"
        frac = active_tile_fraction(gray_img)
        if frac < MIN_ACTIVE_TILE_FRACTION:
            return True, (f"partial decode (mean={mean_v:.0f} "
                          f"std={std_v:.1f} active_tiles={frac:.2f})")
    return False, ""


def validate_image_file(path, min_bytes=MIN_BYTES, log=None):
    """True if *path* is a real frame rather than grey/corrupt output.

    *log* is an optional ``callable(str)`` for the rejection reason; pass
    ``print`` to keep the previous logging behaviour.
    """
    import os

    def _say(msg):
        if log is not None:
            log(msg)

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        _say(f"  Snapshot rejected: cannot stat ({exc})")
        return False
    if size < min_bytes:
        _say(f"  Snapshot rejected: {size} bytes < {min_bytes} min (likely corruption)")
        return False
    try:
        # Promote Pillow's DecompressionBombWarning to an exception for the
        # duration of the decode: by default an oversized image only warns
        # (and the warning would be swallowed), so the allocation happens
        # anyway. Scoped with catch_warnings so we don't alter global state.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            gray = Image.open(path).convert("L")
    except Exception as exc:
        _say(f"  Snapshot rejected: cannot decode ({exc})")
        return False
    bad, reason = is_image_bad(gray)
    if bad:
        _say(f"  Snapshot rejected: {reason}")
        return False
    return True

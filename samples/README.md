# Sample images for on-hardware detection testing

Real captures from past dog-at-fence events, used by
`tests/hardware_smoke_test.py` to verify the detector actually works on the
real Coral TPU + real model, independent of whether a dog happens to be in
frame right now. See that script's docstring for how to run it.

None of these are synthetic — they're genuine alert photos (rear-east) and
one snapshot grabbed live from the fence camera during an active sighting.

## Ground truth (confirmed via Gemini vision + manual review)

| File | Camera | Dog present | Dog size (frame %) | Notes |
|------|--------|-------------|---------------------|-------|
| `rear-east-yes-1.jpg` | rear-east | Yes | ~10-12% width | Full uncropped 2560x1920 frame. Weak but present detection (~0.20) at score_threshold 0.4 — below the production threshold. |
| `rear-east-yes-2.jpg` | rear-east | Yes | small, lower-left | Full uncropped 2560x1920 frame. **Zero detections** at the full frame; recovers strongly (0.80) when cropped to the region containing the dog. |
| `rear-east-yes-3.jpg` | rear-east | Yes | small, bottom-left | Same pattern as #2 — zero at full frame, 0.72-0.80 when cropped. |
| `rear-east-yes-4.jpg` | rear-east | Yes | ~10% width, middle-left | Zero at full frame; weak recovery (0.07) even when cropped — likely partial occlusion/motion blur on top of small size. |
| `fence-camera-yes-1.jpg` | fence (camera) | Yes | ~3-5% width | Native 640x480 fence sub-stream resolution. **Zero detections at any crop tested** — the dog is only ~20-30px wide in absolute terms, likely below what this model can reliably resolve regardless of framing. |

## Why these all show weak/zero detections despite a real dog being present

This is **not** a regression from the `ai-edge-litert` migration (PR #2).
Confirmed via a direct A/B test: running the exact same 5 images through
both the new `ai-edge-litert` detector and the old `pycoral`/`tflite_runtime`
stack on the real Coral TPU produced identical results — same detections,
same scores, same bounding boxes, down to the pixel. Both stacks fail on
the same images for the same reason.

The actual cause is `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite`
struggling with small objects in wide-angle, uncropped frames: the model's
input tensor is a fixed 300x300, so a dog occupying ~10% of a 2560x1920
frame shrinks to roughly 30x30 pixels once resized down to fit — right at
or below this architecture's reliable detection floor. This is a known,
long-standing limitation of this exact model, not something introduced
recently.

**What actually fixes it:** cropping to a tighter region of interest before
detection — exactly what `crop_roi` in the camera config is for (already
used on `rear-east`, not currently set on the fence `camera` config). A
tighter crop makes the same dog occupy a much larger fraction of the model's
300x300 input, which is why cropping rear-east-yes-2/3 from full-frame to
the region containing the dog took detection confidence from 0.00 to
0.72-0.80 in testing.

## Re-running this comparison

If a future dependency bump or model swap is suspected of hurting accuracy,
repeat the A/B test methodology used to produce the table above: build a
throwaway image on the old dependency stack, run the same sample images
through it, and diff the results against the current stack. Don't assume a
detection miss is a regression without this kind of side-by-side check —
it's easy to mistake "this model was always weak at long range" for
"something broke."

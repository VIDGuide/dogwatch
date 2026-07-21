# DogWatch — Coral TPU Dog Detector

[![CI](https://github.com/VIDGuide/dogwatch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VIDGuide/dogwatch/actions/workflows/ci.yml)

Real-time dog-at-fence and digging detection using a Google Coral Edge TPU and
one or more RTSP cameras. Publishes events and annotated snapshots to Home
Assistant via MQTT.

## Features

- **Multi-camera** — runs any number of cameras in a single container
- **Coral TPU** — EfficientDet-Lite3 on the Edge TPU (512×512 input) for low-power, high-accuracy inference
- **Fence zone** — define a polygon per camera; dogs trigger only when their
  paws cross it
- **Digging heuristic** — stationary bounding box + high intra-box pixel change
- **HA auto-discovery** — registers binary sensors and camera entities via MQTT
- **Annotated snapshots** — publishes cropped, labelled JPEGs to the snapshot topic

## Requirements

- Linux with a Coral Edge TPU (PCIe M.2 or USB)
- Coral Edge TPU runtime (`libedgetpu1-std`) — Google's official builds are
  abandoned, so the `Dockerfile` pulls a community-maintained build from
  [`feranick/libedgetpu`](https://github.com/feranick/libedgetpu) instead;
  see "Known limitations" below
- One or more RTSP cameras
- MQTT broker (Mosquitto, Home Assistant add-on, etc.)

## Quick Start

1. **Clone & configure**
   ```bash
   git clone https://github.com/VIDGuide/dogwatch.git
   cd dogwatch
   cp config.example.json config.json
   # Edit config.json with your RTSP URL, MQTT host, fence zone
   ```

2. **Download the model**

   Model and labels come from Google's official [`google-coral/test_data`](https://github.com/google-coral/test_data)
   repo:
   ```bash
   mkdir -p models
   curl -L -o models/efficientdet_lite3_512_ptq_edgetpu.tflite \
     https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite3_512_ptq_edgetpu.tflite
   curl -L -o models/coco_labels.txt \
     https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
   ```
   This is the stock COCO-trained EfficientDet-Lite3 model, already compiled
   for the Edge TPU — no training or conversion needed. The input resolution
   is **512×512** (vs 300×300 for the older MobileNet V2 models), giving
   substantially better detection of small/distant dogs. It detects all 90
   COCO classes; `detector.py` filters to just `dog` at runtime by looking
   up the label id in `coco_labels.txt`.

   The model path is **config-driven**: `detector.py` reads the input shape
   from the model file at load time, so swapping to a different
   Edge-TPU-compiled model (e.g. back to MobileNet V2, or a fine-tuned
   variant) is just a config change — no code edits needed. Available
   alternatives in the same repo include `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite`
   (300×300, faster, lower accuracy) and `ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite`
   (320×320, QAT-trained, middle ground).

3. **Run**
   ```bash
   docker compose up -d
   ```

## Config

Each camera needs its own `config-<name>.json`. See `config.example.json` and
`config-rear-east.example.json` for the full schema.

| Key | Description |
|-----|-------------|
| `rtsp_url` | RTSP stream URL |
| `score_threshold` | Minimum detection confidence (0-1) required to fire an event. Default 0.4. Raise this if you're seeing false positives (fence posts, shadows, soil texture misidentified as a dog) — see "Known limitations" below for a documented example. Each event's `attributes` MQTT payload now includes the actual detection `score`, so you can check how confident a specific false positive was before deciding how far to raise this. |
| `snapshot_url` | (Optional) HTTP snapshot URL for clean stills |
| `crop_roi` | (Optional) `[x1, y1, x2, y2]` normalised 0-1 — zoom into part of frame. Strongly recommended if the camera's full field of view is much wider than the actual fence/zone area: the detection model's input resolution (512×512 for EfficientDet-Lite3, 300×300 for older models) can struggle with small/distant dogs in a wide uncropped frame — see `samples/README.md` for measured evidence. Not currently set for the fence `camera` config, which is the most likely cause of missed detections on that camera specifically. |
| `fence_zone_norm` | Polygon vertices `[[x,y], ...]` normalised 0-1 |
| `stationary_px` | Max centroid drift (px) to consider dog "stationary" |
| `motion_energy_thresh` | Fraction of box pixels changing per frame (0-1) |
| `dig_sustain_seconds` | Seconds of continuous motion before "digging" fires |
| `dig_stationary_px` | Max drift (px) allowed while "digging" (looser than `stationary_px`; a digging dog shuffles in place). Defaults to `2 x stationary_px` |
| `event_cooldown_seconds` | Min seconds between repeated events |
| `off_delay_seconds` | HA `off_delay` for the binary sensors — auto-reverts to OFF this long after the last ON, even if our OFF message is lost (fixes sensors sticking triggered). Default 180 |
| `min_consecutive` | Consecutive detections required before firing events |
| `startup_timeout_seconds` | Max seconds to wait for the first camera frame before failing loudly (non-zero exit) instead of hanging forever. Default 60 |
| `mqtt_username` / `mqtt_password` | (Optional) MQTT broker credentials. Can also be set via the `MQTT_USERNAME` / `MQTT_PASSWORD` env vars |
| `mqtt_tls` | (Optional) Enable TLS for the MQTT connection. Default `false` |
| `debug_capture_enabled` | (Optional) Archive a low-res + high-res snapshot of every fired event to `debug_capture_dir` for offline review. Default `false`. See "Debug capture" below |
| `debug_capture_dir` | (Optional) Where to write archived debug snapshots. Default `debug_captures` (mounted as a volume in `docker-compose.yml` regardless of whether capture is enabled, so turning it on doesn't need a compose edit) |
| `debug_capture_retention_days` | (Optional) Delete archived debug snapshots older than this many days. `0` (default) keeps everything forever — set a real value to bound disk usage |
| `target_fps` | Detection sample rate. The frame grabber decodes at `2 × target_fps`. Default 5 (= 10 decode/s). For high-res main streams (>1080p), use 2–3 to keep CPU decode cost reasonable. Dogs move slowly enough that 2fps is fine for detection cadence. |
| `tracker_max_distance` | (Optional) Max pixel distance between centroids to match a detection to an existing track. Default 120. Scale up for high-res crops where dogs traverse more pixels per frame at the same real-world speed. |
| `tracker_max_misses` | (Optional) Frames a track can go unmatched before deletion. Default 5. |
| `event_store_enabled` | (Optional) Log events to a SQLite database. Default `true`. |
| `event_store_path` | (Optional) Path to the SQLite event database. Default `data/events.db`. |
| `motion_gate_enabled` | (Optional) Skip TPU inference when nothing is moving. Default `true`. Eliminates false positives from static structural elements (beams, railings). |
| `motion_gate_threshold` | (Optional) Fraction of pixels that must change to trigger detection. Default 0.005 (0.5%). |
| `motion_gate_pixel_threshold` | (Optional) Per-pixel abs-diff floor for noise filtering. Default 25. |
| `motion_gate_max_idle_seconds` | (Optional) Force a detection pass at least this often even if no motion, so a dog that walks in and stops isn't missed. Default 10. |
| `gpu_decode` | (Optional) Offload RTSP frame decode to GPU via NVDEC. Default `false`. Requires `Dockerfile.gpu` / `docker-compose.gpu.yml` and an NVIDIA GPU. See "Performance tuning → GPU-accelerated decode" above. |

**MQTT security note:** by default the broker connection is plaintext and
unauthenticated, which is fine for a broker that never leaves
localhost/a trusted LAN. If your broker is reachable beyond that (a
different host, a VPN, etc.), set `mqtt_username`/`mqtt_password` and
`mqtt_tls: true`.

Set `DOGWATCH_DEBUG=1` in the container environment to log the per-frame
digging sub-signals (`stationary`, `motion` fraction, held time) so the digging
thresholds can be tuned against real footage.

## Debug capture

Off by default. When you need to diagnose a specific miss or false positive
(see the false-positive example in "Known limitations") it helps to have
the actual frames on disk rather than relying on whatever happens to still
be retained on MQTT or in `/tmp` at the time — this was a real gap during a
past investigation, where a false-positive snapshot had to be grabbed via
SSH before the next periodic still overwrote it, and there was no separate
high-resolution copy of what the detector actually saw.

**Container side** (`camera_pipeline.py` / `debug_capture.py`): on every
fired event (`dog_at_fence` or `digging`), if `debug_capture_enabled` is set
in that camera's config, saves two files to
`debug_captures/<camera>/<epoch_ts>_<track_id>_<event_type>_{lowres,highres}.jpg`:
- `lowres` — the post-crop ROI exactly as fed into the detection model
- `highres` — the full raw frame, uncropped

Old files are swept once an hour if `debug_capture_retention_days` is set
(0/unset keeps everything forever).

**Notifier side** (`pipeline/dogwatch-notify.py`): controlled by env vars
rather than the camera config JSON, since this script runs outside the
container:

| Env var | Default | Description |
|---------|---------|--------------|
| `DOGWATCH_DEBUG_CAPTURE` | unset (off) | Set to `1`/`true`/`yes` to archive the annotated (bbox-drawn) snapshot the notifier sends to Telegram/HA |
| `DOGWATCH_DEBUG_CAPTURE_DIR` | `debug_captures` | Archive directory (per-camera subfolders, same layout as the container side) |
| `DOGWATCH_DEBUG_CAPTURE_RETENTION_DAYS` | `0` (forever) | Delete archived files older than this many days; swept once an hour |

This also fixes an unrelated leak found during the same investigation:
`dogwatch-check.sh`'s cron job only ever *copies* the notifier's `/tmp`
event snapshots into its own workspace directory — it never deleted the
`/tmp` originals, so they accumulated indefinitely (70+ had built up over a
few days on the actual deployment). The notifier now always removes its own
`/tmp` snapshot ~10 minutes after writing it (comfortably past
`dogwatch-check.sh`'s ~5 minute cron lookback window), regardless of
whether debug capture is enabled.

**Batch-labeling archived captures:** `tests/gemini_batch_label.py` runs a
directory (or specific file list) of archived snapshots through Gemini
vision and writes a CSV (`path,dog,confidence,notes`) — useful for turning
a pile of past events into rough validation data (how many fired events
were real dogs vs false positives, and why) without reviewing each image
by hand:
```bash
python tests/gemini_batch_label.py --dir debug_captures/rear-east --sample 20 --out labels.csv
```
Subject to the Gemini free tier's daily request quota (resets at midnight
Pacific time) — the script retries on rate-limit errors with backoff, but
if the whole day's quota is exhausted, it'll just error out per-image
until the quota resets.

## Notification pipeline (`pipeline/`)

The Coral detector only publishes MQTT. The alerting/verification layer lives in
`pipeline/` and runs outside the container:

| File | Runs as | Role |
|------|---------|------|
| `dogwatch-notify.py` | systemd user service (`dogwatch-notify.service`) | Subscribes to MQTT, republishes annotated snapshots to HA, keeps a periodic live still (60s), writes an event log |
| `dogwatch-check.sh` | cron `*/5 * * * *` | Reads the event log, sends a Telegram ping, runs vision model verification (dog presence **and** digging), sends confirm/false-alarm follow-ups |
| `dogwatch-notify.config.example.json` | — | Template for the camera registry + Telegram chat id used by the notifier |

See **[`pipeline/home-assistant-example.md`](pipeline/home-assistant-example.md)**
for the Home Assistant side: the auto-discovered entities, optional snapshot-
timestamp sensors, and the Lovelace dashboard cards (pulsing boolean status
tiles + camera snapshots) taken from a working dashboard.

**Secrets:** the notifier reads its camera URLs and chat id from
`pipeline/dogwatch-notify.config.json` (gitignored — copy the `.example`).
The Telegram bot token and vision API key are read at runtime from
`~/.openclaw/secrets.json`. No credentials are committed. Since this file
holds live API tokens, lock it down to your user only:
```bash
chmod 600 ~/.openclaw/secrets.json
```
`dogwatch-check.sh` uses `${DOGWATCH_WORKSPACE_DIR:-$HOME/.openclaw/workspace/dogwatch_snaps}`
for its workspace snapshot dir (override with `DOGWATCH_WORKSPACE_DIR` if you
deploy elsewhere), and relies on GNU `date` (`date -d`), so it targets
Linux cron/systemd hosts — it will not run as-is on macOS/BSD.

The pipeline scripts (`dogwatch-notify.py`, `dogwatch-check.sh`) run outside
the Docker image, directly on the host under a plain Python venv (any
current Python 3 — there's no version constraint here, unlike the detector
container). Install their dependencies with:
```bash
pip install -r pipeline/requirements.txt
```

### Vision model (model-agnostic)

`dogwatch-check.sh` calls the vision model through the [OpenAI-compatible
chat completions format](https://ai.google.dev/gemini-api/docs/openai), so
any provider that speaks this API can be used instead of Gemini — swap in
OpenAI, a local Ollama/vLLM server, or another hosted provider without
touching the code. Configure it with env vars (e.g. in the cron
environment or a wrapper script):

| Env var | Default | Description |
|---------|---------|--------------|
| `DOGWATCH_VISION_API_URL` | Gemini's OpenAI-compatible endpoint | Chat completions endpoint URL |
| `DOGWATCH_VISION_MODEL` | `gemini-3-flash-preview` | Model name to request |
| `DOGWATCH_VISION_API_KEY` | (falls back to `secrets.json`) | API key, sent as a `Bearer` token |

Gemini is the default because its free tier is generous for this usage
pattern (a handful of image calls every few minutes), but the pin is a
convenience default, not a hard dependency. If `DOGWATCH_VISION_API_KEY` is
unset, the script falls back to `models.providers.google.apiKey` in
`~/.openclaw/secrets.json` for backwards compatibility with existing
Gemini-only setups.

## Development

Unit tests cover `tracker.py`, `behavior.py`, and `snapshot_quality.py` (the
parts with real logic, as opposed to I/O glue). They run on plain Python —
no Coral hardware or camera feed needed.

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs on every push/PR to `main`: unit tests,
a `py_compile` syntax check across all modules, a `bash -n` check on the
`pipeline/*.sh` scripts, and a full `linux/amd64` Docker image build (no
Coral hardware available in CI, so this only verifies the image builds and
installs cleanly — not that inference actually works).

### On-hardware detection smoke test

`tests/hardware_smoke_test.py` runs the real `DogDetector` against the real
Coral Edge TPU using known-good sample images in `samples/` (real past
detections, not synthetic test data — see `samples/README.md` for what each
one is and its measured baseline score). This exists specifically to check
whether a dependency, model, or runtime change silently hurt detection
accuracy, without needing to wait for a real dog to walk into frame.

It's not part of the pytest suite or CI — it needs the physical TPU device,
so it only runs on the deployment host, with the main `dogwatch` container
stopped first (only one process can hold the Edge TPU delegate at a time):

```bash
docker stop dogwatch
docker run --rm --device /dev/apex_0:/dev/apex_0 \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/samples:/app/samples:ro" \
  -v "$(pwd)/tests/hardware_smoke_test.py:/app/hardware_smoke_test.py" \
  dogtracker-dogwatch python /app/hardware_smoke_test.py
docker start dogwatch
```

All 5 current samples are small/distant dogs in full uncropped frames — a
known weakness of the older `ssd_mobilenet_v2`'s 300×300 input resolution
on small objects, which motivated the switch to EfficientDet-Lite3 (512×512
input). The script tracks each sample's baseline score and flags a
*regression* (a meaningful drop from that baseline) rather than just
treating "no detection" as a failure. See `samples/README.md` for the
full writeup and the cropping-based mitigation (`crop_roi`) that also helps
with small-object misses.

## Performance tuning

### CPU usage from RTSP stream decode

Video frame decoding (H.264/HEVC → raw pixels) is done by ffmpeg **on CPU**
via OpenCV's `VideoCapture` backend — not on the Coral TPU (which only handles
model inference). For high-resolution streams (e.g. a 2592×1944 main stream),
this can be a significant CPU consumer.

Levers to reduce decode CPU:

| Approach | Effort | Effect |
|----------|--------|--------|
| Lower `target_fps` | Config change | The frame grabber decodes at `2 × target_fps`. Use 2–3 for high-res streams; dogs move slowly enough that 2fps detection cadence is fine. |
| Use the sub-stream for detection, main for snapshots | Config change | Most cameras expose a low-res sub-stream (e.g. 640×480). Use it as `rtsp_url` with no `crop_roi` for cheap detection, and let the notifier use the main stream for annotated snapshot capture. |
| Motion gate (default: on) | Already active | When nothing moves, no TPU inference runs — but the frame grabber still decodes. The above two approaches reduce this baseline decode cost. |

### GPU-accelerated decode (NVIDIA)

With an NVIDIA GPU, ffmpeg can use **NVDEC** (hardware decode) to offload
H.264/HEVC decoding entirely off the CPU. Two integration paths:

1. **OpenCV `cudacodec.VideoReader`** — OpenCV's CUDA module includes a
   GPU-based video reader that uses NVDEC directly. Requires building
   OpenCV from source with `-D WITH_CUDA=ON -D WITH_NVCUVID=ON` (the pip
   `opencv-python-headless` package does NOT include this). Gives you
   decoded frames as `cv2.cuda.GpuMat` which can be downloaded to numpy.
   This is the cleanest path for this project — `FrameGrabber` would switch
   from `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` to
   `cv2.cudacodec.createVideoReader(url)`.

2. **ffmpeg with `hwaccel cuvid`** — Build ffmpeg with `--enable-cuvid
   --enable-nvdec`. OpenCV's FFmpeg backend can then use hardware decode via
   the `OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable:
   ```
   OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;cuda|video_codec;h264_cuvid|rtsp_transport;tcp"
   ```
   This requires the NVIDIA Container Toolkit (for GPU access inside Docker)
   and a custom-built ffmpeg in the container image. Less clean than option 1
   but doesn't require building OpenCV from source.

Either path reduces CPU decode cost to near-zero regardless of resolution or
fps, since the GPU's dedicated NVDEC engine handles it.

**Prerequisites:**
- NVIDIA GPU with NVDEC support (GeForce/Quadro Maxwell+, compute capability >= 5.0)
- NVIDIA driver >= 550 on the host
- NVIDIA Container Toolkit installed

**Ready to use:** `Dockerfile.gpu` and `docker-compose.gpu.yml` are provided.
They use [cudawarped's pre-built OpenCV CUDA wheels](https://github.com/cudawarped/opencv-python-cuda-wheels)
(includes `cv2.cudacodec` with NVDEC/NVCUVID) so no from-source build is needed.

```bash
# Install NVIDIA Container Toolkit (one-time, on host):
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Enable GPU decode in camera configs:
# Add "gpu_decode": true to each camera's config.json

# Build and run:
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml up -d
```

The standard `Dockerfile` / `docker-compose.yml` remain fully functional on
hardware without a GPU — `FrameGrabber` automatically falls back to CPU
decode if `cv2.cudacodec` isn't available, regardless of the `gpu_decode`
config flag.

## Known limitations

- **Coral Edge TPU support is community-maintained, not official.** Google
  has effectively abandoned the Coral software stack — `pycoral` and
  `tflite_runtime` saw no meaningful releases in years and only ever shipped
  `cp39` wheels (this project's Python 3.9 pin, and the numpy 1.x /
  opencv-python-headless 4.9.x pins it forced, were resolved by migrating
  off pycoral — see [#1](https://github.com/VIDGuide/dogwatch/issues/1) for
  that history). The detector now uses
  [`ai-edge-litert`](https://pypi.org/project/ai-edge-litert/) (Google's
  actively maintained LiteRT runtime, wheels through Python 3.14) paired
  with [`feranick/libedgetpu`](https://github.com/feranick/libedgetpu), a
  community fork that keeps the native Edge TPU driver building against
  current TensorFlow releases. This removed the structural numpy/opencv
  version ceiling — the `Dockerfile` now tracks each dependency's latest
  stable release with no known constraint forcing them behind. If
  `feranick/libedgetpu` ever goes unmaintained too, the next fallback is
  building `libedgetpu` from source (see their README) or moving off the
  Coral TPU entirely.
- `detector.py` no longer depends on `pycoral` at all — it talks to
  `ai_edge_litert.interpreter` directly (`Interpreter` + `load_delegate`),
  reimplementing the small, pure-Python pieces pycoral used to wrap (input
  tensor resizing/padding, output tensor parsing for SSD-style detection
  models). No compiled bindings are involved on the Python side anymore;
  the only native component is `libedgetpu.so` itself.
- **False positives on fence/ground geometry.** The model can occasionally
  misidentify high-contrast vertical/horizontal lines (fence
  rails, retaining wall beams) plus shadows on dirt/soil as a dog,
  especially on a low-quality/heavily-compressed frame. Confirmed via a
  real event (verified independently with Gemini vision, which found no
  identifiable canine features in the flagged region — just a wooden beam,
  dirt, and shadow). Detection events now include the actual confidence
  `score` in their MQTT `attributes` payload (previously dropped
  silently between `detector.py` and the published event), so a run of
  false positives can be checked for a common low-confidence pattern and
  used to inform raising `score_threshold` for that camera.

### Snapshot quality / grey-frame handling

These cameras use inter-frame compression (the rear-east main stream is HEVC
with a ~2 s GOP). Two mechanisms keep grey/corrupt frames out of Home
Assistant:

1. **Capture waits for a keyframe.** `capture_snapshot` uses ffmpeg
   `-skip_frame nokey` so the first decoded frame is always a self-contained
   I-frame. Grabbing "the next frame" blindly lands mid-GOP on a P/B-frame
   with no reference and renders a flat grey field (the classic "all grey" /
   "grey with a few moving pixels" snapshot).
2. **Validation rejects bad frames** (`is_image_bad` in `snapshot_quality.py`,
   `_validate_image` in the notifier), in three layers:
   - size floor (flat JPEGs are tiny),
   - global grey gate (`105 < mean < 150` and `std < 12`),
   - **spatial-spread backstop**: split into an 8×8 grid and reject if
     fewer than 20% of tiles contain real detail. This catches *partial*
     decodes — a grey field with a localized pixelated "motion" blob — that
     can push global std past the gate yet only light up one or two tiles.
     (Measured: pure grey ~0% active tiles, grey+blob ~6%, real scene ~95%.)

## License

MIT

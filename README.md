# DogWatch — Coral TPU Dog Detector

[![CI](https://github.com/VIDGuide/dogwatch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VIDGuide/dogwatch/actions/workflows/ci.yml)

Real-time dog-at-fence and digging detection using a Google Coral Edge TPU and
one or more RTSP cameras. Publishes events and annotated snapshots to Home
Assistant via MQTT.

## Features

- **Multi-camera** — runs any number of cameras in a single container
- **Coral TPU** — SSD MobileNet V2 on the Edge TPU for low-power inference
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

   Both files come from Google's official [`google-coral/test_data`](https://github.com/google-coral/test_data)
   repo:
   ```bash
   mkdir -p models
   curl -L -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
     https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
   curl -L -o models/coco_labels.txt \
     https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
   ```
   This is the stock COCO-trained SSD MobileNet V2 model, already compiled
   for the Edge TPU — no training or conversion needed. It detects all 90
   COCO classes; `detector.py` filters to just `dog` at runtime by looking
   up the label id in `coco_labels.txt`, so nothing else needs to change if
   you swap in a different (still Edge-TPU-compiled) SSD model later.

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
| `snapshot_url` | (Optional) HTTP snapshot URL for clean stills |
| `crop_roi` | (Optional) `[x1, y1, x2, y2]` normalised 0-1 — zoom into part of frame |
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

**MQTT security note:** by default the broker connection is plaintext and
unauthenticated, which is fine for a broker that never leaves
localhost/a trusted LAN. If your broker is reachable beyond that (a
different host, a VPN, etc.), set `mqtt_username`/`mqtt_password` and
`mqtt_tls: true`.

Set `DOGWATCH_DEBUG=1` in the container environment to log the per-frame
digging sub-signals (`stationary`, `motion` fraction, held time) so the digging
thresholds can be tuned against real footage.

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

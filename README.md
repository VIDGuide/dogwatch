# DogWatch — Coral TPU Dog Detector

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
- Coral runtime: `libedgetpu1-std`
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
   ```bash
   mkdir -p models
   # ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
   # coco_labels.txt
   ```

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
the Docker image, directly on the host under a plain Python 3.9 venv. Install
their dependencies with:
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

## Known limitations

- **Python 3.9 pin (structural, not easily fixable).** The whole detection
  container is pinned to Python 3.9 because `pycoral`/`tflite_runtime` are
  abandoned upstream and only ever shipped `cp39` wheels. Python 3.9 reached
  end-of-life on 2025-10-31, so this image runs on an unsupported CPython
  version by necessity, not choice. All dependency versions in the
  `Dockerfile` and `pipeline/requirements.txt` are bumped to the newest
  release that still ships a `cp39` wheel, and are re-checked whenever a CVE
  or dependency bump is made — but the underlying Python version itself can't
  be moved forward without replacing the Coral inference stack (e.g. ONNX
  Runtime + a different TPU/accelerator path).
- **Pillow capped at 11.3.0** for the same reason (last `cp39` release;
  12.x dropped Python 3.9). Known CVEs fixed in 12.1/12.2 are all in
  PSD/PDF/DDS/FITS parsing; `dogwatch-notify.py` only opens JPEGs it captures
  itself, so exposure is low but not zero if that assumption ever changes.
- **numpy capped at 1.26.4 (the last 1.x release), opencv-python-headless
  capped at 4.9.0.80.** pycoral's compiled bindings are built against the
  numpy 1.x C ABI and break at runtime under numpy 2.x — confirmed on real
  Coral TPU hardware, not just a version-range conflict pip would catch.
  This forces opencv-python-headless back from the 4.12.x line (which
  requires numpy>=2) to 4.9.0.80, the newest release still on numpy<2.
  4.9.0.80 predates CVE-2025-53644 (a heap buffer write via crafted JPEG,
  which only affects 4.10.0 and 4.11.0), so this isn't a regression to a
  known-vulnerable version — it's a deliberate skip over the two versions
  that were actually affected. Re-check this pin if pycoral ever ships a
  numpy-2.x-compatible build upstream.

### Snapshot quality / grey-frame handling

These cameras use inter-frame compression (the rear-east main stream is HEVC
with a ~2 s GOP). Two mechanisms keep grey/corrupt frames out of Home
Assistant:

1. **Capture waits for a keyframe.** `capture_snapshot` uses ffmpeg
   `-skip_frame nokey` so the first decoded frame is always a self-contained
   I-frame. Grabbing "the next frame" blindly lands mid-GOP on a P/B-frame
   with no reference and renders a flat grey field (the classic "all grey" /
   "grey with a few moving pixels" snapshot).
2. **Validation rejects bad frames** (`_is_image_bad` in the detector,
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

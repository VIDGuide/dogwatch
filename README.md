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

Set `DOGWATCH_DEBUG=1` in the container environment to log the per-frame
digging sub-signals (`stationary`, `motion` fraction, held time) so the digging
thresholds can be tuned against real footage.

## Notification pipeline (`pipeline/`)

The Coral detector only publishes MQTT. The alerting/verification layer lives in
`pipeline/` and runs outside the container:

| File | Runs as | Role |
|------|---------|------|
| `dogwatch-notify.py` | systemd user service (`dogwatch-notify.service`) | Subscribes to MQTT, republishes annotated snapshots to HA, keeps a periodic live still (60s), writes an event log |
| `dogwatch-check.sh` | cron `*/5 * * * *` | Reads the event log, sends a Telegram ping, runs **Gemini** vision verification (dog presence **and** digging), sends confirm/false-alarm follow-ups |
| `dogwatch-notify.config.example.json` | — | Template for the camera registry + Telegram chat id used by the notifier |

See **[`pipeline/home-assistant-example.md`](pipeline/home-assistant-example.md)**
for the Home Assistant side: the auto-discovered entities, optional snapshot-
timestamp sensors, and the Lovelace dashboard cards (pulsing boolean status
tiles + camera snapshots) taken from a working dashboard.

**Secrets:** the notifier reads its camera URLs and chat id from
`pipeline/dogwatch-notify.config.json` (gitignored — copy the `.example`).
The Gemini/Telegram tokens are read at runtime from `~/.openclaw/secrets.json`.
No credentials are committed. `dogwatch-check.sh` still uses absolute
`$HOME/.openclaw` paths for its workspace snapshot dir — adjust if you deploy
elsewhere.

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

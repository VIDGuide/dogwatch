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

   Use the `download_models.sh` script to pull models from Google's
   [`google-coral/test_data`](https://github.com/google-coral/test_data) repo:
   ```bash
   ./download_models.sh efficientdet   # recommended: EfficientDet-Lite3 512×512 (~38 mAP)
   ./download_models.sh mobilenet      # faster: SSD MobileNet V2 300×300
   ./download_models.sh mobiledet      # middle ground: SSDLite MobileDet 320×320
   ./download_models.sh all            # download all three + COCO labels
   ```
   This downloads a pre-compiled Edge-TPU model — no training or conversion
   needed. The labels file (`coco_labels.txt`) is always downloaded alongside.
   `detector.py` filters to just `dog` at runtime by looking up the label id.

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

**Which keys are per-camera, and which aren't.** Almost everything below is
per-camera, including `score_threshold`. The exceptions are `model_path` and
`labels_path`: only one process can bind the Edge TPU, so there is exactly one
interpreter, built from the first config loaded. If another config names a
different model, that value is ignored — the detector now logs a warning saying
so instead of ignoring it silently. On startup it also prints each camera's
effective `score_threshold`, so you can confirm what's actually in force.

`mqtt_base_topic` goes further than per-camera: it must be **unique** per
camera, and the detector refuses to start otherwise. Two cameras sharing a base
topic register two distinct pairs of Home Assistant entities that both subscribe
to the same state topic, so one camera's dog switches on the other camera's
sensor and their retained snapshots overwrite each other — with nothing visible
from the HA side to explain it. The same check rejects `MQTT_TOPIC` in the
environment whenever more than one camera is configured, since it is a single
global value that would override every camera's setting at once. (`MQTT_HOST`
and `MQTT_PORT` are genuinely global — one broker — so they are unaffected.)

| Key | Description |
|-----|-------------|
| `rtsp_url` | RTSP stream URL |
| `mqtt_host` / `mqtt_port` | Broker address. Overridable fleet-wide via the `MQTT_HOST` / `MQTT_PORT` env vars |
| `mqtt_base_topic` | Root topic for this camera's state, attribute, snapshot, geometry and availability topics. **Must be unique per camera** — see the note above. Overridable via `MQTT_TOPIC` only when a single camera is configured |
| `clip_dir` | Where digging event clips are written. Default `clips`. Give each camera its own directory (`docker-compose.yml` mounts a separate `clips-rear-east` for exactly this reason): clip retention sweeps the whole directory by age, so cameras sharing one with different `clip_retention_days` will each apply their own cutoff to the other's files. Filenames are `dig_<camera>_<epoch>_<track>.jpg`, so a shared directory is at least collision-free |
| `score_threshold` | Minimum detection confidence (0-1) required to fire an event. Default 0.4, and genuinely **per-camera** — raise it on just the camera that's producing false positives (fence posts, shadows, soil texture misidentified as a dog); see "Known limitations" for a documented example. Each event's `attributes` MQTT payload includes the actual detection `score`, so you can check how confident a specific false positive was before deciding how far to raise this. Values outside `(0, 1]` are rejected with a warning and fall back to the default — confidences are fractions, so `0.55`, not `55`. |
| `snapshot_url` | (Optional) HTTP snapshot URL for clean stills |
| `crop_roi` | (Optional) `[x1, y1, x2, y2]` normalised 0-1 — zoom into part of frame. Strongly recommended if the camera's full field of view is much wider than the actual fence/zone area: the detection model's input resolution (512×512 for EfficientDet-Lite3, 300×300 for older models) can struggle with small/distant dogs in a wide uncropped frame — see `samples/README.md` for measured evidence. Not currently set for the fence `camera` config, which is the most likely cause of missed detections on that camera specifically. |
| `fence_zone_norm` | Polygon vertices `[[x,y], ...]` normalised 0-1 **relative to the cropped frame, not the full frame**. If `crop_roi` is set, these coordinates are fractions of the crop — so changing `crop_roi` invalidates an existing zone. Use `render_zone_overlay.py` to check the zone against a real frame grab. A paw point exactly on the polygon edge counts as inside. |
| `stationary_px` | Max centroid drift (px) to consider dog "stationary" |
| `motion_energy_thresh` | Fraction of box pixels changing per frame (0-1) |
| `dig_sustain_seconds` | Seconds of continuous motion before "digging" fires |
| `dig_stationary_px` | Max drift (px) allowed while "digging" (looser than `stationary_px`; a digging dog shuffles in place). Defaults to `2 x stationary_px` |
| `event_cooldown_seconds` | Min seconds between repeated events |
| `off_delay_seconds` | HA `off_delay` for the binary sensors — auto-reverts to OFF this long after the last ON, even if our OFF message is lost (fixes sensors sticking triggered). Default 180 |
| `min_consecutive` | Consecutive detections required before firing events |
| `startup_timeout_seconds` | Max seconds to wait for this camera's first frame before giving up on it. Default 60. A camera that times out is **skipped**, not fatal: the other cameras start normally and the detector logs a `DEGRADED` line naming the ones that failed. The process only exits non-zero (letting the restart policy retry) when *no* camera at all could be started. This matches the running-state policy described under "Container health" — one dead camera should not restart a container that is successfully watching others. A skipped camera is not retried until the container restarts. |
| `frame_stale_seconds` | If the newest decoded frame is older than this, the camera is treated as stale: detection is skipped and the HA entities go **unavailable** rather than sitting silently at OFF. Default 30. This is what makes a dead RTSP reader visible — a frozen frame is otherwise byte-identical to a static scene. Set 0 to disable. |
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
| `static_suppression_enabled` | (Optional) Suppress events from a bbox region that fires repeatedly at the same position without anything ever having moved there — i.e. a structural element the model keeps scoring as a dog. Default `true`. |
| `static_suppression_iou_threshold` | (Optional) Bbox overlap required to treat two detections as "the same spot". Default 0.7. |
| `static_suppression_max_hits` | (Optional) Same-spot hits before the region is treated as static. Default 3. |
| `static_suppression_decay_seconds` | (Optional) Forget a region after this long with no detections, so a false-positive spot that shifts with the light doesn't stay suppressed forever. Default 300. |
| `static_suppression_protected_events` | (Optional) Event types that are **never** suppressed. Default `["digging"]`. Digging is the event the siren depends on, and the cost of dropping a real one (dogs get out) is far worse than the cost of letting a false one through to vision verification. A protected event also *clears* an existing suppression for that region — a fence beam doesn't dig. |
| `static_suppression_movement_grace_seconds` | (Optional) After the tracker sees real movement into a region, withhold suppression for this long. Defaults to `static_suppression_decay_seconds`. This is what stops a dog that walks in and then holds still from being reclassified as a beam. |
| `write_queue_size` | (Optional) Depth of the per-camera background write queue (event-clip JPEG encodes, debug captures, SQLite inserts). Default 256. These writes are off the detection thread; if the queue overflows the write is dropped and logged rather than stalling detection. |
| `event_store_retention_days` | (Optional) Delete SQLite events older than this. `0` (default) keeps everything forever. Swept hourly. |
| `event_store_busy_timeout_ms` | (Optional) How long a write waits for a locked database before failing. Default 5000. A concurrent reader (`export-daily-events.py`, the `sqlite3` CLI) could otherwise surface "database is locked" on the detection path. |
| `clip_retention_days` | (Optional) Delete event clips in `clip_dir` older than this. `0` (default) keeps everything forever, matching the historical "clips are a permanent record" behaviour — but a full disk makes `imwrite` fail, so being able to cap it matters. Swept hourly. |
| `gpu_decode` | (Optional) Offload RTSP frame decode to GPU via NVDEC. Default `false`. Requires `Dockerfile.gpu` / `docker-compose.gpu.yml` and an NVIDIA GPU. See "Performance tuning → GPU-accelerated decode" above. |

**MQTT security note:** by default the broker connection is plaintext and
unauthenticated, which is fine for a broker that never leaves
localhost/a trusted LAN. If your broker is reachable beyond that (a
different host, a VPN, etc.), secure *both* sides:

| Side | Where to set it | Keys |
| --- | --- | --- |
| Detector (`dogwatch`) | `config.json` per camera | `mqtt_username`, `mqtt_password`, `mqtt_tls` |
| Notifier + find-dogs listener (`dogwatch-notify`) | environment (see `docker-compose.yml`) | `MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_TLS`, `MQTT_TLS_CA_CERT`, `MQTT_TLS_INSECURE` |

The notifier side matters at least as much as the detector's: these are
*subscribers* that act on what they receive. `find-dogs-mqtt.py` runs a full
camera scan on any message to `<base>/find-dogs/trigger/+`, and the notifier
sends Telegram alerts and republishes snapshots — so on an exposed broker,
anyone who can reach the port can drive both. Each process logs its posture
at startup (`auth=none tls=off`), so you can confirm what is actually in
effect rather than what you meant to configure.

### Credentials and what reaches the broker / the logs

Camera URLs (`rtsp_url`, `snapshot_url`) embed credentials inline, so this
project is deliberate about where those strings can end up:

- **Nothing credential-bearing is published to MQTT.** The retained
  `<base>/geometry` topic carries only `detect_w`, `detect_h` and `crop_roi`.
  It previously also carried `snapshot_url` — i.e. NVR credentials, on a
  *retained* topic, on a broker that is unauthenticated by default, replayed to
  every new subscriber, surfaced in Home Assistant entity attributes, and
  outliving the container. Nothing ever consumed the field. Because a retained
  publish replaces the previous message on that topic, simply starting the
  updated detector once purges the old payload from your broker.
- **Credentials are redacted before logging.** All log paths that interpolate a
  URL or a caught subprocess exception go through `redact.py` /
  `pipeline/dw_redact.py`. This matters most for the non-obvious case:
  `subprocess.TimeoutExpired` and `CalledProcessError` stringify their entire
  argv, and the ffmpeg argv always contains the camera URL — so an ordinary
  camera timeout used to print `rtsp://user:pass@host` into `docker logs`.
- **The notifier's config and `secrets.json` remain the only homes for
  credentials**, both gitignored / outside the repo. Lock the latter down with
  `chmod 600 ~/.openclaw/secrets.json`.

### Container health

`restart: unless-stopped` only recovers a process that *exits*. It cannot see
this system's actual failure mode: the process stays alive and goes blind (a
wedged frame grabber, a publisher that never connected, a loop overrunning its
interval). None of those change the exit code.

So `dogwatch.py` writes a heartbeat — the loop's last tick time plus each
camera's frame age, write-queue depth and suppression counters — and
`healthcheck.py` turns it into a container health status:

```bash
docker exec dogwatch python /app/healthcheck.py
# ok: 2 camera(s) healthy
# ok: degraded: 1/2 camera(s) stale (camera), others healthy
# UNHEALTHY: all 1 camera(s) stale: rear-east — no usable frames
# UNHEALTHY: main loop has not ticked for 600s (limit 60s) — detector is wedged

docker exec dogwatch python /app/healthcheck.py --json   # for scripting
```

A *single* stale camera among several is reported as degraded but stays
**healthy** on purpose: the other cameras still work, a restart would disrupt
them too, that camera is already `unavailable` in Home Assistant, and a camera
unplugged for a day shouldn't cause a restart loop.

**Plain Docker does not restart unhealthy containers** (only Swarm does), so the
`HEALTHCHECK` gives you visibility — `docker ps` shows `unhealthy` — and
`pipeline/dogwatch-watchdog.sh` is what acts on it. The watchdog previously only
detected stopped or "zombie" containers; it now also restarts on `unhealthy`,
rate-limited to once per `DOGWATCH_HEALTH_RESTART_MIN_INTERVAL` (default 900s) so
a genuinely offline camera can't cause a restart loop.

| Env var | Default | Description |
|---------|---------|-------------|
| `DOGWATCH_HEARTBEAT_FILE` | `/tmp/dogwatch-heartbeat.json` | Heartbeat location |
| `DOGWATCH_HEARTBEAT_INTERVAL` | `5` | Seconds between heartbeat writes |
| `DOGWATCH_HEALTH_RESTART_MIN_INTERVAL` | `900` | Watchdog: min seconds between health-triggered restarts |

### Disk growth

Three stores used to grow without any bound. All are now cappable, and all
default to the previous keep-forever behaviour so nothing is deleted unless you
opt in:

| Store | Key | Default |
|-------|-----|---------|
| SQLite `events` table | `event_store_retention_days` | 0 (forever) |
| Event clips (`clip_dir`) | `clip_retention_days` | 0 (forever) |
| Debug captures | `debug_capture_retention_days` | 0 (forever) |
| Notifier event log | `DOGWATCH_STATUS_MAX_BYTES` env | 5 MB, then rotated to `.1` |

The first three are swept hourly on the background writer thread. The event log
rotates in place; `dogwatch_check.py` detects a file that has shrunk and re-reads
from the start, and its timestamp watermark stops anything being alerted twice.

### Container hardening

Applied:

- **The third-party `libedgetpu` `.deb` is digest-pinned.** It's fetched over the
  network and installed with `dpkg` as root — the most privileged step in the
  build — and was previously unverified, so a swapped release asset would have
  been installed silently. `Dockerfile.gpu` also no longer swallows install
  failure with `|| true`, which used to produce an image that built fine and then
  failed at inference.
- **`no-new-privileges`** on all compose services.
- **`gcc` removed** from the runtime image (every dependency ships a wheel;
  verified by building).
- **`.dockerignore`** added at both build-context roots. The context previously
  included `models/`, `clips/`, `data/events.db`, `debug_captures/` and all of
  `.git` — none of which is COPYed into either image — plus, for the notifier,
  the credential-bearing `dogwatch-notify.config.json`.

Not applied, deliberately:

- **Non-root `USER`.** Both `/dev/apex_0` and the bind-mounted
  `clips/`/`data/`/`debug_captures/` directories are root-owned on an existing
  deployment. Switching the `USER` without also adding a udev rule (or
  `--group-add`) for the apex device *and* chowning those volumes would break a
  running install on the next `docker compose up`. It's a migration, not a config
  tweak, so it's documented rather than silently changed.
- **`cap_drop: [ALL]`** — plausibly fine, but not verifiable without the Coral
  hardware, so untested.

### Availability / staleness

Every auto-discovered entity now has an `availability_topic`
(`<base>/availability`, retained, with an MQTT Last Will so the *broker*
publishes `offline` if the detector dies uncleanly). A camera is marked
unavailable when its newest decoded frame is older than `frame_stale_seconds`.

This closes a genuine blind spot: a sensor sitting at OFF is indistinguishable
from a detector that has stopped watching. Worse, a frozen frame is
byte-identical to a static scene, and because `motion_gate_max_idle_seconds`
forces a detection pass every 10s regardless of motion, a frozen frame that
happened to contain a dog would re-fire `dog_at_fence` indefinitely. Frames now
carry a capture timestamp, stale frames are skipped, and the RTSP reader logs
its reconnects with exponential backoff instead of retrying silently forever.

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
| `dogwatch-check.sh` | every 5 min (container entrypoint loop; or cron `*/5 * * * *`) | Thin `flock` wrapper — serialises runs so a long cycle can't be double-processed by the next tick |
| `dogwatch_check.py` | invoked by the wrapper | All the actual logic: reads the event log, sends a Telegram ping, runs vision model verification (dog presence **and** digging), sends confirm/false-alarm follow-ups, fires the optional siren |
| `image_quality.py` | imported | Shared grey/partial-decode frame rejection, used by both the notifier and the check script |
| `dogwatch-notify.config.example.json` | — | Template for the camera registry + Telegram chat id used by the notifier |

**Event bookkeeping.** The check script tracks progress in
`/tmp/dogwatch-check-state.json` (`{"ts": ..., "offset": ...}`): `ts` is the
newest event already processed, `offset` is how far into the append-only event
log it has read, so each cycle parses only new lines. The watermark is advanced
per completed event and persisted even if a cycle fails partway, so a crash
re-processes at most one event rather than re-alerting the whole window. A
legacy bare-float `/tmp/dogwatch-last-ts` is read once on upgrade. Events older
than `DOGWATCH_MAX_EVENT_AGE` (default 30 min) are ignored so extended downtime
doesn't replay ancient history.

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
| `DOGWATCH_VISION_API_URL` | OpenRouter chat completions endpoint | Primary chat completions endpoint URL |
| `DOGWATCH_VISION_MODEL` | `qwen/qwen3.7-flash` | Primary model name to request |
| `DOGWATCH_VISION_API_KEY` | (falls back to `secrets.json`) | API key, sent as a `Bearer` token |
| `DOGWATCH_VISION_FALLBACK_API_URL` | OpenRouter chat completions endpoint | Fallback endpoint, tried when the primary fails |
| `DOGWATCH_VISION_FALLBACK_MODEL` | `google/gemini-3-flash-preview` | Fallback model name |
| `DOGWATCH_VISION_FALLBACK_API_KEY` | (falls back to `secrets.json`) | Fallback API key |

The primary is OpenRouter + Qwen (fast, cheap, strong on small objects in wide
frames); **Gemini is the fallback, not the default** — it moved when the free
tier started 429ing mid-scan. When a key is unset it is resolved from
`~/.openclaw/secrets.json`, picking the provider that matches the endpoint
being called (`models.providers.openrouter.apiKey` for `openrouter.ai`,
`models.providers.google.apiKey` otherwise), so the key always matches the API.

**The fallback only helps if it is a genuinely different provider.** With the
defaults, primary and fallback resolve to the same OpenRouter endpoint *and*
the same key, so an account-level 429 rejects both identically. The check
script detects that case, logs it, and skips the pointless second call — point
`DOGWATCH_VISION_FALLBACK_API_URL`/`_API_KEY` at a different provider for a
fallback that actually adds resilience. Note also that there is **no retry or
backoff**: each configured provider is attempted exactly once per event.

### Dog Alarm (optional — siren via Home Assistant)

An optional add-on that sounds a Home Assistant **siren** (e.g. a Tuya
"Dog Alarm" in the back yard) to break the dogs' attention when they dig.
It is completely optional: if the `alarm` section is absent from
`dogwatch-notify.config.json` (or `enabled` is `false`), nothing ever
happens — no code path runs, no errors, no noise.

**What triggers it**

* **Automatic** — only after vision verification **confirms digging**
  (`dogwatch-check.sh` already verifies every event; when the model says
  `digging: YES` it fires the alarm). A plain "dog detected" is never
  enough on its own.
* **Manual** — any chat/agent can ask for it:
  ```bash
  docker exec dogwatch-notify /app/dog-alarm.sh --manual "reason here"
  ```

**Guard rails (all enforced inside `pipeline/dog-alarm.sh`)**

| Guard | Default | Notes |
|-------|---------|-------|
| Time window | `07:00`–`20:00` local | The alarm can **never** sound outside this window. The window may wrap midnight (`22:00`–`06:00`) if you set start > end. |
| Replay guard | `min_interval_sec: 60` | At most one sound per interval, tracked via a persistent `state_file` (survives container restarts). |
| Config gate | `enabled: false` | Silent exit (code 3) when unconfigured. |

Every time the alarm actually sounds, a Telegram message is raised back to
the configured chat (🔔 Dog Alarm sounded — reason + time). Blocked
*manual* attempts also message (🔕 with the reason it was blocked, e.g.
outside hours); blocked automatic attempts only log.

**Configuration** (`alarm` section of `dogwatch-notify.config.json`):

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Master switch for the whole feature. |
| `ha_url` | `http://localhost:8123` | Home Assistant base URL. From the notifier container use `http://172.17.0.1:8123` (docker bridge gateway). |
| `ha_token` | `""` | HA long-lived access token (Bearer). Recommended: HA UI → Profile → Security → Long-lived access tokens. Either `ha_token` *or* `ha_refresh_token` is required. |
| `ha_refresh_token` | `""` | HA refresh token — exchanged for a short-lived access token on every run (`POST /auth/token`). Use when long-lived tokens are unavailable/revoked in your instance. |
| `entity_id` | `siren.dog_alarm` | HA entity to turn on (`siren.turn_on`). The siren's own duration/volume entities (e.g. Tuya `number.*_alarm_time`, `select.*_alarm_volume`) control how long it plays — the script just fires it once. |
| `window_start` / `window_end` | `07:00` / `20:00` | Local-time window (HH:MM, 24h) during which the alarm may sound. |
| `min_interval_sec` | `60` | Minimum seconds between sounds (replay guard). |
| `state_file` | next to config | Persistent file storing the last-sounded epoch. In the container point this at a mounted volume, e.g. `/app/workspace/dog-alarm.state`. |
| `notify_chat` | `true` | Send the Telegram event (sounded / manual-blocked / failure) to `chat_id`. |

All keys can be overridden per-run with env vars: `DOGWATCH_HA_URL`,
`DOGWATCH_HA_TOKEN`, `DOGWATCH_HA_REFRESH_TOKEN`, `DOGWATCH_ALARM_ENTITY`,
`DOGWATCH_ALARM_WINDOW_START`, `DOGWATCH_ALARM_WINDOW_END`,
`DOGWATCH_ALARM_MIN_INTERVAL`, `DOGWATCH_ALARM_STATE_FILE`.

Exit codes: `0` sounded · `2` blocked (outside window) · `3` disabled ·
`4` replay guard · `5` error (config/HA).

See `pipeline/dogwatch-notify.config.example.json` for a fully commented
template.

## Development

Unit tests cover the parts with real logic, as opposed to I/O glue:
`tracker.py`, `behavior.py`, `snapshot_quality.py`, `motion_gate.py`,
`static_suppressor.py`, `event_store.py`, `debug_capture.py`, `detector.py`'s
tensor bookkeeping and bbox clamping, `frame_grabber.py`'s staleness/backoff
signalling, `redact.py`, `dogwatch.py`'s per-camera startup isolation, topic
collision guards and credential-safe error logging, `camera_pipeline.py`'s
`tick()` event bookkeeping, `pipeline/find-dogs.py`'s vision-provider fallback,
and `pipeline/dogwatch_check.py`'s watermark, read-offset
and dedupe logic. They run on plain Python — no Coral hardware or camera feed
needed.

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

`detector.py` imports `ai_edge_litert` lazily (inside `_make_interpreter`), so
its pure-Python helpers are testable without installing the ML runtime — which
is why `requirements-test.txt` doesn't include it.

CI (`.github/workflows/ci.yml`) runs on every push/PR to `main`: unit tests, a
`compileall` syntax check across **every** module (the previous hand-maintained
list silently omitted six files, including the ~870-line `find-dogs.py`), a
`bash -n` check plus an advisory `shellcheck` pass on the shell scripts, and a
full `linux/amd64` build of both Docker images (no Coral hardware in CI, so this
only verifies the images build and install cleanly — not that inference works).

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
| Background writes | Already active | Event-clip JPEG encodes, debug captures and SQLite inserts run on a per-camera writer thread instead of inline in the detection loop. A full-res `cv2.imwrite` measured ~42ms on a 5MP frame — roughly 20% of the 200ms *fleet-wide* budget at `target_fps: 5`, spent precisely when an event was firing. See `write_queue_size`. |
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
  used to inform raising `score_threshold` for that camera. Raising it for a
  single camera genuinely works now — it previously had no effect on any camera
  except the first one loaded, because the shared detector baked in that
  camera's threshold for the whole fleet.

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

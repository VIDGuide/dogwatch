#!/usr/bin/env python3
"""find-dogs.py — ad-hoc "find the dogs" scan across NVR channels.

On-demand only (no persistent inference): grabs one full-res frame per
channel from the NVR main stream, quality-checks it, asks the vision model
whether a dog is visible, and reports results to Telegram with photos of
any dogs found.

Usage (inside the notifier container):
    python3 /app/find-dogs.py scan             # scan configured channels
    python3 /app/find-dogs.py scan 1,14,12     # explicit channel ids
    python3 /app/find-dogs.py montage          # labeled grid of ALL NVR
                                               # channels (identification aid)
    python3 /app/find-dogs.py inbed            # print the 'in bed' line at
                                               # bedtime (overnight fast path)

Config: "find_dogs" section in dogwatch-notify.config.json (gitignored):
    "find_dogs": {
      "nvr_host": "192.168.1.20",
      "nvr_user": "admin",
      "nvr_password": "...",
      "channels": [1, 8, 10, 12, 14],           # ids to scan (in scope)
      "channel_order": [8, 14, 10, 12, 1],      # optional: try these first
                                                 # (early-exit friendly)
      "channel_names": {"1": "Side East", ...}  # optional; overrides NVR names
    }
Channel names come from config find_dogs.channel_names when present, falling
back to a live NVR ISAPI fetch (so labels stay readable even where the NVR
still shows default names like "IPCamera 03").
Env overrides: DOGWATCH_FIND_DOGS_NVR_HOST/_USER/_PASSWORD, and the channel
ids can be passed on the command line.

Vision: same pipeline as dogwatch-check.sh — Gemini (OpenAI-compatible
endpoint) primary, OpenRouter fallback. Keys from secrets.json providers
google / openrouter, or DOGWATCH_VISION_API_KEY / fallback env vars.

Exit codes: 0 ok, 1 config/usage error, 2 all channels failed to grab.
"""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import requests
from PIL import Image, ImageDraw, ImageFont, ImageStat

# ---------------------------------------------------------------------------
# Paths / config resolution (same order as dogwatch-check.sh / dog-alarm.sh)
# ---------------------------------------------------------------------------
NOTIFY_CONFIG = os.environ.get(
    'DOGWATCH_NOTIFY_CONFIG',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'dogwatch-notify.config.json'),
)
SECRETS_FILE = os.environ.get(
    'DOGWATCH_SECRETS_FILE',
    os.path.expanduser('~/.openclaw/secrets.json'),
)

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

GRAB_TIMEOUT = 25          # per ffmpeg attempt
GRAB_ATTEMPTS = 3          # retries on grey/corrupt frame

# Dog-door state: written by find-dogs-mqtt.py from the dogwatch/dogdoor
# MQTT topic (HA automation publishes 'locked'/'open' from the Tuya reed
# switch on the inner locking panel). Read at scan time so a full sweep
# that finds nothing can say "they're inside" when the door is open.
DOOR_STATE_FILE = os.environ.get(
    'DOGWATCH_DOOR_STATE_FILE', '/app/workspace/dogdoor.state')

# Bedtime gates (local time): dogs are in their crates overnight. Michael's
# rules: never outside before 6am; bedtime between 20:00-22:00 when they are
# crated and the door is locked again. So midnight-06:00 and 22:00+ are a
# no-scan fast path; 20:00-21:59 with a locked door is a soft "inside for the
# night" inference on an empty full scan.
IN_BED_HOURS = (0, 6)      # hour < 6 -> in bed, no scan
BEDTIME_START_HOUR = 20    # 20:00+ with door locked -> likely inside/crated
BEDTIME_END_HOUR = 22      # 22:00+ -> crated, hard in-bed gate


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config():
    try:
        with open(NOTIFY_CONFIG) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f'ERROR: cannot read notify config {NOTIFY_CONFIG}: {e}',
              file=sys.stderr)
        sys.exit(1)

    chat_id = cfg.get('chat_id', '')
    bot_token = cfg.get('botToken', '')
    if not bot_token:
        try:
            with open(SECRETS_FILE) as f:
                secrets = json.load(f)
            accounts = secrets['channels']['telegram']['accounts']
            bot_token = (accounts.get('dogwatch', {}).get('botToken')
                         or accounts['default']['botToken'])
        except Exception as e:
            print(f'ERROR: cannot resolve bot token: {e}', file=sys.stderr)
            sys.exit(1)
    if not chat_id or not bot_token:
        print('ERROR: chat_id / botToken missing from config', file=sys.stderr)
        sys.exit(1)

    fd = cfg.get('find_dogs', {})
    nvr = {
        'host': os.environ.get('DOGWATCH_FIND_DOGS_NVR_HOST',
                               fd.get('nvr_host', '')),
        'user': os.environ.get('DOGWATCH_FIND_DOGS_NVR_USER',
                               fd.get('nvr_user', '')),
        'password': os.environ.get('DOGWATCH_FIND_DOGS_NVR_PASSWORD',
                                   fd.get('nvr_password', '')),
    }
    if not nvr['host'] or not nvr['user'] or not nvr['password']:
        print('ERROR: find_dogs.nvr_host/user/password not configured',
              file=sys.stderr)
        sys.exit(1)

    return cfg, chat_id, bot_token, fd, nvr


def load_vision_keys():
    """Primary vision (qwen via OpenRouter) + fallback, provider-aware keys."""
    api_url = os.environ.get(
        'DOGWATCH_VISION_API_URL',
        'https://openrouter.ai/api/v1/chat/completions')
    model = os.environ.get('DOGWATCH_VISION_MODEL', 'qwen/qwen3.7-flash')
    api_key = os.environ.get('DOGWATCH_VISION_API_KEY', '')
    fb_url = os.environ.get(
        'DOGWATCH_VISION_FALLBACK_API_URL',
        'https://openrouter.ai/api/v1/chat/completions')
    fb_model = os.environ.get('DOGWATCH_VISION_FALLBACK_MODEL',
                              'google/gemini-3-flash-preview')
    fb_key = os.environ.get('DOGWATCH_VISION_FALLBACK_API_KEY', '')

    try:
        with open(SECRETS_FILE) as f:
            secrets = json.load(f)
        providers = secrets['models']['providers']
        if not api_key:
            if 'openrouter.ai' in api_url:
                api_key = providers.get('openrouter', {}).get('apiKey', '')
            else:
                api_key = providers.get('google', {}).get('apiKey', '')
        if not fb_key:
            if 'openrouter.ai' in fb_url:
                fb_key = providers.get('openrouter', {}).get('apiKey', '')
            else:
                fb_key = providers.get('google', {}).get('apiKey', '')
    except Exception:
        pass

    if not api_key:
        print('ERROR: no vision API key (set DOGWATCH_VISION_API_KEY or '
              'secrets.json models.providers.*.apiKey)', file=sys.stderr)
        sys.exit(1)
    return api_url, model, api_key, fb_url, fb_model, fb_key


# ---------------------------------------------------------------------------
# NVR channel list (live from ISAPI, digest auth)
# ---------------------------------------------------------------------------
def fetch_channels(nvr):
    """Return {channel_id: name} for all NVR input channels."""
    url = (f'http://{nvr["host"]}/ISAPI/ContentMgmt/InputProxy/channels')
    try:
        r = requests.get(url, auth=requests.auth.HTTPDigestAuth(
            nvr['user'], nvr['password']), timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f'WARN: cannot fetch NVR channel names ({e}) — using ids only',
              file=sys.stderr)
        return {}
    names = {}
    for ch in r.text.split('<InputProxyChannel'):
        cid = None
        cname = None
        for m in ch.split('<'):
            if m.startswith('id>'):
                cid = m[3:].split('<')[0].strip()
            elif m.startswith('name>'):
                cname = m[5:].split('<')[0].strip()
        if cid:
            names[int(cid)] = cname or f'Channel {cid}'
    return names


# ---------------------------------------------------------------------------
# Frame grabbing (RTSP main stream via ffmpeg) + quality guard
# ---------------------------------------------------------------------------
def active_tile_fraction(gray, tiles=8, tile_std_thresh=15.0):
    """PIL port of snapshot_quality.active_tile_fraction (no cv2 here)."""
    w, h = gray.size
    th, tw = h // tiles, w // tiles
    if th == 0 or tw == 0:
        return 1.0
    active = 0
    for ty in range(tiles):
        for tx in range(tiles):
            tile = gray.crop((tx * tw, ty * th, (tx + 1) * tw, (ty + 1) * th))
            if ImageStat.Stat(tile).stddev[0] >= tile_std_thresh:
                active += 1
    return active / (tiles * tiles)


def is_image_bad(img):
    """PIL port of snapshot_quality.is_image_bad (grey/static/corrupt)."""
    if img is None:
        return True
    gray = img.convert('L')
    stat = ImageStat.Stat(gray)
    mean_v, std_v = stat.mean[0], stat.stddev[0]
    if std_v < 8:
        return True
    if 105 < mean_v < 150 and std_v < 12:
        return True
    if 105 < mean_v < 150 and active_tile_fraction(gray) < 0.20:
        return True
    return False


def grab_frame(channel, nvr, out_path):
    """Grab one clean frame from an NVR channel's main stream.

    Grabs a burst of frames and keeps the last one — the first decodable
    frame of an HEVC RTSP stream is often a P-frame without a full
    reference (grey/corrupt). After a few frames the decoder has a clean
    keyframe. Retries on grey/corrupt mid-GOP frames (Hikvision NVR
    quirk). Returns True on success, False if the channel is unreachable.
    """
    url = (f'rtsp://{nvr["user"]}:{nvr["password"]}@{nvr["host"]}:554/'
           f'Streaming/Channels/{channel}01')
    for attempt in range(GRAB_ATTEMPTS):
        try:
            with tempfile.TemporaryDirectory() as td:
                pat = os.path.join(td, 'f_%d.jpg')
                subprocess.run(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error',
                     '-rtsp_transport', 'tcp',
                     '-i', url, '-frames:v', '8', '-q:v', '2',
                     '-f', 'image2', pat],
                    timeout=GRAB_TIMEOUT, capture_output=True, check=True)
                # Keep the last frame of the burst (most likely clean).
                picked = None
                for i in range(8, 0, -1):
                    cand = os.path.join(td, f'f_{i}.jpg')
                    if os.path.exists(cand):
                        img = Image.open(cand)
                        img.load()
                        if not is_image_bad(img):
                            picked = img
                            break
                if picked is None:
                    print(f'  ch{channel}: all burst frames bad (attempt {attempt + 1})',
                          file=sys.stderr)
                    continue
                picked.save(out_path, quality=92)
                return True
        except subprocess.TimeoutExpired:
            print(f'  ch{channel}: grab timeout (attempt {attempt + 1})',
                  file=sys.stderr)
        except Exception as e:
            print(f'  ch{channel}: grab error (attempt {attempt + 1}): {e}',
                  file=sys.stderr)
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Vision verification (Gemini primary, OpenRouter fallback)
# ---------------------------------------------------------------------------
PROMPT = (
    'You are analysing a security camera snapshot of a residential property '
    'to locate the resident dogs.\n'
    'Report whether any dog is clearly visible ANYWHERE in the frame — the '
    'dogs may be small, far from the camera, partly hidden, or in shadow.\n'
    'Consider motion blur, lighting, and common false positives (leaves, '
    'shadows, garden ornaments, cats, people).\n'
    'If a dog is present, also describe its activity in a short phrase.\n'
    'Respond with STRICT JSON only, no prose, in exactly this form:\n'
    '{"dog": "YES"|"NO"|"UNCERTAIN", "activity": "short phrase"}\n'
    'dog = YES if a dog is clearly or very likely present, NO if definitely '
    'not, UNCERTAIN if you cannot tell.\n'
    'activity = 2-5 words describing ONLY the dog(s)\' action or pose (e.g. '
    '"sleeping", "running", "barking", "lying down", "digging", "sitting", '
    '"pacing", "playing") when dog is YES; empty string when dog is NO or '
    'UNCERTAIN. Do NOT include any location or place words (no "by", "at", '
    '"near", "on the", "in the yard", furniture, rooms, or camera names) — '
    'the location is already known from the camera name.'
)


def vision_verify_with(image_path, api_url, model, api_key, label):
    try:
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
    except OSError as e:
        print(f'  vision[{label}] cannot read {image_path}: {e}',
              file=sys.stderr)
        return None, ''

    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': PROMPT},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ],
        }],
        'max_tokens': 512,
        'response_format': {'type': 'json_object'},
    }
    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {api_key}'}
    if 'openrouter.ai' in api_url:
        headers['HTTP-Referer'] = 'https://github.com/VIDGuide/dogwatch'
        headers['X-Title'] = 'DogWatch'
    try:
        r = requests.post(api_url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        print(f'  vision[{label}] API error: {e}', file=sys.stderr)
        return None, ''

    combined = ''
    for choice in result.get('choices', []):
        content = choice.get('message', {}).get('content', '')
        if isinstance(content, str):
            combined += content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    combined += part.get('text', '')
    combined = combined.strip()
    if not combined or len(combined) < 5:
        print(f'  vision[{label}] truncated/empty response: {combined!r}',
              file=sys.stderr)
        return None, ''

    dog = 'UNCERTAIN'
    activity = ''
    try:
        parsed = json.loads(combined)
        dog = str(parsed.get('dog', 'UNCERTAIN')).upper()
        activity = str(parsed.get('activity', '') or '').strip()
    except json.JSONDecodeError:
        up = combined.upper()
        for kw in ('YES', 'NO', 'UNCERTAIN'):
            if kw in up:
                dog = kw
                break
        print(f'  vision[{label}] non-JSON response: {combined}',
              file=sys.stderr)
    if dog not in ('YES', 'NO', 'UNCERTAIN'):
        dog = 'UNCERTAIN'
    print(f'  vision[{label}] OK: dog={dog} activity={activity!r}',
          file=sys.stderr)
    return dog, activity


def vision_verify(image_path, keys):
    api_url, model, api_key, fb_url, fb_model, fb_key = keys
    dog, activity = vision_verify_with(image_path, api_url, model, api_key,
                                       'primary')
    if dog is None:
        dog, activity = vision_verify_with(image_path, fb_url, fb_model,
                                           fb_key, 'fallback')
    return dog, activity


def _deepseek_line(prompt):
    """Call DeepSeek to compose ONE short line. Returns '' on any failure.

    Shared by all voice-line composers (found / no-dogs / scanning ack).
    Env overrides: DOGWATCH_VOICE_API_URL/_MODEL/_API_KEY
    (default: deepseek provider key from secrets.json).
    """
    api_url = os.environ.get(
        'DOGWATCH_VOICE_API_URL',
        'https://api.deepseek.com/v1/chat/completions')
    model = os.environ.get('DOGWATCH_VOICE_MODEL', 'deepseek-chat')
    api_key = os.environ.get('DOGWATCH_VOICE_API_KEY', '')
    try:
        with open(SECRETS_FILE) as f:
            secrets = json.load(f)
        if not api_key:
            api_key = (secrets.get('models', {}).get('providers', {})
                       .get('deepseek', {}).get('apiKey', ''))
    except Exception:
        pass
    if not api_key:
        print('  voice[deepseek] no API key — using template',
              file=sys.stderr)
        return ''

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 60,
        'temperature': 0.9,
    }
    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.post(api_url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        result = r.json()
        text = ''
        for choice in result.get('choices', []):
            content = choice.get('message', {}).get('content', '')
            if isinstance(content, str):
                text += content
        text = text.strip().strip('"').strip()
        if not text or len(text) > 200:
            print(f'  voice[deepseek] unusable response: {text!r}',
                  file=sys.stderr)
            return ''
        print(f'  voice[deepseek] OK: {text}', file=sys.stderr)
        return text
    except Exception as e:
        print(f'  voice[deepseek] API error: {e} — using template',
              file=sys.stderr)
        return ''


def compose_voice_line(spots, keys):
    """DeepSeek composes a varied, natural one-liner for the Alexa announce.

    spots = [(voice_name, activity), ...] — e.g. [('Back Door', 'sleeping')].
    Returns the composed sentence, or '' if the call fails (caller falls
    back to the deterministic template).
    """
    loc = ', '.join(f'{n} ({a})' if a else n for n, a in spots)
    prompt = (
        'Write ONE short, warm, natural sentence telling the owner where '
        'their dogs are and what they are doing. Speak plainly, like a '
        'helpful assistant — no emojis, no markdown, no quotes, no '
        'introductory words. Vary the phrasing and structure each time; '
        'do not always start with "Found". Keep it under 14 words.\n'
        'The location name (e.g. "Back Door") is WHERE the dog is — refer '
        'to it naturally as a place ("by the back door", "near the back '
        'gate"), NEVER as the subject of the sentence, and never repeat a '
        'location word that already appears in the activity description.\n'
        f'Dogs found: {loc}.'
    )
    return _deepseek_line(prompt)


def compose_no_dogs_line(location=''):
    """DeepSeek composes a varied 'no dogs found' line (canned fallback).

    location: optional voice name (e.g. 'Back Gate') when only that camera
    was scanned — composes 'no dogs at the back gate right now' instead of
    the whole-yard phrasing.
    """
    if location:
        prompt = (
            'Write ONE short, warm, natural sentence telling the owner that '
            f'no dogs were found at {location} right now. Speak plainly, '
            'like a helpful assistant — no emojis, no markdown, no quotes, '
            'no introductory words. Vary the phrasing and structure each '
            'time; do not always start with "No". Keep it under 10 words.\n'
            f'No dogs at {location} right now.'
        )
        line = _deepseek_line(prompt)
        return line or f'No dogs at {location} right now.'
    prompt = (
        'Write ONE short, warm, natural sentence telling the owner that no '
        'dogs were found anywhere in the yard this time. Speak plainly, '
        'like a helpful assistant — no emojis, no markdown, no quotes, no '
        'introductory words. Vary the phrasing and structure each time; do '
        'not always start with "No". Keep it under 12 words.\n'
        'No dogs were found in the yard.'
    )
    return _deepseek_line(prompt) or 'No dogs found.'


def compose_ack_line():
    """DeepSeek composes a varied 'scanning the yard' ack (canned fallback)."""
    prompt = (
        'Write ONE short, natural sentence acknowledging that you are '
        'starting to scan the yard cameras to find the dogs. Speak plainly, '
        'like a helpful assistant — no emojis, no markdown, no quotes, no '
        'introductory words. Vary the phrasing and structure each time; do '
        'not always start with "On it" or "Scanning". Keep it under 12 '
        'words.\n'
        'On it, scanning the yard for the dogs.'
    )
    return _deepseek_line(prompt) or 'On it, scanning the yard for the dogs.'


def compose_in_bed_line():
    """DeepSeek composes a varied 'dogs are in bed / crates' line."""
    prompt = (
        'Write ONE short, warm, natural sentence telling the owner that '
        'their dogs are in bed, in their crates for the night. Speak '
        'plainly, like a helpful assistant — no emojis, no markdown, no '
        'quotes, no introductory words. Vary the phrasing and structure '
        'each time; do not always start with "They". Keep it under 10 '
        'words.\n'
        'The dogs are in their crates for the night.'
    )
    return _deepseek_line(prompt) or 'They are in their crates for the night.'


def compose_inside_line():
    """DeepSeek composes a varied 'door open -> they must be inside' line."""
    prompt = (
        'Write ONE short, warm, natural sentence telling the owner that '
        'the dogs are not in the yard and, since the doggy door is open, '
        'they must have come inside the house. Speak plainly, like a '
        'helpful assistant — no emojis, no markdown, no quotes, no '
        'introductory words. Vary the phrasing and structure each time; '
        'do not always start with "They". Keep it under 14 words.\n'
        'The doggy door is open, so they must be inside.'
    )
    return _deepseek_line(prompt) or 'The doggy door is open — they must be inside.'


def load_door_state():
    """Return 'open' / 'locked' / '' (unknown) from the door state file.

    Written by find-dogs-mqtt.py whenever dogwatch/dogdoor arrives on MQTT
    (HA automation publishes 'locked'/'open' from the Tuya reed switch on
    the doggy-door inner locking panel). Missing/unreadable file -> ''.
    """
    try:
        with open(DOOR_STATE_FILE) as f:
            return json.load(f).get('state', '')
    except (OSError, ValueError, json.JSONDecodeError):
        return ''


def bedtime_gate():
    """Return (in_bed, line) — true overnight (midnight-06:00 and 22:00+).

    When true the caller should skip the camera scan entirely and just
    answer with the in-bed line (dogs are crated, no point scanning).
    """
    hour = int(time.strftime('%H', time.localtime()))
    if hour < IN_BED_HOURS[1] or hour >= BEDTIME_END_HOUR:
        return True, compose_in_bed_line()
    return False, ''


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def tg_send(token, chat_id, text):
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id': chat_id, 'text': text}, timeout=15)
        return r.ok
    except Exception as e:
        print(f'  TG send error: {e}', file=sys.stderr)
        return False


def tg_send_photo(token, chat_id, photo_path, caption):
    try:
        with open(photo_path, 'rb') as f:
            r = requests.post(
                f'https://api.telegram.org/bot{token}/sendPhoto',
                data={'chat_id': chat_id, 'caption': caption},
                files={'photo': ('frame.jpg', f, 'image/jpeg')}, timeout=30)
        return r.ok
    except Exception as e:
        print(f'  TG photo error: {e}', file=sys.stderr)
        return False


def resolve_names(fd, nvr):
    """Channel display names: config find_dogs.channel_names win; NVR ISAPI
    names fill the gaps (fallback). The NVR still shows default names for
    several channels (e.g. "IPCamera 03"), so the config map is authoritative."""
    names = fetch_channels(nvr)
    for cid, name in (fd.get('channel_names') or {}).items():
        names[int(cid)] = name
    return names


def resolve_voice_names(fd, nvr):
    """Voice/TTS-friendly names for the announce path. Config
    find_dogs.voice_names win (e.g. "12": "Back Door" — "Rear East" is
    awkward aloud), then channel_names, then NVR names."""
    names = resolve_names(fd, nvr)
    for cid, name in (fd.get('voice_names') or {}).items():
        names[int(cid)] = name
    return names


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_montage(cfg, chat_id, bot_token, fd, nvr):
    """Grab every NVR channel, stitch into one labeled grid, send it."""
    names = resolve_names(fd, nvr)
    ids = sorted(names.keys()) or list(range(1, 17))
    grabs, failed = [], []
    with tempfile.TemporaryDirectory() as td:
        for ch in ids:
            p = os.path.join(td, f'ch{ch}.jpg')
            if grab_frame(ch, nvr, p):
                img = Image.open(p).convert('RGB')
                img.thumbnail((640, 480))
                grabs.append((ch, names.get(ch, f'Channel {ch}'), img))
            else:
                failed.append(ch)

        if not grabs:
            print('ERROR: no channels grabbed', file=sys.stderr)
            return 2

        cols = 4
        cell_w = max(i.width for _, _, i in grabs)
        cell_h = max(i.height for _, _, i in grabs) + 34  # label bar
        rows = (len(grabs) + cols - 1) // cols
        grid = Image.new('RGB', (cols * cell_w, rows * cell_h), (30, 30, 30))
        draw = ImageDraw.Draw(grid)
        font = ImageFont.truetype(FONT_PATH, 20)
        for idx, (ch, name, img) in enumerate(grabs):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            grid.paste(img, (x + (cell_w - img.width) // 2, y))
            draw.text((x + 8, y + cell_h - 30), f'ch{ch}  {name}',
                      fill=(255, 255, 255), font=font)
        grid_path = os.path.join(td, 'montage.jpg')
        grid.save(grid_path, quality=90)

        caption = f'📷 NVR channels ({len(grabs)}/16)'
        if failed:
            caption += f' — no signal: ch{", ".join(map(str, failed))}'
        ok = tg_send_photo(bot_token, chat_id, grid_path, caption)
        print(f'montage sent: {ok} ({len(grabs)} channels)')
    return 0 if ok else 1


def mode_scan(channel_ids, cfg, chat_id, bot_token, fd, nvr, keys, summary_file=None):
    """Grab + vision-verify each in-scope channel, report with photos.

    Exits early once find_dogs.max_found cameras have a dog (default 1 —
    there are at most 2 dogs and they're nearly always together, so the
    first camera that has one is enough). Remaining cameras are skipped
    and the summary notes the early stop.
    """
    # Overnight fast path: dogs are crated (midnight-06:00 and 22:00+).
    # Answer directly — no point scanning an empty yard, and this keeps the
    # 2am "where are the dogs" question instant.
    in_bed, bed_line = bedtime_gate()
    if in_bed:
        ok = tg_send(bot_token, chat_id, f'🌙 {bed_line}')
        print('summary sent:', ok)
        if summary_file:
            try:
                with open(summary_file, 'w') as f:
                    f.write(bed_line)
                print(f'summary written: {summary_file}')
            except OSError as e:
                print(f'ERROR: cannot write summary file {summary_file}: {e}',
                      file=sys.stderr)
        print('voice summary:', bed_line)
        return 0 if ok else 1

    names = resolve_names(fd, nvr)
    if not channel_ids:
        channel_ids = fd.get('channels', [])
        # Preferred scan order (early-exit friendly): channel_order lists
        # channels to try first, in order; the remaining in-scope channels
        # follow as configured. Absent key = scan channels in config order.
        # Explicit CLI channel lists are never reordered.
        order = fd.get('channel_order') or []
        if order:
            preferred = [c for c in order if c in channel_ids]
            rest = [c for c in channel_ids if c not in preferred]
            channel_ids = preferred + rest
    if not channel_ids:
        print('ERROR: no channels to scan (config find_dogs.channels or CLI)',
              file=sys.stderr)
        return 1
    try:
        max_found = int(fd.get('max_found', 1) or 1)
    except (TypeError, ValueError):
        max_found = 1

    found, clear, uncertain, failed = [], [], [], []
    scanned = 0
    early_exit = False
    with tempfile.TemporaryDirectory() as td:
        for ch in channel_ids:
            if len(found) >= max_found:
                early_exit = True
                break
            scanned += 1
            name = names.get(ch, f'Channel {ch}')
            p = os.path.join(td, f'ch{ch}.jpg')
            if not grab_frame(ch, nvr, p):
                failed.append((ch, name))
                continue
            dog, activity = vision_verify(p, keys)
            if dog == 'YES':
                found.append((ch, name, p, activity))
            elif dog == 'NO':
                clear.append((ch, name))
            else:
                uncertain.append((ch, name))

        door = load_door_state()
        lines = [f'🔍 Find the dogs — {scanned} cameras scanned']
        if early_exit:
            lines.append('⏩ Stopped early: dogs found')
        for ch, name, p, activity in found:
            line = f'🐕 {name} (ch{ch})'
            if activity:
                line += f' — {activity}'
            lines.append(line)
        if uncertain:
            lines.append('❓ Uncertain: '
                         + ', '.join(f'{n} (ch{c})' for c, n in uncertain))
        if clear:
            lines.append('✅ Clear: '
                         + ', '.join(f'{n} (ch{c})' for c, n in clear))
        if failed:
            lines.append('📡 No signal: '
                         + ', '.join(f'{n} (ch{c})' for c, n in failed))
        if not found:
            if door == 'open' and len(channel_ids) != 1:
                lines.append('\nNo dogs found in the yard — doggy door is '
                             'open, they may be inside.')
            else:
                lines.append('\nNo dogs found.')
        ok = tg_send(bot_token, chat_id, '\n'.join(lines))
        print('summary sent:', ok)

        # Plain-text voice summary for the Alexa/HA announce path — concise
        # by design: only where the dogs WERE found (or a plain "no dogs
        # found"). The full breakdown (clear / uncertain / no signal) stays
        # on Telegram; a verbose announce would get the skill disabled.
        # If the dogs were found, a DeepSeek call composes a varied, natural
        # one-liner from location + activity (sleeping / running / ...); the
        # deterministic template below is the fallback if that call fails.
        vnames = resolve_voice_names(fd, nvr)
        if found:
            spots = [(vnames.get(ch, n), act) for ch, n, _, act in found]
            voice_summary = compose_voice_line(spots, keys)
            if not voice_summary:
                voice_summary = ('Found the dogs at the '
                                 + ', the '.join(n for n, _ in spots) + '.')
        else:
            if door == 'open' and len(channel_ids) != 1:
                voice_summary = compose_inside_line()
            else:
                loc = ''
                if len(channel_ids) == 1 and channel_ids[0] not in [f[0] for f in failed]:
                    loc = vnames.get(channel_ids[0], names.get(channel_ids[0], ''))
                voice_summary = compose_no_dogs_line(loc)
        if summary_file:
            try:
                with open(summary_file, 'w') as f:
                    f.write(voice_summary)
                print(f'summary written: {summary_file}')
            except OSError as e:
                print(f'ERROR: cannot write summary file {summary_file}: {e}',
                      file=sys.stderr)
        print('voice summary:', voice_summary)

        for ch, name, p, activity in found:
            okp = tg_send_photo(bot_token, chat_id, p,
                                f'🐕 {name} (ch{ch})')
            print(f'photo ch{ch} sent: {okp}')
            time.sleep(1)
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    mode = args[0] if args else 'scan'
    cfg, chat_id, bot_token, fd, nvr = load_config()

    if mode == 'montage':
        return mode_montage(cfg, chat_id, bot_token, fd, nvr)

    if mode == 'ack':
        # Compose (and print) the 'scanning the yard' ack line for the
        # Alexa/HA voice path — called by find-dogs-mqtt.py before the scan
        # so the Echo can acknowledge while the cameras are being scanned.
        print(compose_ack_line())
        return 0

    if mode == 'inbed':
        # Overnight fast-path check for find-dogs-mqtt.py: prints the
        # composed 'in bed' line when it's bedtime (dogs are crated), else
        # nothing. The listener uses this to skip the scan entirely.
        in_bed, line = bedtime_gate()
        if in_bed:
            print(line)
        return 0

    if mode != 'scan':
        print(f'ERROR: unknown mode {mode!r} (scan|montage|ack)',
              file=sys.stderr)
        return 1

    channel_ids = []
    summary_file = None
    rest = args[1:]
    # Parse --summary-file wherever it appears (the listener passes it AFTER
    # the channel id: 'scan 14 --summary-file <path>').
    i = 0
    while i < len(rest):
        if rest[i] == '--summary-file' and i + 1 < len(rest):
            summary_file = rest[i + 1]
            i += 2
            continue
        for part in rest[i].split(','):
            try:
                channel_ids.append(int(part.strip()))
            except ValueError:
                print(f'ERROR: bad channel id {part!r}', file=sys.stderr)
                return 1
        i += 1
    keys = load_vision_keys()
    return mode_scan(channel_ids, cfg, chat_id, bot_token, fd, nvr, keys,
                     summary_file=summary_file)


if __name__ == '__main__':
    sys.exit(main())

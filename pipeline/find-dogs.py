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

Config: "find_dogs" section in dogwatch-notify.config.json (gitignored):
    "find_dogs": {
      "nvr_host": "192.168.1.20",
      "nvr_user": "admin",
      "nvr_password": "...",
      "channels": [1, 8, 10, 12, 14]           # ids to scan (in scope)
    }
Channel names are fetched live from the NVR (ISAPI) so labels stay current.
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
    """Gemini primary + OpenRouter fallback keys (same as check.sh)."""
    api_url = os.environ.get(
        'DOGWATCH_VISION_API_URL',
        'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
    model = os.environ.get('DOGWATCH_VISION_MODEL', 'gemini-3-flash-preview')
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
            api_key = providers.get('google', {}).get('apiKey', '')
        if not fb_key:
            fb_key = providers.get('openrouter', {}).get('apiKey', '')
    except Exception:
        pass

    if not api_key:
        print('ERROR: no vision API key (set DOGWATCH_VISION_API_KEY or '
              'secrets.json models.providers.google.apiKey)', file=sys.stderr)
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
    'Respond with STRICT JSON only, no prose, in exactly this form:\n'
    '{"dog": "YES"|"NO"|"UNCERTAIN"}\n'
    'dog = YES if a dog is clearly or very likely present, NO if definitely '
    'not, UNCERTAIN if you cannot tell.'
)


def vision_verify_with(image_path, api_url, model, api_key, label):
    try:
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
    except OSError as e:
        print(f'  vision[{label}] cannot read {image_path}: {e}',
              file=sys.stderr)
        return None

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
        return None

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
        return None

    dog = 'UNCERTAIN'
    try:
        dog = str(json.loads(combined).get('dog', 'UNCERTAIN')).upper()
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
    print(f'  vision[{label}] OK: dog={dog}', file=sys.stderr)
    return dog


def vision_verify(image_path, keys):
    api_url, model, api_key, fb_url, fb_model, fb_key = keys
    dog = vision_verify_with(image_path, api_url, model, api_key, 'primary')
    if dog is None:
        dog = vision_verify_with(image_path, fb_url, fb_model, fb_key,
                                 'fallback')
    return dog


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


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_montage(cfg, chat_id, bot_token, fd, nvr):
    """Grab every NVR channel, stitch into one labeled grid, send it."""
    names = fetch_channels(nvr)
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


def mode_scan(channel_ids, cfg, chat_id, bot_token, fd, nvr, keys):
    """Grab + vision-verify each in-scope channel, report with photos."""
    names = fetch_channels(nvr)
    if not channel_ids:
        channel_ids = fd.get('channels', [])
    if not channel_ids:
        print('ERROR: no channels to scan (config find_dogs.channels or CLI)',
              file=sys.stderr)
        return 1

    found, clear, uncertain, failed = [], [], [], []
    with tempfile.TemporaryDirectory() as td:
        for ch in channel_ids:
            name = names.get(ch, f'Channel {ch}')
            p = os.path.join(td, f'ch{ch}.jpg')
            if not grab_frame(ch, nvr, p):
                failed.append((ch, name))
                continue
            dog = vision_verify(p, keys)
            if dog == 'YES':
                found.append((ch, name, p))
            elif dog == 'NO':
                clear.append((ch, name))
            else:
                uncertain.append((ch, name))

        lines = [f'🔍 Find the dogs — {len(channel_ids)} cameras scanned']
        for ch, name, p in found:
            lines.append(f'🐕 {name} (ch{ch})')
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
            lines.append('\nNo dogs found.')
        ok = tg_send(bot_token, chat_id, '\n'.join(lines))
        print('summary sent:', ok)

        for ch, name, p in found:
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

    if mode != 'scan':
        print(f'ERROR: unknown mode {mode!r} (scan|montage)', file=sys.stderr)
        return 1

    channel_ids = []
    if len(args) > 1:
        for part in args[1].split(','):
            try:
                channel_ids.append(int(part.strip()))
            except ValueError:
                print(f'ERROR: bad channel id {part!r}', file=sys.stderr)
                return 1
    keys = load_vision_keys()
    return mode_scan(channel_ids, cfg, chat_id, bot_token, fd, nvr, keys)


if __name__ == '__main__':
    sys.exit(main())

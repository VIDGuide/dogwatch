#!/usr/bin/env python3
"""DogWatch daily statistics — per-date counters feeding the Daily Dog Report.

Components bump counters into a single JSON file in the shared workspace
volume (notifier container). The host cron wrapper copies it into the clips
dir each morning so the n8n report script (/mnt/clips, read-only) can read
it alongside daily-events.json.

Counters (all per local Sydney day):
  vision_checks          vision verification attempts (snapshot available)
  vision_primary_ok      primary provider returned a verdict
  vision_fallback_ok     primary failed, fallback returned a verdict
  vision_failed          every configured provider failed
  vision_dog_confirmed   verdict DOG
  vision_false_alarm     verdict NO_DOG
  vision_uncertain       verdict UNCERTAIN
  alarm_sounds           siren actually sounded (exit 0)
  alarm_blocked_window   blocked by the time window (exit 2)
  alarm_blocked_replay   blocked by the replay guard (exit 4)
  alarm_errors           Home Assistant / config errors (exit 5)
  find_dogs              "where are the dogs" skill triggers (any device)
  find_dogs_found        scan result: dog(s) found
  find_dogs_empty        scan result: no dogs found (yard clear)
  find_dogs_inside       scan result: no dogs found + doggy door open
  find_dogs_inbed        bedtime fast path (crates) — no scan run

Usage:
  stats.py bump <key> [amount]   increment today's counter (default 1)
  stats.py get [YYYY-MM-DD]      print one day's counters as JSON
                                 (default: today, Sydney time)
  stats.py prune [days]          drop entries older than N days (default 14)

Env overrides:
  DOGWATCH_STATS_FILE    stats file path (default /app/workspace/daily-stats.json)

Capture must NEVER break the alert pipeline — callers ignore failures and
this module degrades gracefully (missing file → empty stats; bad file →
backed up and restarted).
"""

import fcntl
import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

STATS_FILE = os.environ.get('DOGWATCH_STATS_FILE',
                            '/app/workspace/daily-stats.json')
LOCK_FILE = STATS_FILE + '.lock'
TZ = ZoneInfo('Australia/Sydney')
RETAIN_DAYS = 14

KNOWN_KEYS = (
    'vision_checks', 'vision_primary_ok', 'vision_fallback_ok',
    'vision_failed', 'vision_dog_confirmed', 'vision_false_alarm',
    'vision_uncertain',
    'alarm_sounds', 'alarm_blocked_window', 'alarm_blocked_replay',
    'alarm_errors',
    'find_dogs', 'find_dogs_found', 'find_dogs_empty', 'find_dogs_inside',
    'find_dogs_inbed',
)


def _today():
    return datetime.now(TZ).strftime('%Y-%m-%d')


def _load():
    """Return the full stats structure (possibly empty)."""
    try:
        with open(STATS_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('days'), dict):
            return data
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as e:
        # Corrupt file — back it up and start fresh rather than crash callers.
        try:
            os.rename(STATS_FILE, STATS_FILE + '.corrupt-' +
                      datetime.now(TZ).strftime('%Y%m%d%H%M%S'))
        except OSError:
            pass
        print(f'stats: resetting corrupt stats file: {e}', file=sys.stderr)
    return {'days': {}}


def _save(data):
    """Atomic write (tmp + rename) so concurrent readers never see a torn file."""
    tmp = STATS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATS_FILE)


def _locked(fn):
    """Serialize read-modify-write across processes (check loop, alarm, listener)."""
    os.makedirs(os.path.dirname(LOCK_FILE) or '.', exist_ok=True)
    with open(LOCK_FILE, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def bump(key, amount=1):
    """Increment today's counter for `key`. Unknown keys are still recorded
    (forward-compatible); callers are responsible for typos."""
    if key not in KNOWN_KEYS:
        print(f'stats: warning — bumping unknown key {key!r}', file=sys.stderr)
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        amount = 1

    def _do():
        data = _load()
        days = data['days']
        today = _today()
        day = days.setdefault(today, {})
        day[key] = int(day.get(key, 0)) + amount
        # Prune: keep only recent days so the file stays tiny.
        cutoff = (datetime.now(TZ) - timedelta(days=RETAIN_DAYS)
                  ).strftime('%Y-%m-%d')
        for d in [d for d in days if d < cutoff]:
            del days[d]
        _save(data)

    try:
        _locked(_do)
        return True
    except OSError as e:
        print(f'stats: bump {key} failed: {e}', file=sys.stderr)
        return False


def get(day=None):
    """Return the counters dict for `day` (default today), never None."""
    day = day or _today()
    data = _load()
    return dict(data['days'].get(day, {}))


def prune(days=None):
    days = int(days or RETAIN_DAYS)
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime('%Y-%m-%d')

    def _do():
        data = _load()
        before = len(data['days'])
        for d in [d for d in data['days'] if d < cutoff]:
            del data['days'][d]
        _save(data)
        return before - len(data['days'])

    try:
        return _locked(_do)
    except OSError as e:
        print(f'stats: prune failed: {e}', file=sys.stderr)
        return 0


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    cmd = args[0]
    if cmd == 'bump' and len(args) >= 2:
        amount = int(args[2]) if len(args) > 2 else 1
        ok = bump(args[1], amount)
        print(f'{args[1]}: {get().get(args[1], 0)}')
        return 0 if ok else 1
    if cmd == 'get':
        print(json.dumps(get(args[1] if len(args) > 1 else None),
                         sort_keys=True))
        return 0
    if cmd == 'prune':
        removed = prune(args[1] if len(args) > 1 else None)
        print(f'pruned {removed} stale day(s)')
        return 0
    print(f'stats: unknown command {cmd!r}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())

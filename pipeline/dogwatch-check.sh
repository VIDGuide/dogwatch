#!/bin/bash
# DogWatch event checker — thin wrapper around dogwatch_check.py.
#
# All logic now lives in dogwatch_check.py. This script used to embed ~590
# lines of Python in a quoted heredoc, which meant the watermark, dedupe and
# vision-verification logic could not be imported, unit tested, or even
# syntax-checked (CI could only run `bash -n` on the wrapper). See that
# module's docstring for the full list of behavioural fixes.
#
# The wrapper's only remaining jobs are:
#   1. Serialise runs with flock (see below).
#   2. Locate the interpreter and the module.
#
# Configuration is by environment variable and is read directly by the Python
# module — no shell-side resolution, so there is no longer any place where a
# path or a config value is interpolated into a `python3 -c` program.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PYTHON_BIN="${DOGWATCH_PYTHON:-python3}"

# Lock file location.
#
# NOT a bare /tmp/dogwatch-check.lock. That is a predictable name in a
# world-writable, sticky directory, and this script's documented deployment
# includes a host crontab entry — so on a multi-account host any local user
# could:
#
#   * pre-create the path and hold flock on it, silently disabling every
#     future check cycle (no alerts, no siren, and we exit 0 so cron and the
#     entrypoint loop both consider the run a success), or
#   * point it at a symlink, which the `exec 9>` below would follow and
#     truncate.
#
# Instead the lock lives in a directory we own with mode 0700, which the
# sticky bit on /tmp cannot be used to hijack: if the path exists but is not
# a directory we own, mkdir/chmod fails and `set -e` aborts the run rather
# than proceeding against an attacker-controlled lock. Per-UID so two accounts
# on the same host don't collide. Override wholesale with DOGWATCH_LOCK_FILE
# (e.g. to put it on a tmpfs of your choosing).
if [ -n "${DOGWATCH_LOCK_DIR:-}" ]; then
  LOCK_DIR="$DOGWATCH_LOCK_DIR"
elif [ -n "${XDG_RUNTIME_DIR:-}" ]; then
  LOCK_DIR="$XDG_RUNTIME_DIR/dogwatch"
else
  LOCK_DIR="/tmp/dogwatch-$(id -u)"
fi
mkdir -m 0700 -p "$LOCK_DIR"
chmod 0700 "$LOCK_DIR"
LOCK_FILE="${DOGWATCH_LOCK_FILE:-$LOCK_DIR/dogwatch-check.lock}"
# Refuse a symlinked lock outright rather than following it.
if [ -L "$LOCK_FILE" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] refusing to use symlinked lock file $LOCK_FILE" >&2
  exit 1
fi

# Serialise concurrent runs.
#
# Why this matters: the README documents this script as a `*/5 * * * *` cron
# entry, and the containerised entrypoint runs it in a `sleep 300` loop. Either
# way, a cycle that runs long (each vision-confirmed digging event costs two
# model calls plus a 30s siren follow-up) overlaps the next invocation. With no
# lock, two runs read the same watermark before either wrote it, so both sent
# the alert, both spent vision API quota, and both fired the siren — with only
# dog-alarm.sh's 60s replay guard limiting the damage.
#
# Non-blocking: if a run is already in progress we exit 0 quietly rather than
# queueing up, because the next scheduled tick will pick up any new events
# anyway (the watermark makes that safe).
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] previous check still running — skipping this cycle"
  exit 0
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/dogwatch_check.py" "$@"

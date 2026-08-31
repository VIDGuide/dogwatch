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
LOCK_FILE="${DOGWATCH_LOCK_FILE:-/tmp/dogwatch-check.lock}"

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

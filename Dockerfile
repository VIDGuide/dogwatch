# dogwatch — Coral Edge TPU dog detector
#
# Uses ai-edge-litert (LiteRT, the successor to tflite_runtime) instead of
# pycoral/tflite_runtime. pycoral is abandoned upstream and only ever shipped
# cp39 wheels, which pinned this whole image to Python 3.9 (EOL 2025-10-31)
# and, downstream of that, to numpy 1.x (pycoral's compiled bindings were
# built against the numpy 1.x C ABI). ai-edge-litert ships wheels through
# Python 3.14 and has no numpy ceiling, which is what unblocks the numpy/
# opencv bumps below. See README "Known limitations" and GitHub issue #1 for
# the full history of the Python 3.9 pin this replaces.
FROM python:3.12-slim-bookworm

# libedgetpu runtime (std = standard clock speed, good thermals).
# feranick's fork is the community-maintained continuation of Google's
# abandoned libedgetpu — this build (16.0TF2.19.1-1) is built against
# TensorFlow 2.19.1 and lists ai-edge-litert as its recommended pairing.
ADD https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.bookworm_amd64.deb \
    /tmp/libedgetpu.deb

# Verify the download before installing it as root.
#
# This is a third-party community .deb fetched over the network and installed
# with dpkg as uid 0 — the most privileged thing in the whole build. It was
# previously unverified, so a compromised or swapped release asset would have
# been installed silently. Pinned by digest instead.
#
# Deliberately `sha256sum -c` in a RUN rather than `ADD --checksum=`: the latter
# needs a recent BuildKit, while this works with any builder and puts the
# verification in the build log.
#
# To bump the version, change the URL above and replace the digest with the
# output of:
#   curl -fsSL <url> | sha256sum
RUN echo "23be53c72eff4d44afc2f727700da185791d3ca0867bd0b5e082ec3a0de21925  /tmp/libedgetpu.deb" \
      | sha256sum -c - \
    && apt-get update && apt-get install -y --no-install-recommends \
    udev \
    usbutils \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg -i /tmp/libedgetpu.deb \
    && rm /tmp/libedgetpu.deb

# ai-edge-litert + current dependency versions (cp312 wheels, no numpy
# ceiling). numpy/opencv/requests/paho-mqtt/shapely are all on their latest
# stable releases as of this writing — re-check periodically, but there is
# no known structural constraint pinning any of them anymore.
RUN pip install --no-cache-dir \
    ai-edge-litert==2.1.6 \
    paho-mqtt==2.1.0 \
    numpy==2.5.1 \
    opencv-python-headless==5.0.0.93 \
    shapely==2.1.2 \
    requests==2.34.2

COPY *.py /app/
WORKDIR /app

# Report whether the detector is actually still watching, not merely running.
#
# The process can be alive and completely blind: a wedged/dead frame grabber
# keeps returning the same stale frame, and before frame timestamps existed that
# was indistinguishable from a static scene. dogwatch.py now writes a heartbeat
# containing the loop timestamp and each camera's frame age; healthcheck.py
# fails when the loop has stopped or every camera has gone stale.
#
# NOTE: plain Docker does not restart unhealthy containers (only Swarm does), so
# this provides visibility (`docker ps` shows "unhealthy") and a signal for
# pipeline/dogwatch-watchdog.sh, which acts on it.
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

# Runs as root because /dev/apex_0 is typically root-owned and the bind-mounted
# clips/data/debug_captures directories are root-owned on existing deployments.
# To run unprivileged you would need to (a) grant the runtime user access to the
# apex device (udev rule / --group-add) and (b) chown the mounted volumes —
# switching the USER here without both would break an existing install, so it is
# left as a deliberate, documented choice rather than a silent one. See the
# "Container hardening" section of the README.
CMD ["python", "-u", "dogwatch.py"]

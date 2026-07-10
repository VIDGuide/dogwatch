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

RUN apt-get update && apt-get install -y --no-install-recommends \
    udev \
    usbutils \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    gcc \
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

CMD ["python", "-u", "dogwatch.py"]

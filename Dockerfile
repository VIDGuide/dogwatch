# dogwatch — Coral Edge TPU dog detector
# Uses Python 3.9 because pycoral wheels only support up to cp39

FROM python:3.9-slim-bookworm

# libedgetpu runtime (std = standard clock speed, good thermals)
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

# pycoral + tflite-runtime from Google Coral (cp39 linux x86_64)
#
# Dependency currency notes (see README "Known limitations" for the full
# writeup): pycoral/tflite_runtime are abandoned upstream and only ever
# shipped cp39 wheels, which is what pins the whole image to Python 3.9
# (EOL 2025-10-31). Everything below this line is bumped to the latest
# version that still supports cp39, and should be re-checked whenever the
# Python/pycoral constraint is revisited.
#
# numpy is pinned to 1.26.4 (the last 1.x release) rather than bumping to
# numpy 2.x: pycoral's compiled bindings (_pywrap_coral) are built against
# the numpy 1.x C ABI and break under numpy 2.x at runtime (confirmed on
# real Coral TPU hardware — pip's resolver alone won't catch this, since it
# only checks declared version ranges, not compiled ABI compatibility).
# This forces opencv-python-headless back to 4.9.0.80, the newest release
# in the numpy<2 line: 4.10.0/4.11.0 carry CVE-2025-53644 (heap buffer
# write via crafted JPEG) and 4.12.0+ requires numpy>=2, so 4.9.0.80 is the
# newest version that is both numpy-1.x-compatible and outside the CVE's
# affected range (4.10.0-4.11.0 only). Re-check this whole chain if the
# pycoral/numpy-2.x ABI break is ever resolved upstream.
RUN pip install --no-cache-dir \
    "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl" \
    "https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl" \
    paho-mqtt==2.1.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    shapely==2.0.6 \
    requests==2.32.4

COPY *.py /app/
WORKDIR /app

CMD ["python", "-u", "dogwatch.py"]

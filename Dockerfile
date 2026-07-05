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
RUN pip install --no-cache-dir \
    "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl" \
    "https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl" \
    'paho-mqtt<2' \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    shapely==2.0.6 \
    requests==2.32.3

COPY *.py /app/
WORKDIR /app

CMD ["python", "-u", "dogwatch.py"]

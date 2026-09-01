# dogwatch-notify — MQTT event notifier + Telegram + vision verification
#
# Separate from the detector container (which needs libedgetpu/Coral).
# This container handles: event-driven Telegram alerts with annotated
# snapshots, periodic HA dashboard stills, and cron-equivalent vision
# verification (Gemini/any OpenAI-compatible API).
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY dogwatch-notify.py /app/
COPY dogwatch-check.sh /app/
COPY dogwatch_check.py /app/
COPY dog-alarm.sh /app/
COPY find-dogs.py /app/
COPY find-dogs-mqtt.py /app/
COPY stats.py /app/
# Shared helper modules imported by the scripts above.
COPY dw_redact.py /app/
COPY image_quality.py /app/
COPY mqtt_auth.py /app/
RUN chmod +x /app/dogwatch-check.sh /app/dog-alarm.sh /app/find-dogs.py /app/find-dogs-mqtt.py

WORKDIR /app

# The notifier runs as a long-lived MQTT subscriber. The check script
# is invoked periodically by the entrypoint wrapper.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

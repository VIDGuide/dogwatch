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
RUN chmod +x /app/dogwatch-check.sh

WORKDIR /app

# The notifier runs as a long-lived MQTT subscriber. The check script
# is invoked periodically by the entrypoint wrapper.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

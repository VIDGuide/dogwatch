"""gemini_image_quality_check.py — ask Gemini vision to assess a snapshot's
image quality and whether it can identify a dog, as a second opinion
independent of the Coral TPU / ssd_mobilenet_v2 pipeline.

Useful for distinguishing "the detector missed a dog that was clearly
visible" from "the source frame itself was too degraded (motion smear,
compression artifacts, bad exposure) for any model to reliably work with" —
the latter turned out to be the actual explanation for a fence-camera
capture that produced zero detections during a July 2026 investigation (see
samples/README.md).

Reads the Gemini API key from ~/.openclaw/secrets.json at runtime, same
pattern as pipeline/dogwatch-check.sh — no credentials are hardcoded or
committed here.

Usage:
    python tests/gemini_image_quality_check.py samples/some_image.jpg

Requires network access to the Gemini API and a valid
models.providers.google.apiKey in ~/.openclaw/secrets.json (or set
DOGWATCH_VISION_API_KEY to override, matching the env var pipeline/
dogwatch-check.sh uses).
"""
import base64
import json
import os
import sys
import urllib.request

VISION_API_URL = os.environ.get(
    "DOGWATCH_VISION_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
)
VISION_MODEL = os.environ.get("DOGWATCH_VISION_MODEL", "gemini-3-flash-preview")


def _load_api_key() -> str:
    key = os.environ.get("DOGWATCH_VISION_API_KEY", "")
    if key:
        return key
    secrets_path = os.path.expanduser("~/.openclaw/secrets.json")
    try:
        with open(secrets_path) as f:
            secrets = json.load(f)
        return secrets["models"]["providers"]["google"]["apiKey"]
    except (OSError, KeyError) as exc:
        raise RuntimeError(
            f"No vision API key available — set DOGWATCH_VISION_API_KEY or "
            f"add models.providers.google.apiKey to {secrets_path}"
        ) from exc


def check_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = (
        "Look very carefully at this security camera image. Is there a dog "
        "anywhere in it? Describe exactly where (pixel region if you can "
        "estimate it) and how sharp/clear vs blurry it appears. Also tell "
        "me if the image itself looks sharp and well-focused overall, or "
        "blurry/motion-smeared/low quality. Be precise and thorough, this "
        "is for debugging a computer vision pipeline."
    )
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 2000,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(VISION_API_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {_load_api_key()}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def main():
    if len(sys.argv) < 2:
        print("Usage: gemini_image_quality_check.py <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        try:
            print(check_image(path))
        except Exception as exc:
            print(f"  ERROR: {exc}")
        print()


if __name__ == "__main__":
    main()

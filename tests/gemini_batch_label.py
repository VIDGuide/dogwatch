"""gemini_batch_label.py — batch-label a set of archived debug capture
snapshots using Gemini vision, producing structured (dog present / absent /
uncertain, plus notes) results rather than free-text per image.

Intended for turning a pile of past event snapshots (e.g. debug_captures/,
or the old /tmp leak once archived there — see README "Debug capture") into
labeled validation data: how many of the events that actually fired were
real dogs vs false positives, and why.

Reads the Gemini API key from ~/.openclaw/secrets.json at runtime (or
DOGWATCH_VISION_API_KEY), same pattern as pipeline/dogwatch-check.sh and
tests/gemini_image_quality_check.py — no credentials hardcoded or committed.

Usage:
    # Label a specific list of files
    python tests/gemini_batch_label.py img1.jpg img2.jpg ...

    # Label every file in a directory (non-recursive)
    python tests/gemini_batch_label.py --dir debug_captures/rear-east

    # Label a random sample of N files from a directory
    python tests/gemini_batch_label.py --dir debug_captures/camera --sample 10

Writes a CSV summary to stdout (or --out <file>) with columns:
    path, dog, confidence, notes
"""
import argparse
import base64
import csv
import glob
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

VISION_API_URL = os.environ.get(
    "DOGWATCH_VISION_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
)
VISION_MODEL = os.environ.get("DOGWATCH_VISION_MODEL", "gemini-3-flash-preview")

PROMPT = (
    "You are validating output from a backyard dog-detection camera system. "
    "Look carefully at this image and assess whether a dog is genuinely "
    "present. Respond with ONLY a single JSON object and nothing else — no "
    "preamble, no explanation, no markdown code fences, just the raw JSON "
    "on its own, in exactly this form:\n"
    '{"dog": "YES"|"NO"|"UNCERTAIN", "confidence": "HIGH"|"MEDIUM"|"LOW", '
    '"notes": "<one short sentence, under 15 words>"}\n'
    "dog=YES only if you can identify clear canine features (head, legs, "
    "ears, tail, fur). dog=NO if the image clearly shows no dog (e.g. "
    "empty yard, or the region a detector might have flagged is actually "
    "fence/shadow/debris/dirt). dog=UNCERTAIN if genuinely ambiguous. "
    "confidence reflects how sure you are of the dog judgement, not image "
    "quality. Keep notes very short."
)


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


def label_image(image_path: str, api_key: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(VISION_API_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    last_exc = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 and attempt < 2:
                time.sleep(5 * (attempt + 1))  # back off harder each retry
                continue
            raise
    else:
        raise last_exc

    text = result["choices"][0]["message"]["content"]
    # Some responses wrap the JSON in a markdown code fence or add a
    # preamble sentence despite the prompt asking for raw JSON only —
    # extract the {...} substring rather than assuming the whole string
    # parses cleanly.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return {"dog": "UNCERTAIN", "confidence": "LOW", "notes": f"non-JSON response: {text[:200]}"}
        else:
            return {"dog": "UNCERTAIN", "confidence": "LOW", "notes": f"non-JSON response: {text[:200]}"}
    return {
        "dog": str(parsed.get("dog", "UNCERTAIN")).upper(),
        "confidence": str(parsed.get("confidence", "LOW")).upper(),
        "notes": str(parsed.get("notes", "")),
    }


def collect_paths(args) -> list:
    if args.dir:
        paths = sorted(
            glob.glob(os.path.join(args.dir, "*.jpg"))
            + glob.glob(os.path.join(args.dir, "*.jpeg"))
            + glob.glob(os.path.join(args.dir, "*.png"))
        )
        if args.sample and args.sample < len(paths):
            paths = sorted(random.sample(paths, args.sample))
        return paths
    return args.paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Specific image files to label")
    parser.add_argument("--dir", help="Label every image in this directory instead")
    parser.add_argument("--sample", type=int, help="Random sample size when using --dir")
    parser.add_argument("--out", help="Write CSV to this file instead of stdout")
    parser.add_argument("--delay", type=float, default=2.5,
                        help="Seconds to sleep between API calls (default 2.5, "
                             "to stay under Gemini's free-tier rate limit)")
    args = parser.parse_args()

    paths = collect_paths(args)
    if not paths:
        print("No images to label — pass file paths or --dir <directory>", file=sys.stderr)
        sys.exit(1)

    api_key = _load_api_key()

    rows = []
    for i, path in enumerate(paths):
        print(f"[{i + 1}/{len(paths)}] {path}", file=sys.stderr)
        try:
            result = label_image(path, api_key)
        except Exception as exc:
            result = {"dog": "ERROR", "confidence": "", "notes": str(exc)}
        rows.append({"path": path, **result})
        if i < len(paths) - 1:
            time.sleep(args.delay)

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=["path", "dog", "confidence", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()

    dog_count = sum(1 for r in rows if r["dog"] == "YES")
    no_count = sum(1 for r in rows if r["dog"] == "NO")
    uncertain_count = sum(1 for r in rows if r["dog"] == "UNCERTAIN")
    error_count = sum(1 for r in rows if r["dog"] == "ERROR")
    print(
        f"\nSummary: {len(rows)} labeled — YES={dog_count} NO={no_count} "
        f"UNCERTAIN={uncertain_count} ERROR={error_count}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

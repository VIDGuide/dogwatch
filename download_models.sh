#!/usr/bin/env bash
# download_models.sh — pull Edge-TPU-compiled models and COCO labels
# from Google's google-coral/test_data repo.
#
# Usage:
#   ./download_models.sh [efficientdet|mobilenet|mobiledet|all]
#
#   efficientdet  EfficientDet-Lite3 512×512 (recommended, ~38 mAP)
#   mobilenet     SSD MobileNet V2 300×300 (fastest, lower accuracy)
#   mobiledet     SSDLite MobileDet 320×320  (QAT-trained, middle ground)
#   all           Download all three + labels (default if no arg given)
#
# All models use the shared COCO 90-class label file; 'dog' is class 18.
# Labels are always downloaded (once) regardless of which model you pick.

set -euo pipefail

# Pinned to a commit, not to `master`, and every artifact is checksum-verified.
#
# What this closes: these files are fetched over the network and then loaded
# straight into the detector as the model it runs inference with. Pointing at
# `master` meant the bytes could change under us at any time, and nothing
# checked them, so a repo compromise or a MITM on the download became a
# silent swap of the model. The digest-pinned libedgetpu .deb in the
# Dockerfiles already works this way; this brings the models in line.
#
# To move to a newer upstream revision: update REPO_REF, run this script with
# a clean OUTDIR, and paste the printed sums into SHA256 below.
REPO_REF="${REPO_REF:-104342d2d3480b3e66203073dac24f4e2dbb4c41}"
REPO="https://raw.githubusercontent.com/google-coral/test_data/$REPO_REF"
OUTDIR="${OUTDIR:-models}"

MODELS=(
  "efficientdet:efficientdet_lite3_512_ptq_edgetpu.tflite"
  "mobilenet:ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
  "mobiledet:ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
)

# filename -> expected SHA256, as fetched from REPO_REF above.
SHA256_coco_labels_txt="dc183f003fc753c4c43fae6fdf7f387559449573f13fa32e517fb7453fd380f1"
SHA256_efficientdet="4f98f09872404d9e28744d3ff694d8427a968ddb467a9aec0ac861bd9f3dba14"
SHA256_mobilenet="b94e2d58222c32f31062c7604e10488e2aba9259ab77462039476a3ba4597fef"
SHA256_mobiledet="b69e508ef2a670e06b80bd3e5559a827d5cd8d557c95d5e332cbf1d31d434a2e"

# sha256 of a file, on either GNU coreutils or macOS.
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo "ERROR: neither sha256sum nor shasum found — cannot verify downloads" >&2
    return 1
  fi
}

# verify <file> <expected-sha256> — deletes the file and fails on mismatch, so
# a bad download can never be left behind for the detector to load.
verify() {
  local file="$1" expected="$2" actual
  actual="$(sha256_of "$file")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$file"
    echo "" >&2
    echo "ERROR: checksum mismatch for $(basename "$file") — download discarded." >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    echo "  If you deliberately changed REPO_REF, update the SHA256_* values." >&2
    return 1
  fi
}

usage() {
  sed -n '2,14p' "$0" | sed 's/^# //'
  exit 0
}

# Resolve the selection.
SELECTION="${1:-all}"
case "$SELECTION" in
  -h|--help) usage ;;
  efficientdet|mobilenet|mobiledet|all) ;;
  *)
    echo "Unknown model: '$SELECTION'"
    echo "Usage: $0 [efficientdet|mobilenet|mobiledet|all]"
    exit 1
    ;;
esac

mkdir -p "$OUTDIR"

# Labels — always fetch to be safe (idempotent small file).
LABEL_FILE="$OUTDIR/coco_labels.txt"
if [[ -f "$LABEL_FILE" ]]; then
  echo "[labels] $LABEL_FILE already exists (skipping)"
else
  echo "[labels] Downloading coco_labels.txt"
  curl -fL -o "$LABEL_FILE" "$REPO/coco_labels.txt"
  verify "$LABEL_FILE" "$SHA256_coco_labels_txt"
  echo "[labels] Done — checksum OK"
fi

download_model() {
  local label="$1" file="$2"
  local dest="$OUTDIR/$file"
  if [[ -f "$dest" ]]; then
    echo "[$label] $file already exists (skipping)"
    return
  fi
  # Indirect expansion picks SHA256_efficientdet / _mobilenet / _mobiledet.
  local var="SHA256_$label"
  local expected="${!var:-}"
  if [[ -z "$expected" ]]; then
    echo "[$label] ERROR: no expected checksum recorded for $file" >&2
    return 1
  fi
  echo "[$label] Downloading $file ..."
  curl -fL -o "$dest" "$REPO/$file"
  verify "$dest" "$expected"
  local size
  size=$(du -h "$dest" | cut -f1)
  echo "[$label] Done — $size, checksum OK"
}

for entry in "${MODELS[@]}"; do
  label="${entry%%:*}"
  file="${entry#*:}"
  if [[ "$SELECTION" == "all" ]] || [[ "$SELECTION" == "$label" ]]; then
    download_model "$label" "$file"
  fi
done

# Quick summary.
echo ""
echo "Models in $OUTDIR/:"
ls -lh "$OUTDIR"/*.tflite 2>/dev/null || echo "  (none)"
echo ""
echo "Labels in $OUTDIR/:"
ls -lh "$OUTDIR"/*.txt 2>/dev/null

if [[ "$SELECTION" == "all" ]] || [[ "$SELECTION" == "efficientdet" ]]; then
  echo ""
  echo "Tip: point config.json to models/efficientdet_lite3_512_ptq_edgetpu.tflite"
  echo "     for the recommended EfficientDet-Lite3 (512×512, ~38 mAP)."
fi

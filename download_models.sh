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

REPO="https://raw.githubusercontent.com/google-coral/test_data/master"
OUTDIR="${OUTDIR:-models}"

MODELS=(
  "efficientdet:efficientdet_lite3_512_ptq_edgetpu.tflite"
  "mobilenet:ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
  "mobiledet:ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
)

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
  echo "[labels] Done"
fi

download_model() {
  local label="$1" file="$2"
  local dest="$OUTDIR/$file"
  if [[ -f "$dest" ]]; then
    echo "[$label] $file already exists (skipping)"
    return
  fi
  echo "[$label] Downloading $file ..."
  curl -fL -o "$dest" "$REPO/$file"
  local size
  size=$(du -h "$dest" | cut -f1)
  echo "[$label] Done — $size"
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

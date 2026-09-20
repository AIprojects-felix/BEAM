#!/usr/bin/env bash
# End-to-end pipeline check: dummy data -> train (3 epochs) -> evaluate -> predict.
# Run from the repo root (the directory containing train.py).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

OUTPUT_DIR=output_smoke
SAMPLE_DIR=data/sample
LABELS=$SAMPLE_DIR/labels.csv

echo "=== [1/4] Prepare dummy dataset ==="
if [[ ! -f "$LABELS" ]]; then
    python data/make_sample_data.py --output_dir "$SAMPLE_DIR" --n_pos 50 --n_neg 50
else
    echo "Found existing $LABELS, skipping generation"
fi

echo ""
echo "=== [2/4] Train (pipeline check) ==="
rm -rf "$OUTPUT_DIR"
python train.py \
    --config config.yaml \
    --data_path "$SAMPLE_DIR" \
    --labels_file "$LABELS" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 100 \
    --batch_size 4

# Locate best_model.pth
BEST_CKPT=$(find "$OUTPUT_DIR" -name 'best_model.pth' | head -1)
if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: best_model.pth not found"
    exit 1
fi
echo "Got checkpoint: $BEST_CKPT"

echo ""
echo "=== [3/4] Evaluate ==="
python evaluate.py \
    --config config.yaml \
    --model_path "$BEST_CKPT" \
    --test_data "$SAMPLE_DIR" \
    --labels_file "$LABELS" \
    --output_dir "$OUTPUT_DIR/eval"

echo ""
echo "=== [4/4] Predict ==="
python predict.py \
    --config config.yaml \
    --model_path "$BEST_CKPT" \
    --mri_dir "$SAMPLE_DIR/patient_001"

python predict.py \
    --config config.yaml \
    --model_path "$BEST_CKPT" \
    --batch_mode \
    --data_dir "$SAMPLE_DIR" \
    --output_dir "$OUTPUT_DIR/pred"

echo ""
echo "=== Pipeline check PASSED ==="

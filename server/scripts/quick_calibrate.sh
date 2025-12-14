#!/bin/bash
# Quick calibration training script
# Usage: ./quick_calibrate.sh CHECKPOINT_IN [YOUTUBE_URLS...]

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 CHECKPOINT_IN YOUTUBE_URL1 [YOUTUBE_URL2 ...]"
    echo "Example: $0 checkpoint.pth https://youtube.com/watch?v=VIDEO1 https://youtube.com/watch?v=VIDEO2"
    exit 1
fi

CHECKPOINT_IN=$1
shift
YOUTUBE_URLS=("$@")

CALIBRATION_DIR="./calibration_data"
CHECKPOINT_OUT="checkpoint_calibrated.pth"

echo "============================================================"
echo "Quick Calibration Training"
echo "============================================================"
echo "Input checkpoint: $CHECKPOINT_IN"
echo "YouTube URLs: ${#YOUTUBE_URLS[@]} videos"
echo "Output directory: $CALIBRATION_DIR"
echo "Output checkpoint: $CHECKPOINT_OUT"
echo "============================================================"
echo ""

# Step 1: Prepare data
echo "Step 1: Downloading videos and extracting faces..."
python scripts/prepare_youtube_calibration.py \
    --urls "${YOUTUBE_URLS[@]}" \
    --output_dir "$CALIBRATION_DIR" \
    --frame_interval 30

# Step 2: Train calibration
echo ""
echo "Step 2: Running calibration training..."
python train/train_calibration.py \
    --data_dir "$CALIBRATION_DIR" \
    --checkpoint_in "$CHECKPOINT_IN" \
    --checkpoint_out "$CHECKPOINT_OUT" \
    --epochs 1 \
    --batch_size 16 \
    --lr 1e-4

echo ""
echo "============================================================"
echo "[OK] Calibration complete!"
echo "  Calibrated checkpoint: $CHECKPOINT_OUT"
echo ""
echo "To use the calibrated checkpoint:"
echo "  export MODEL_CHECKPOINT=$CHECKPOINT_OUT"
echo "  python server.py"
echo "============================================================"


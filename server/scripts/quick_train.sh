#!/bin/bash
# Quick training script for FaceForensics dataset

# Default values
DATA_DIR="${1:-./data/faceforensics}"
CHECKPOINT_OUT="${2:-./checkpoint.pth}"
EPOCHS="${3:-10}"
BATCH_SIZE="${4:-32}"

echo "=========================================="
echo "FaceForensics Training Script"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Checkpoint output: $CHECKPOINT_OUT"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please provide a valid path to your FaceForensics dataset"
    exit 1
fi

# Run training
python train/train_faceforensics.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_out "$CHECKPOINT_OUT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-4 \
    --val_split 0.2 \
    --freeze_backbone

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint saved to: $CHECKPOINT_OUT"
echo "=========================================="


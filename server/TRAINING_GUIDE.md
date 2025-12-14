# Training Guide for FaceForensics Dataset

This guide explains how to train the Xception model on FaceForensics dataset for deepfake detection.

## Prerequisites

1. **FaceForensics Dataset**: Download from [FaceForensics++](https://github.com/ondyari/FaceForensics)
2. **Python Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The training script supports multiple dataset formats:

### Format 1: Real/Fake Folders
```
data/
  real/
    image1.jpg
    image2.jpg
  fake/
    image3.jpg
    image4.jpg
```

### Format 2: Manipulation-Based Structure
```
data/
  Deepfakes/
    real/
    fake/
  FaceSwap/
    real/
    fake/
  NeuralTextures/
    real/
    fake/
```

### Format 3: CSV Labels
```
data/
  images/
    image1.jpg
    image2.jpg
  labels.csv  # filename,label format
```

## Step 1: Prepare Your Dataset

If your dataset is not in the expected format, use the preparation script:

```bash
# Generate CSV labels from directory structure
python scripts/prepare_faceforensics.py \
  --mode csv \
  --source_dir /path/to/your/data \
  --output_csv labels.csv

# Or organize by manipulation method
python scripts/prepare_faceforensics.py \
  --mode organize \
  --source_dir /path/to/source \
  --target_dir /path/to/organized \
  --val_split 0.2
```

## Step 2: Train the Model

### Basic Training (All Layers)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/faceforensics/data \
  --checkpoint_out checkpoint.pth \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_split 0.2
```

### Training with Frozen Backbone (Faster, Less Memory)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/faceforensics/data \
  --checkpoint_out checkpoint.pth \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --val_split 0.2 \
  --freeze_backbone
```

### Training with CSV Labels
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/images \
  --labels_csv labels.csv \
  --checkpoint_out checkpoint.pth \
  --epochs 10 \
  --batch_size 32
```

## Training Parameters

- `--data_dir`: Path to your dataset root directory
- `--labels_csv`: Optional CSV file with filename,label mapping
- `--checkpoint_out`: Output path for trained checkpoint (default: `checkpoint.pth`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--val_split`: Validation split ratio 0.0-1.0 (default: 0.2)
- `--freeze_backbone`: Freeze backbone, only train classifier (faster training)
- `--device`: Device to use (cuda/cpu, auto-detected if not specified)
- `--seed`: Random seed for reproducibility (default: 42)

## Step 3: Use the Trained Model

After training, the checkpoint will be saved. The server will automatically find it if placed in:
- `checkpoint.pth` (current directory)
- `checkpoints/checkpoint.pth`
- `checkpoints/best.pth`
- Or set `MODEL_CHECKPOINT` environment variable

```bash
# Set checkpoint path via environment variable
export MODEL_CHECKPOINT=/path/to/checkpoint.pth

# Or copy checkpoint to server directory
cp checkpoint.pth server/checkpoint.pth

# Start the server
python server.py
```

## Monitoring Training

The training script will output:
- Training loss and accuracy per epoch
- Validation loss and accuracy (if val_split > 0)
- Best model checkpoint saved automatically

Example output:
```
Epoch 1/10
Training: 100%|████████| 125/125 [02:30<00:00,  1.20s/it]
Train Loss: 0.523456, Train Acc: 0.7234
Validating: 100%|████████| 32/32 [00:15<00:00,  2.10it/s]
Val Loss: 0.456789, Val Acc: 0.7890
✓ Saved best checkpoint (val_loss=0.456789, val_acc=0.7890)
```

## Tips

1. **Start with frozen backbone**: Use `--freeze_backbone` for faster initial training
2. **Adjust batch size**: Larger batch sizes (64-128) if you have enough GPU memory
3. **Learning rate**: Start with 1e-4, reduce to 1e-5 if loss plateaus
4. **Validation split**: Use 0.2 (20%) for validation to monitor overfitting
5. **Epochs**: 10-20 epochs usually sufficient, monitor validation loss to avoid overfitting

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 16 or 8)
- Use `--freeze_backbone` to reduce memory usage

### No Images Found
- Check dataset path is correct
- Verify images are in supported formats (.jpg, .jpeg, .png)
- Check folder structure matches expected format

### Poor Performance
- Ensure dataset is balanced (similar number of real/fake samples)
- Try unfreezing backbone for full fine-tuning
- Increase number of epochs
- Check data quality and preprocessing


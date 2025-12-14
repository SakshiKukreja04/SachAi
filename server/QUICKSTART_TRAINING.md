# Quick Start: Training the Model

## Prerequisites

1. **FaceForensics Dataset**: You need the FaceForensics++ dataset with face crops
2. **Python Environment**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Training (3 Steps)

### Step 1: Organize Your Dataset

Your dataset should be in one of these formats:

**Option A: Real/Fake Folders**
```
your_data/
  real/
    image1.jpg
    image2.jpg
  fake/
    image3.jpg
    image4.jpg
```

**Option B: Manipulation-Based**
```
your_data/
  Deepfakes/
    real/
    fake/
  FaceSwap/
    real/
    fake/
```

### Step 2: Run Training

**On Windows (PowerShell):**
```powershell
cd server
python train/train_faceforensics.py --data_dir "C:\path\to\your\data" --checkpoint_out checkpoint.pth --epochs 10 --batch_size 32 --val_split 0.2
```

**On Linux/Mac:**
```bash
cd server
python train/train_faceforensics.py --data_dir /path/to/your/data --checkpoint_out checkpoint.pth --epochs 10 --batch_size 32 --val_split 0.2
```

**Or use the quick script:**
```bash
# Windows PowerShell
.\scripts\quick_train.ps1 -DataDir "C:\path\to\data" -CheckpointOut checkpoint.pth

# Linux/Mac
bash scripts/quick_train.sh /path/to/data checkpoint.pth
```

### Step 3: Use the Trained Model

After training completes, the checkpoint will be saved. The server will automatically find it:

```bash
# Start the server (it will auto-detect checkpoint.pth)
python server.py
```

Or set the checkpoint path explicitly:
```bash
# Windows
$env:MODEL_CHECKPOINT="checkpoint.pth"
python server.py

# Linux/Mac
export MODEL_CHECKPOINT=checkpoint.pth
python server.py
```

## Training Options

### Fast Training (Recommended for First Run)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/data \
  --checkpoint_out checkpoint.pth \
  --epochs 10 \
  --batch_size 32 \
  --freeze_backbone  # Only trains classifier, faster
```

### Full Fine-Tuning (Better Results, Slower)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/data \
  --checkpoint_out checkpoint.pth \
  --epochs 20 \
  --batch_size 16 \
  # No --freeze_backbone, trains all layers
```

## What to Expect

During training, you'll see:
```
Epoch 1/10
Training: 100%|████████| 125/125 [02:30<00:00]
Train Loss: 0.523456, Train Acc: 0.7234
Validating: 100%|████████| 32/32 [00:15<00:00]
Val Loss: 0.456789, Val Acc: 0.7890
✓ Saved best checkpoint (val_loss=0.456789, val_acc=0.7890)
```

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use frozen backbone: `--freeze_backbone`

**No Images Found?**
- Check your data directory path
- Verify images are .jpg, .jpeg, or .png
- Check folder structure matches expected format

**Poor Results?**
- Try more epochs: `--epochs 20`
- Remove `--freeze_backbone` for full fine-tuning
- Check dataset balance (similar real/fake counts)

## Next Steps

After training:
1. Test the model by uploading a video through the web interface
2. Check the analysis results - fake videos should show high confidence scores
3. If results are poor, retrain with more epochs or different hyperparameters

For detailed information, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)


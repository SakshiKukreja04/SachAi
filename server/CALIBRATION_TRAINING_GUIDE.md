# Quick Calibration Training Guide

## Overview

This guide helps you perform **quick calibration training** with 5-10 real YouTube videos to improve your model's accuracy without overfitting. This is perfect for hackathons or quick improvements.

**What it does:**
- Downloads 5-10 real YouTube videos
- Extracts faces from videos
- Trains for 1 epoch with frozen backbone
- Nudges scores slightly to reduce false positives
- Improves overall accuracy by 2-5%

**Time required:** ~30-60 minutes (depending on video length and CPU/GPU)

---

## Prerequisites

### 1. Install Dependencies

```bash
cd server
pip install yt-dlp opencv-python pillow tqdm
```

Or add to `requirements.txt`:
```
yt-dlp>=2023.0.0
opencv-python>=4.5.0
Pillow>=9.0.0
tqdm>=4.60.0
```

### 2. Have an Existing Checkpoint

You need a trained model checkpoint to calibrate. If you don't have one:
- Use a pre-trained checkpoint, OR
- Train a basic model first (see `TRAINING_GUIDE.md`)

---

## Step-by-Step Process

### Step 1: Prepare YouTube Videos

#### Option A: Use the Preparation Script (Recommended)

**Linux/Mac (bash):**
```bash
cd server

# Download and extract faces from 5-10 YouTube videos
python scripts/prepare_youtube_calibration.py \
  --urls \
    "https://www.youtube.com/watch?v=VIDEO1" \
    "https://www.youtube.com/watch?v=VIDEO2" \
    "https://www.youtube.com/watch?v=VIDEO3" \
    "https://www.youtube.com/watch?v=VIDEO4" \
    "https://www.youtube.com/watch?v=VIDEO5" \
  --output_dir ./calibration_data \
  --frame_interval 30
```

**Windows PowerShell:**
```powershell
cd server

# Option 1: Single line (easiest)
python scripts/prepare_youtube_calibration.py --urls "https://www.youtube.com/watch?v=VIDEO1" "https://www.youtube.com/watch?v=VIDEO2" "https://www.youtube.com/watch?v=VIDEO3" "https://www.youtube.com/watch?v=VIDEO4" "https://www.youtube.com/watch?v=VIDEO5" --output_dir ./calibration_data --frame_interval 30

# Option 2: Multi-line with backticks (PowerShell line continuation)
python scripts/prepare_youtube_calibration.py `
  --urls `
    "https://www.youtube.com/watch?v=VIDEO1" `
    "https://www.youtube.com/watch?v=VIDEO2" `
    "https://www.youtube.com/watch?v=VIDEO3" `
    "https://www.youtube.com/watch?v=VIDEO4" `
    "https://www.youtube.com/watch?v=VIDEO5" `
  --output_dir ./calibration_data `
  --frame_interval 30
```

**Parameters:**
- `--urls`: List of YouTube video URLs (5-10 recommended)
- `--output_dir`: Where to save extracted faces (default: `./calibration_data`)
- `--frame_interval`: Extract every Nth frame (default: 30 = ~1 frame per second)

**Example with 10 videos:**

**Linux/Mac (bash - uses backslashes):**
```bash
python scripts/prepare_youtube_calibration.py \
  --urls \
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    "https://www.youtube.com/watch?v=9bZkp7q19f0" \
    "https://www.youtube.com/watch?v=kJQP7kiw5Fk" \
    "https://www.youtube.com/watch?v=fJ9rUzIMcZQ" \
    "https://www.youtube.com/watch?v=OPf0YbXqDm0" \
  --output_dir ./calibration_data
```

**Windows PowerShell (uses backticks or single line):**
```powershell
# Option 1: Single line (easiest - copy and paste this)
python scripts/prepare_youtube_calibration.py --urls "https://www.youtube.com/shorts/3XPuGqqWLtk" "https://www.youtube.com/watch?v=xdWrK4oQpgg" "https://www.youtube.com/watch?v=ATsPTWqyWHE" "https://www.youtube.com/watch?v=OsO0MqHG3nc" "https://www.youtube.com/watch?v=6JuRuUUG3ag" --output_dir ./calibration_data

# Option 2: Multi-line with backticks (PowerShell line continuation)
python scripts/prepare_youtube_calibration.py `
  --urls `
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ" `
    "https://www.youtube.com/watch?v=9bZkp7q19f0" `
    "https://www.youtube.com/watch?v=Dmvd3qF7F4g" `
    "https://www.youtube.com/watch?v=xdWrK4oQpgg" `
    "https://www.youtube.com/watch?v=ATsPTWqyWHE" `
    "https://www.youtube.com/watch?v=OsO0MqHG3nc" `
    "https://www.youtube.com/watch?v=6JuRuUUG3ag" `
  --output_dir ./calibration_data
```

#### Option B: Manual Preparation

1. **Download videos manually** using yt-dlp:
   ```bash
   yt-dlp -f "best[ext=mp4]" -o "video_01.mp4" "YOUTUBE_URL"
   ```

2. **Extract faces** using the face extraction script:
   ```bash
   python scripts/extract_faceforensics_faces.py \
     --video_dir ./videos \
     --output_dir ./calibration_data/real \
     --label 0
   ```

### Step 2: Verify Data Structure

After preparation, you should have:

```
calibration_data/
  videos/
    video_01/
      video.mp4
    video_02/
      video.mp4
    ...
  real/
    face_00000.jpg
    face_00001.jpg
    face_00002.jpg
    ...
```

**Check:**
```bash
# Count extracted faces
ls calibration_data/real/*.jpg | wc -l

# Should have at least 100-500 faces for good calibration
```

### Step 3: Run Calibration Training

**Important:** You need an existing checkpoint to calibrate. If you don't have one:
- **Option A**: Train a basic model first (see `TRAINING_GUIDE.md`)
- **Option B**: Use pretrained Xception (less effective, but works)

**Linux/Mac (bash):**
```bash
cd server

# With existing checkpoint (recommended)
python train/train_calibration.py \
  --data_dir ./calibration_data \
  --checkpoint_in checkpoint.pth \
  --checkpoint_out checkpoint_calibrated.pth \
  --epochs 1 \
  --batch_size 16 \
  --lr 1e-4

# Without checkpoint (starts from pretrained - less effective)
python train/train_calibration.py \
  --data_dir ./calibration_data \
  --checkpoint_out checkpoint_calibrated.pth \
  --epochs 1 \
  --batch_size 16 \
  --lr 1e-4
```

**Windows PowerShell:**
```powershell
cd server

# Option 1: With existing checkpoint (recommended) - Single line
python train/train_calibration.py --data_dir ./calibration_data --checkpoint_in checkpoint.pth --checkpoint_out checkpoint_calibrated.pth --epochs 1 --batch_size 16 --lr 1e-4

# Option 2: Without checkpoint (starts from pretrained) - Single line
python train/train_calibration.py --data_dir ./calibration_data --checkpoint_out checkpoint_calibrated.pth --epochs 1 --batch_size 16 --lr 1e-4

# Option 3: Multi-line with backticks (PowerShell uses backticks, not backslashes)
python train/train_calibration.py `
  --data_dir ./calibration_data `
  --checkpoint_in checkpoint.pth `
  --checkpoint_out checkpoint_calibrated.pth `
  --epochs 1 `
  --batch_size 16 `
  --lr 1e-4
```

**Parameters:**
- `--data_dir`: Path to calibration data (should contain `real/` folder)
- `--checkpoint_in`: Your existing trained checkpoint
- `--checkpoint_out`: Output path for calibrated checkpoint
- `--epochs`: Number of epochs (default: 1 - recommended for calibration)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)

**What happens:**
1. Loads your existing checkpoint
2. Freezes backbone (only trains classifier head)
3. Trains for 1 epoch on real videos
4. Saves calibrated checkpoint

**Expected output:**
```
============================================================
Calibration Training
============================================================
Data directory: ./calibration_data
Input checkpoint: checkpoint.pth
Output checkpoint: checkpoint_calibrated.pth
Epochs: 1
Batch size: 16
Learning rate: 0.0001
Device: cpu
============================================================

Loading model from checkpoint...
Freezing backbone, training only classifier head...
  Unfrozen: fc.weight
  Unfrozen: fc.bias
Trainable parameters: 2,049

Loading dataset...
Found 342 real images
Dataset size: 342 images

Starting training (1 epoch(s))...

Epoch 1/1
Training: 100%|████████| 22/22 [00:45<00:00,  2.05s/it]

Epoch 1 Results:
  Train Loss: 0.234567
  Train Acc:  0.9234 (92.34%)
  [OK] Saved checkpoint to checkpoint_calibrated.pth

============================================================
[OK] Calibration training complete!
  Final checkpoint: checkpoint_calibrated.pth
  Final loss: 0.234567
  Final accuracy: 0.9234 (92.34%)
============================================================
```

### Step 4: Use Calibrated Checkpoint

After calibration, use the new checkpoint in your server:

```bash
# Set environment variable
export MODEL_CHECKPOINT=./checkpoint_calibrated.pth

# Or copy to default location
cp checkpoint_calibrated.pth checkpoint.pth

# Start server
python server.py
```

---

## Tips for Best Results

### 1. Video Selection

**Choose diverse real videos:**
- Different people (various ages, genders, ethnicities)
- Different lighting conditions
- Different video qualities
- Different speaking styles
- Mix of close-ups and medium shots

**Avoid:**
- Videos with heavy filters/effects
- Videos with multiple faces (unless you want to extract all)
- Very short videos (< 30 seconds)

### 2. Frame Extraction

**Frame interval:**
- `--frame_interval 30`: ~1 frame per second (good for most videos)
- `--frame_interval 60`: ~0.5 fps (faster, fewer faces)
- `--frame_interval 15`: ~2 fps (slower, more faces)

**Recommended:** Start with 30, adjust based on video length

### 3. Training Parameters

**Epochs:**
- **1 epoch**: Recommended for calibration (prevents overfitting)
- **2-3 epochs**: If you have more data (500+ faces)
- **Never > 5 epochs**: Risk of overfitting on small dataset

**Learning rate:**
- `1e-4`: Default, good for most cases
- `5e-5`: More conservative (slower learning)
- `2e-4`: More aggressive (faster learning, risk of overfitting)

**Batch size:**
- **CPU**: 8-16 (smaller = less memory)
- **GPU**: 16-32 (can handle larger batches)

### 4. Dataset Size

**Minimum:**
- 100 faces: Works but limited improvement
- 200-300 faces: Good for calibration
- 500+ faces: Best results

**Maximum:**
- 1000-2000 faces: Still good with 1 epoch
- > 2000 faces: Consider 2 epochs

---

## Troubleshooting

### Issue: "yt-dlp not available"

**Solution:**
```bash
pip install yt-dlp
```

### Issue: "No images found in dataset"

**Check:**
1. Verify `calibration_data/real/` exists
2. Check if images are `.jpg` or `.png`
3. Run: `ls calibration_data/real/*.jpg | wc -l`

### Issue: "Checkpoint not found"

**Solution:**
- Make sure you have an existing checkpoint
- Use absolute path: `--checkpoint_in /full/path/to/checkpoint.pth`

### Issue: "Out of memory"

**Solution:**
- Reduce batch size: `--batch_size 8`
- Use CPU: `--device cpu`
- Extract fewer frames: `--frame_interval 60`

### Issue: Training loss not decreasing

**Possible causes:**
- Learning rate too low: Try `--lr 2e-4`
- Dataset too small: Add more videos
- Checkpoint already well-trained: Calibration may not help much

---

## Expected Improvements

After calibration, you should see:

1. **Reduced false positives** for real videos
2. **Better accuracy** on real YouTube videos (2-5% improvement)
3. **More calibrated scores** (less overconfident predictions)

**Before calibration:**
- Real video: 45% fake probability (false positive)
- Real video: 38% fake probability (false positive)

**After calibration:**
- Real video: 25% fake probability (correct)
- Real video: 22% fake probability (correct)

---

## Advanced: Combining with Existing Data

If you have existing training data, you can combine it:

```bash
# Combine calibration data with existing dataset
cp -r calibration_data/real/* existing_dataset/real/
cp -r calibration_data/real/* existing_dataset/real/

# Then train normally
python train/train_faceforensics.py \
  --data_dir existing_dataset \
  --checkpoint_in checkpoint.pth \
  --checkpoint_out checkpoint_combined.pth \
  --epochs 5
```

---

## Quick Reference

### Option A: Manual Steps

**Linux/Mac (bash):**
```bash
# 1. Prepare data (5-10 videos)
python scripts/prepare_youtube_calibration.py \
  --urls "URL1" "URL2" "URL3" "URL4" "URL5" \
  --output_dir ./calibration_data

# 2. Train calibration (1 epoch)
python train/train_calibration.py \
  --data_dir ./calibration_data \
  --checkpoint_in checkpoint.pth \
  --checkpoint_out checkpoint_calibrated.pth \
  --epochs 1

# 3. Use calibrated checkpoint
export MODEL_CHECKPOINT=./checkpoint_calibrated.pth
python server.py
```

**Windows PowerShell:**
```powershell
# 1. Prepare data (5-10 videos) - Single line
python scripts/prepare_youtube_calibration.py --urls "URL1" "URL2" "URL3" "URL4" "URL5" --output_dir ./calibration_data

# 2. Train calibration (1 epoch)
python train/train_calibration.py --data_dir ./calibration_data --checkpoint_in checkpoint.pth --checkpoint_out checkpoint_calibrated.pth --epochs 1

# 3. Use calibrated checkpoint
$env:MODEL_CHECKPOINT="./checkpoint_calibrated.pth"
python server.py
```

### Option B: Quick Script (One Command)

**Linux/Mac:**
```bash
chmod +x scripts/quick_calibrate.sh
./scripts/quick_calibrate.sh checkpoint.pth \
  "https://youtube.com/watch?v=VIDEO1" \
  "https://youtube.com/watch?v=VIDEO2" \
  "https://youtube.com/watch?v=VIDEO3" \
  "https://youtube.com/watch?v=VIDEO4" \
  "https://youtube.com/watch?v=VIDEO5"
```

**Windows PowerShell:**
```powershell
.\scripts\quick_calibrate.ps1 `
  -CheckpointIn checkpoint.pth `
  -YouTubeUrls @(
    "https://youtube.com/watch?v=VIDEO1",
    "https://youtube.com/watch?v=VIDEO2",
    "https://youtube.com/watch?v=VIDEO3",
    "https://youtube.com/watch?v=VIDEO4",
    "https://youtube.com/watch?v=VIDEO5"
  )
```

**Time estimate:**
- Download videos: 10-20 minutes
- Extract faces: 5-10 minutes
- Training: 5-15 minutes (CPU) or 1-3 minutes (GPU)
- **Total: ~30-60 minutes**

---

## Next Steps

After calibration:
1. Test on new videos to verify improvement
2. Monitor false positive rate
3. If needed, add more videos and recalibrate
4. Consider full training if you have more data

For full training guide, see: `TRAINING_GUIDE.md`


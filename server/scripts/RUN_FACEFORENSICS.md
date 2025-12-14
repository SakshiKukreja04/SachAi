# How to Run FaceForensics Scripts

This guide shows you exactly how to run the FaceForensics-provided scripts step by step.

## Prerequisites

1. **Python 3.7+** installed
2. **FFmpeg** installed (for video processing)
3. **OpenCV** installed: `pip install opencv-python opencv-contrib-python`
4. **Access to FaceForensics dataset** (you need to request access first)

## Step-by-Step Instructions

### Step 1: Get the FaceForensics Download Script

1. Go to: https://github.com/ondyari/FaceForensics
2. Click on the dataset link and fill out the Google Form
3. Once approved, you'll receive `download-FaceForensics.py` via email or download link
4. Save this script to a folder (e.g., `C:\FaceForensics\scripts\`)

### Step 2: Install Dependencies

Open PowerShell or Command Prompt and run:

```powershell
pip install tqdm opencv-python opencv-contrib-python
```

### Step 3: Run the Download Script

**Basic command format:**
```powershell
python download-FaceForensics.py <output_path> -d <dataset> -c <compression> -t <type>
```

**Example - Download original (real) videos:**
```powershell
python download-FaceForensics.py C:\FaceForensics\data -d original -c c23 -t videos --server EU2
```

**Example - Download Deepfakes videos:**
```powershell
python download-FaceForensics.py C:\FaceForensics\data -d Deepfakes -c c23 -t videos --server EU2
```

**Example - Download FaceSwap videos:**
```powershell
python download-FaceForensics.py C:\FaceForensics\data -d FaceSwap -c c23 -t videos --server EU2
```

**Important:** Only EU2 server is currently available. Always include `--server EU2` flag.

**What each parameter means:**
- `C:\FaceForensics\data` - Where to save downloaded videos
- `-d original` - Download original (real) videos
- `-d Deepfakes` - Download Deepfakes manipulation videos
- `-c c23` - High quality compression (recommended)
- `-c c40` - Lower quality (smaller files)
- `-t videos` - Download video files

### Step 4: Extract Faces from Videos

After downloading videos, you need to extract face crops. Use our extraction script:

**For real videos:**
```powershell
cd C:\SachAi\server
python scripts/extract_faceforensics_faces.py `
  --video_dir "C:\FaceForensics\data\downloaded_videos\original_sequences\youtube\c23\videos" `
  --output_dir "C:\FaceForensics\faces\real" `
  --label 0
```

**For fake videos (Deepfakes):**
```powershell
python scripts/extract_faceforensics_faces.py `
  --video_dir "C:\FaceForensics\data\downloaded_videos\manipulated_sequences\Deepfakes\c23\videos" `
  --output_dir "C:\FaceForensics\faces\fake" `
  --label 1
```

**For fake videos (FaceSwap):**
```powershell
python scripts/extract_faceforensics_faces.py `
  --video_dir "C:\FaceForensics\data\downloaded_videos\manipulated_sequences\FaceSwap\c23\videos" `
  --output_dir "C:\FaceForensics\faces\fake" `
  --label 1
```

### Step 5: Organize Data for Training

After extracting faces, organize them:

```powershell
python scripts/prepare_faceforensics.py `
  --mode organize `
  --source_dir "C:\FaceForensics\faces" `
  --target_dir "C:\FaceForensics\training_data" `
  --val_split 0.2
```

This creates:
```
C:\FaceForensics\training_data\
  train\
    real\
    fake\
  val\
    real\
    fake\
```

### Step 6: Train the Model

```powershell
python train/train_faceforensics.py `
  --data_dir "C:\FaceForensics\training_data\train" `
  --checkpoint_out "C:\FaceForensics\checkpoint.pth" `
  --epochs 3 `
  --batch_size 32 `
  --val_split 0.2
```

## Complete Example Workflow

Here's a complete example from start to finish:

```powershell
# 1. Download original videos (real)
python C:\FaceForensics\scripts\download-FaceForensics.py C:\FaceForensics\data -d original -c c23 -t videos --server EU2

# 2. Download Deepfakes videos (fake)
python C:\FaceForensics\scripts\download-FaceForensics.py C:\FaceForensics\data -d Deepfakes -c c23 -t videos --server EU2
```

# 3. Extract faces from real videos
cd C:\SachAi\server
python scripts/extract_faceforensics_faces.py `
  --video_dir "C:\FaceForensics\data\downloaded_videos\original_sequences\youtube\c23\videos" `
  --output_dir "C:\FaceForensics\faces\real" `
  --label 0

# 4. Extract faces from fake videos
python scripts/extract_faceforensics_faces.py `
  --video_dir "C:\FaceForensics\data\downloaded_videos\manipulated_sequences\Deepfakes\c23\videos" `
  --output_dir "C:\FaceForensics\faces\fake" `
  --label 1

# 5. Organize for training
python scripts/prepare_faceforensics.py `
  --mode organize `
  --source_dir "C:\FaceForensics\faces" `
  --target_dir "C:\FaceForensics\training_data" `
  --val_split 0.2

# 6. Train the model
python train/train_faceforensics.py `
  --data_dir "C:\FaceForensics\training_data\train" `
  --checkpoint_out "C:\FaceForensics\checkpoint.pth" `
  --epochs 3 `
  --batch_size 32

# The model checkpoint will be saved automatically during training.
# The best model (lowest validation loss) is saved to the checkpoint file.
# Once training is complete, you can use this checkpoint without retraining.

# To resume training from a checkpoint (if training was interrupted):
python train/train_faceforensics.py `
  --data_dir "C:\FaceForensics\training_data\train" `
  --checkpoint_in "C:\FaceForensics\checkpoint.pth" `
  --checkpoint_out "C:\FaceForensics\checkpoint.pth" `
  --epochs 3 `
  --batch_size 32
```

## Common Issues and Solutions

### Issue: "python: command not found"
**Solution:** Use `py` instead of `python` on Windows:
```powershell
py download-FaceForensics.py ...
```

### Issue: "ModuleNotFoundError: No module named 'tqdm'"
**Solution:** Install missing dependencies:
```powershell
pip install tqdm opencv-python
```

### Issue: "download-FaceForensics.py: No such file or directory"
**Solution:** 
- Make sure you have the script from FaceForensics
- Use full path: `python C:\path\to\download-FaceForensics.py ...`
- Or navigate to the script directory first

### Issue: "Authentication failed" or "Access denied"
**Solution:**
- Make sure you've been approved for dataset access
- Check you're using the correct download script
- Verify your internet connection

### Issue: "FFmpeg not found"
**Solution:** Install FFmpeg:
- Windows: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
- Add FFmpeg to your PATH environment variable

## Tips

1. **Start small**: Download one manipulation type first to test
2. **Use c23 compression**: Good balance of quality and file size
3. **Extract faces incrementally**: Process videos in batches
4. **Monitor disk space**: FaceForensics dataset is large (100GB+)
5. **Use SSD if possible**: Faster for video processing

## Using the Trained Model

Once training is complete, the checkpoint is saved and **does not need to be retrained**. You can use it for inference with the web UI.

### Step 1: Set Up Checkpoint for Flask Server

The Flask server (port 8000) needs to load your trained checkpoint. You have two options:

**Option 1: Set Environment Variable (Recommended)**
```powershell
cd C:\SachAi\server
$env:MODEL_CHECKPOINT = "C:\FaceForensics\checkpoint.pth"
python server.py
```

**Option 2: Copy Checkpoint to Server Directory**
```powershell
Copy-Item "C:\FaceForensics\checkpoint.pth" "C:\SachAi\server\checkpoint.pth"
cd C:\SachAi\server
python server.py
```

The server will automatically detect the checkpoint and use it for inference. You should see:
```
Loading model with checkpoint: C:\FaceForensics\checkpoint.pth
```

### Step 2: Start the Flask Inference Server

The Flask server runs on port 8000 and handles ML model inference:

```powershell
cd C:\SachAi\server
python server.py --port 8000
```

You should see:
```
Loading model with checkpoint: C:\FaceForensics\checkpoint.pth
Starting server on 127.0.0.1:8000 using device=cpu
```

### Step 3: Start the Node.js Backend Server (if using Web UI)

If you have a web UI, start the Node.js server on port 3000:

```powershell
cd C:\SachAi\server
npm start
# or
npm run dev
```

### Step 4: Test with Web UI

1. Open your web UI (usually at `http://localhost:3000`)
2. Upload a video file
3. The system will:
   - Extract frames and faces from the video
   - Send face images to the Flask server (port 8000)
   - Run inference using your trained model (69.58% accuracy)
   - Return deepfake detection results

### Improving Accuracy (Optional)

Your current accuracy is **69.58%**, which is a good start. To improve:

1. **Train for more epochs:**
   ```powershell
   python train/train_faceforensics.py `
     --data_dir "C:\FaceForensics\training_data\train" `
     --checkpoint_in "C:\FaceForensics\checkpoint.pth" `
     --checkpoint_out "C:\FaceForensics\checkpoint.pth" `
     --epochs 5 `
     --batch_size 32
   ```

2. **Train all layers (unfreeze backbone):**
   ```powershell
   python train/train_faceforensics.py `
     --data_dir "C:\FaceForensics\training_data\train" `
     --checkpoint_in "C:\FaceForensics\checkpoint.pth" `
     --checkpoint_out "C:\FaceForensics\checkpoint.pth" `
     --epochs 3 `
     --batch_size 32 `
     --train_all_layers
   ```

3. **Use more training data** - Add more real/fake samples to your dataset

## Next Steps

After completing the workflow:
1. Verify your training data has both real and fake faces
2. Check the data balance (similar number of real/fake samples)
3. Run training and monitor validation accuracy
4. Test the trained model on new videos

For more details, see:
- `FACEFORENSICS_SETUP.md` - Detailed setup guide
- `TRAINING_GUIDE.md` - Training instructions
- `QUICKSTART_TRAINING.md` - Quick start guide


# FaceForensics Dataset Setup Guide

This guide explains how to use the FaceForensics-provided scripts to download and prepare the dataset for training.

## Step 1: Download the Dataset

### 1.1 Get Access to FaceForensics

1. Go to [FaceForensics GitHub](https://github.com/ondyari/FaceForensics)
2. Fill out the Google Form to request access
3. Once approved, you'll receive `download-FaceForensics.py`

### 1.2 Install Required Dependencies

```bash
pip install tqdm opencv-python
```

### 1.3 Download the Dataset

**Basic usage:**
```bash
python download-FaceForensics.py <output_path> -d <dataset_type> -c <compression> -t <file_type>
```

**Parameters:**
- `<output_path>`: Where to save the dataset (e.g., `C:\FaceForensics` or `/home/user/FaceForensics`)
- `-d <dataset_type>`: 
  - `original` - Original real videos
  - `Deepfakes` - Deepfakes manipulation
  - `Face2Face` - Face2Face manipulation
  - `FaceSwap` - FaceSwap manipulation
  - `NeuralTextures` - NeuralTextures manipulation
  - `all` - Download all types
- `-c <compression>`:
  - `c23` - High quality (recommended)
  - `c40` - Low quality (smaller files)
  - `raw` - Uncompressed (very large)
- `-t <file_type>`:
  - `videos` - Download videos
  - `masks` - Download masks (optional)
  - `models` - Download models (optional)

**Example commands:**

```bash
# IMPORTANT: Only EU2 server is available. Always use --server EU2 flag.

# Download original (real) videos with high quality
python download-FaceForensics.py C:\FaceForensics -d original -c c23 -t videos --server EU2

# Download Deepfakes manipulation videos
python download-FaceForensics.py C:\FaceForensics -d Deepfakes -c c23 -t videos --server EU2

# Download FaceSwap manipulation videos
python download-FaceForensics.py C:\FaceForensics -d FaceSwap -c c23 -t videos --server EU2

# Download all manipulation types at once
python download-FaceForensics.py C:\FaceForensics -d all -c c23 -t videos --server EU2

# Lower quality, faster download (recommended for testing)
python download-FaceForensics.py C:\FaceForensics -d original -c c40 -t videos --server EU2
```

## Step 2: Extract Frames from Videos

FaceForensics provides `extracted_compressed_videos.py` to extract frames. However, we need to extract **faces** from frames for training.

### Option A: Use FaceForensics Script (Extract Frames Only)

```bash
python extracted_compressed_videos.py <output_path> -d <dataset_type> -c <compression>
```

This extracts frames, but you'll still need to extract faces from those frames.

### Option B: Use Our Face Extraction Script (Recommended)

We'll create a script that:
1. Extracts frames from videos
2. Detects and crops faces from frames
3. Organizes them into `real/` and `fake/` folders

## Step 3: Extract Faces from Videos

After downloading videos, you need to extract face crops. Use our face extraction script:

```bash
# Navigate to server directory
cd server

# Extract faces from FaceForensics videos
python scripts/extract_faceforensics_faces.py \
  --video_dir "C:\FaceForensics\downloaded_videos\original_sequences\youtube\c23\videos" \
  --output_dir "C:\FaceForensics\faces\real" \
  --label 0

python scripts/extract_faceforensics_faces.py \
  --video_dir "C:\FaceForensics\downloaded_videos\manipulated_sequences\Deepfakes\c23\videos" \
  --output_dir "C:\FaceForensics\faces\fake" \
  --label 1
```

## Step 4: Organize Data for Training

After extracting faces, organize them into the training format:

```
training_data/
  real/
    face_00001.jpg
    face_00002.jpg
    ...
  fake/
    face_00001.jpg
    face_00002.jpg
    ...
```

Or use our preparation script:

```bash
python scripts/prepare_faceforensics.py \
  --mode organize \
  --source_dir "C:\FaceForensics\faces" \
  --target_dir "C:\FaceForensics\training_data" \
  --val_split 0.2
```

## Step 5: Train the Model

Once data is organized:

```bash
python train/train_faceforensics.py \
  --data_dir "C:\FaceForensics\training_data\train" \
  --checkpoint_out checkpoint.pth \
  --epochs 10 \
  --batch_size 32 \
  --val_split 0.2
```

## Complete Workflow Example

Here's a complete example workflow:

```bash
# 1. Download original videos (use --server EU2, only available server)
python download-FaceForensics.py C:\FaceForensics -d original -c c23 -t videos --server EU2

# 2. Download Deepfakes videos
python download-FaceForensics.py C:\FaceForensics -d Deepfakes -c c23 -t videos --server EU2

# 3. Extract faces from original videos (real)
python scripts/extract_faceforensics_faces.py \
  --video_dir "C:\FaceForensics\downloaded_videos\original_sequences\youtube\c23\videos" \
  --output_dir "C:\FaceForensics\faces\real"

# 4. Extract faces from Deepfakes videos (fake)
python scripts/extract_faceforensics_faces.py \
  --video_dir "C:\FaceForensics\downloaded_videos\manipulated_sequences\Deepfakes\c23\videos" \
  --output_dir "C:\FaceForensics\faces\fake"

# 5. Organize for training
python scripts/prepare_faceforensics.py \
  --mode organize \
  --source_dir "C:\FaceForensics\faces" \
  --target_dir "C:\FaceForensics\training_data" \
  --val_split 0.2

# 6. Train the model
python train/train_faceforensics.py \
  --data_dir "C:\FaceForensics\training_data\train" \
  --checkpoint_out checkpoint.pth \
  --epochs 10 \
  --batch_size 32
```

## Troubleshooting

### Issue: "download-FaceForensics.py not found"
- Make sure you've received the script from FaceForensics after approval
- Check the script is in your current directory or provide full path

### Issue: "Permission denied" or authentication errors
- Make sure you've been approved for dataset access
- Check your internet connection
- The script may require authentication tokens

### Issue: "FFmpeg not found"
- Install FFmpeg: https://ffmpeg.org/download.html
- On Windows: `choco install ffmpeg` or download from website
- On Linux: `sudo apt-get install ffmpeg`
- On Mac: `brew install ffmpeg`

### Issue: Videos downloaded but can't extract faces
- Make sure videos are in supported format (mp4, avi, etc.)
- Check that OpenCV can read the videos
- Verify face detection model is available

## Next Steps

After completing the setup:
1. Verify your training data structure
2. Run training (see QUICKSTART_TRAINING.md)
3. Test the trained model
4. Deploy to your server


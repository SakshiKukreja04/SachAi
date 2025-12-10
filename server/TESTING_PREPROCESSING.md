# Preprocessing Pipeline Testing Guide

## Quick Start

Run the automated test script:

```bash
chmod +x scripts/test_preprocessing.sh
./scripts/test_preprocessing.sh ./path/to/video.mp4
```

This will:
1. ✓ Verify dependencies (ffmpeg, Python packages)
2. ✓ Extract frames and detect faces
3. ✓ Validate output structure
4. ✓ Count frames and faces
5. ✓ Validate JSON integrity

---

## Manual Testing & Verification

### Step 1: Extract Frames Only

Test if FFmpeg integration works:

```bash
python3 ml/face_preprocess.py --video ./test.mp4 --out ./tmp --jobId frame-test --fps 1
ls ./tmp/frame-test/frames/ | head
```

**Expected output:**
- Frame files: `frame_00001.jpg`, `frame_00002.jpg`, ...
- Number of frames ≈ video duration in seconds (at fps=1)

**Common issues:**
- ❌ No files created → FFmpeg not installed or video path invalid
- ❌ Files but no content → Video codec not supported, try converting with: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

---

### Step 2: Run Full Pipeline

Extract frames AND detect faces:

```bash
python3 ml/face_preprocess.py \
  --video ./test.mp4 \
  --out ./tmp \
  --jobId full-test \
  --fps 1 \
  --workers 4
```

**Check outputs exist:**

```bash
# Frames
ls -la ./tmp/full-test/frames/ | wc -l

# Faces (cropped 299x299)
ls -la ./tmp/full-test/faces/ | wc -l

# Landmarks JSON
cat ./tmp/full-test/landmarks.json | python3 -m json.tool | head -100
```

---

### Step 3: Validate Frames

**Check a single frame:**

```bash
# Linux/macOS
file ./tmp/full-test/frames/frame_00001.jpg
identify ./tmp/full-test/frames/frame_00001.jpg  # if ImageMagick installed

# Check file size (should be > 0 KB)
ls -lh ./tmp/full-test/frames/frame_00001.jpg
```

**Quick visual check:**

```bash
# Display first frame (Linux/macOS with display/feh installed)
display ./tmp/full-test/frames/frame_00001.jpg

# Or use PIL in Python
python3 << 'EOF'
from PIL import Image
img = Image.open('./tmp/full-test/frames/frame_00001.jpg')
print(f"Frame size: {img.size}")
print(f"Mode: {img.mode}")
img.show()
EOF
```

---

### Step 4: Validate Face Crops

**Check face crop dimensions:**

```bash
python3 << 'EOF'
from PIL import Image
import os

faces_dir = './tmp/full-test/faces'
for f in sorted(os.listdir(faces_dir))[:3]:  # first 3 faces
    path = os.path.join(faces_dir, f)
    img = Image.open(path)
    print(f"{f}: {img.size} pixels, {os.path.getsize(path)} bytes")
EOF
```

**Expected output:**
- All faces should be **299x299 pixels** (hardcoded in script)
- File size typically 5-50 KB

---

### Step 5: Validate Landmarks JSON

**Structure check:**

```bash
python3 << 'EOF'
import json

with open('./tmp/full-test/landmarks.json', 'r') as f:
    data = json.load(f)

print(f"jobId: {data['jobId']}")
print(f"Total frames: {len(data['frames'])}")

# Count total faces
total_faces = sum(len(f['faces']) for f in data['frames'])
print(f"Total faces: {total_faces}")

# Show first frame with faces
if data['frames']:
    frame = data['frames'][0]
    print(f"\nFirst frame: {frame['frame']}")
    print(f"  Faces detected: {len(frame['faces'])}")
    for i, face in enumerate(frame['faces'][:2]):  # show first 2
        print(f"    Face {i+1}:")
        print(f"      bbox: {face['bbox']}")
        print(f"      center: {face['center']}")
        print(f"      path: {face['face_path']}")
EOF
```

**Expected JSON structure:**

```json
{
  "jobId": "test-job",
  "frames": [
    {
      "frame": "frame_00001.jpg",
      "faces": [
        {
          "bbox": [x, y, width, height],
          "center": [cx, cy],
          "face_path": "faces/face_frame_00001_1.jpg"
        }
      ]
    }
  ]
}
```

---

## Success Criteria

✅ **Pipeline Working Correctly** if:

1. **Frames extracted**
   - ✓ Frame count > 0
   - ✓ All files are valid JPEG
   - ✓ Each frame is 5-100 KB

2. **Faces detected** (depends on video content)
   - ✓ Face count ≥ 0 (okay if no faces in test video)
   - ✓ All face crops are exactly 299x299
   - ✓ Face files are valid JPEG

3. **Landmarks JSON valid**
   - ✓ Valid JSON syntax
   - ✓ Has `jobId` and `frames` fields
   - ✓ Each frame has `frame` and `faces` fields
   - ✓ Each face has `bbox`, `center`, `face_path`

4. **No errors during processing**
   - ✓ No OpenCV errors
   - ✓ No file I/O errors
   - ✓ Processing completes in reasonable time

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cv2'`

**Solution:**
```bash
pip install opencv-python numpy pillow tqdm
```

### Issue: `ffmpeg: command not found`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (using scoop)
scoop install ffmpeg
```

### Issue: `No frames extracted`

**Causes:**
1. Video file corrupted or invalid codec
2. ffmpeg not installed
3. Output directory permission denied

**Fix:**
```bash
# Test video with ffmpeg directly
ffmpeg -i ./test.mp4 -t 5 -f null -  # try first 5 seconds

# Convert to standard H.264 if needed
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

### Issue: `No faces detected in video`

**This is normal!** Haar cascade requires visible frontal faces. If a test video has no visible faces:
- Use a video with at least one visible frontal face
- Try with `--sample 100` to process more frames
- Note: Haar cascade may miss faces at angles > 30°

### Issue: `JSON validation failed`

**Check JSON manually:**
```bash
# View entire file
cat ./tmp/full-test/landmarks.json | python3 -m json.tool

# Validate syntax
python3 -m json.tool < ./tmp/full-test/landmarks.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

---

## Performance Notes

- Frame extraction: ~1-5 seconds per minute of video (depends on resolution)
- Face detection: ~0.5-2 seconds per frame (with 4 workers)
- Typical test: 10-second video → 10 frames → processing in ~10-20 seconds

---

## Next Steps

Once verified:

1. **Integrate with backend**: Update `src/services/processor.ts` to call Python script
2. **Test with real model server**: Send face crops + landmarks to ML inference
3. **Monitor latency**: Track processing time per video in production


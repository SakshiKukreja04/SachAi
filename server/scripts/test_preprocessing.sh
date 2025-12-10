#!/usr/bin/env bash

# Test preprocessing pipeline end-to-end
# Usage: ./test_preprocessing.sh [video_path] [jobId]

set -e

VIDEO=${1:-./test-video.mp4}
JOB_ID=${2:-test-job-$(date +%s)}
OUT_DIR="./tmp"
JOB_DIR="$OUT_DIR/$JOB_ID"

echo "=========================================="
echo "SachAI Preprocessing Pipeline Test"
echo "=========================================="
echo "Video: $VIDEO"
echo "JobID: $JOB_ID"
echo "Output: $JOB_DIR"
echo ""

# Check if video exists
if [ ! -f "$VIDEO" ]; then
  echo "❌ ERROR: Video file not found: $VIDEO"
  echo "   Please provide a valid video path"
  echo "   Usage: $0 <video_path> [jobId]"
  exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
  echo "❌ ERROR: ffmpeg not installed"
  echo "   Install via: apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)"
  exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
  echo "❌ ERROR: python3 not installed"
  exit 1
fi

# Check Python dependencies
echo "[1/5] Checking Python dependencies..."
python3 -c "import cv2; import numpy; import PIL" 2>/dev/null || {
  echo "❌ Missing Python packages. Install with:"
  echo "   pip install opencv-python numpy pillow tqdm"
  exit 1
}
echo "✓ Python dependencies OK"

# Run preprocessing
echo ""
echo "[2/5] Running face_preprocess.py..."
python3 ml/face_preprocess.py \
  --video "$VIDEO" \
  --out "$OUT_DIR" \
  --jobId "$JOB_ID" \
  --fps 1 \
  --workers 4 2>&1 || {
  echo "❌ Preprocessing failed"
  exit 1
}
echo "✓ Preprocessing completed"

# Check output directories
echo ""
echo "[3/5] Validating output structure..."

FRAMES_DIR="$JOB_DIR/frames"
FACES_DIR="$JOB_DIR/faces"
LANDMARKS_JSON="$JOB_DIR/landmarks.json"

if [ ! -d "$FRAMES_DIR" ]; then
  echo "❌ ERROR: frames directory not created at $FRAMES_DIR"
  exit 1
fi
echo "✓ Frames directory: $FRAMES_DIR"

if [ ! -d "$FACES_DIR" ]; then
  echo "❌ ERROR: faces directory not created at $FACES_DIR"
  exit 1
fi
echo "✓ Faces directory: $FACES_DIR"

if [ ! -f "$LANDMARKS_JSON" ]; then
  echo "❌ ERROR: landmarks.json not created at $LANDMARKS_JSON"
  exit 1
fi
echo "✓ Landmarks JSON: $LANDMARKS_JSON"

# Count outputs
echo ""
echo "[4/5] Checking output counts..."

FRAME_COUNT=$(find "$FRAMES_DIR" -name "frame_*.jpg" 2>/dev/null | wc -l)
FACE_COUNT=$(find "$FACES_DIR" -name "face_*.jpg" 2>/dev/null | wc -l)

echo "  • Frames extracted: $FRAME_COUNT"
if [ "$FRAME_COUNT" -eq 0 ]; then
  echo "  ⚠ WARNING: No frames extracted (video may be corrupted or too short)"
else
  echo "  ✓ Frames extracted successfully"
fi

echo "  • Faces detected: $FACE_COUNT"
if [ "$FACE_COUNT" -eq 0 ]; then
  echo "  ⚠ WARNING: No faces detected (video may not contain visible faces)"
else
  echo "  ✓ Faces detected and cropped"
fi

# Validate JSON structure
echo ""
echo "[5/5] Validating JSON structure..."

python3 << 'EOF'
import json
import sys

landmarks_path = sys.argv[1]

try:
  with open(landmarks_path, 'r') as f:
    data = json.load(f)
  
  # Check required fields
  assert 'jobId' in data, "Missing 'jobId' field"
  assert 'frames' in data, "Missing 'frames' field"
  assert isinstance(data['frames'], list), "'frames' should be a list"
  
  # Check frame structure
  total_faces = 0
  for i, frame in enumerate(data['frames']):
    assert 'frame' in frame, f"Frame {i}: missing 'frame' field"
    assert 'faces' in frame, f"Frame {i}: missing 'faces' field"
    assert isinstance(frame['faces'], list), f"Frame {i}: 'faces' should be a list"
    
    for j, face in enumerate(frame['faces']):
      assert 'bbox' in face, f"Frame {i}, Face {j}: missing 'bbox'"
      assert 'center' in face, f"Frame {i}, Face {j}: missing 'center'"
      assert 'face_path' in face, f"Frame {i}, Face {j}: missing 'face_path'"
      total_faces += 1
  
  print(f"✓ JSON structure valid (jobId: {data['jobId']}, total faces: {total_faces})")
  
except Exception as e:
  print(f"❌ JSON validation failed: {e}")
  sys.exit(1)
EOF
python3 -c "
import json
import sys

landmarks_path = '$LANDMARKS_JSON'

try:
  with open(landmarks_path, 'r') as f:
    data = json.load(f)
  
  # Check required fields
  assert 'jobId' in data, 'Missing jobId field'
  assert 'frames' in data, 'Missing frames field'
  assert isinstance(data['frames'], list), 'frames should be a list'
  
  # Check frame structure
  total_faces = 0
  for i, frame in enumerate(data['frames']):
    assert 'frame' in frame, f'Frame {i}: missing frame field'
    assert 'faces' in frame, f'Frame {i}: missing faces field'
    assert isinstance(frame['faces'], list), f'Frame {i}: faces should be a list'
    
    for j, face in enumerate(frame['faces']):
      assert 'bbox' in face, f'Frame {i}, Face {j}: missing bbox'
      assert 'center' in face, f'Frame {i}, Face {j}: missing center'
      assert 'face_path' in face, f'Frame {i}, Face {j}: missing face_path'
      total_faces += 1
  
  print(f'✓ JSON structure valid (jobId: {data[\"jobId\"]}, total faces: {total_faces})')
  
except Exception as e:
  print(f'❌ JSON validation failed: {e}')
  sys.exit(1)
" || exit 1

echo ""
echo "=========================================="
echo "✅ PREPROCESSING PIPELINE TEST PASSED"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  Frames:      $FRAMES_DIR"
echo "  Faces:       $FACES_DIR"
echo "  Landmarks:   $LANDMARKS_JSON"
echo ""
echo "Next steps:"
echo "  1. Inspect frames: ls $FRAMES_DIR | head"
echo "  2. Inspect faces: ls $FACES_DIR | head"
echo "  3. View landmarks: cat $LANDMARKS_JSON | head -50"
echo ""

#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-video> [jobId]"
  exit 1
fi

VIDEO=$1
JOBID=${2:-test}
OUTDIR=./tmp

echo "Running sample validation for video: $VIDEO, jobId: $JOBID"
python ml/face_preprocess.py --video "$VIDEO" --out "$OUTDIR" --jobId "$JOBID" --fps 1 --workers 4 --sample 10

JOB_DIR="$OUTDIR/$JOBID"
FRAMES_DIR="$JOB_DIR/frames"
FACES_DIR="$JOB_DIR/faces"
LANDMARKS="$JOB_DIR/landmarks.json"

echo "Frames: $FRAMES_DIR"
echo "Faces: $FACES_DIR"
echo "Landmarks: $LANDMARKS"

"""Extract face crops from FaceForensics videos.

This script processes FaceForensics videos and extracts face crops,
organizing them for training.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


def detect_face(image: np.ndarray) -> Optional[tuple]:
    """Detect face in image using OpenCV's DNN face detector.
    
    Returns (x, y, w, h) bounding box or None if no face found.
    """
    # Load face detection model (OpenCV DNN)
    # You can download the model files from:
    # https://github.com/opencv/opencv/tree/master/samples/dnn
    
    # Try to find model files in common locations
    model_dir = Path(__file__).parent.parent / "models" / "opencv_dnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    prototxt = model_dir / "deploy.prototxt"
    model_file = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    # If models don't exist, use Haar Cascade as fallback
    if not model_file.exists():
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # Return largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                return tuple(largest)
        return None
    
    # Use DNN face detector
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model_file))
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x2, y2 = box.astype("int")
            return (x, y, x2 - x, y2 - y)
    
    return None


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    label: int,
    frame_interval: int = 30,
    min_face_size: int = 64,
):
    """Extract face crops from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save face crops
        label: Label for this video (0=real, 1=fake)
        frame_interval: Extract every Nth frame (default: 30 = ~1 per second at 30fps)
        min_face_size: Minimum face size in pixels
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    video_name = video_path.stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            face_box = detect_face(frame)
            
            if face_box:
                x, y, w, h = face_box
                
                # Check minimum size
                if w >= min_face_size and h >= min_face_size:
                    # Expand crop slightly for context
                    margin = 0.2
                    x = max(0, int(x - w * margin))
                    y = max(0, int(y - h * margin))
                    w = min(frame.shape[1] - x, int(w * (1 + 2 * margin)))
                    h = min(frame.shape[0] - y, int(h * (1 + 2 * margin)))
                    
                    # Crop and resize to 299x299 (Xception input size)
                    face_crop = frame[y:y+h, x:x+w]
                    face_crop = cv2.resize(face_crop, (299, 299))
                    
                    # Save face crop
                    output_filename = f"{video_name}_frame_{frame_count:06d}_face.jpg"
                    output_path = output_dir / output_filename
                    cv2.imwrite(str(output_path), face_crop)
                    saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract faces from FaceForensics videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for face crops")
    parser.add_argument("--label", type=int, default=0, help="Label: 0=real, 1=fake")
    parser.add_argument("--frame_interval", type=int, default=30, 
                       help="Extract every Nth frame (default: 30 = ~1 per second)")
    parser.add_argument("--min_face_size", type=int, default=64,
                       help="Minimum face size in pixels (default: 64)")
    parser.add_argument("--video_extensions", type=str, nargs="+", 
                       default=[".mp4", ".avi", ".mov", ".mkv"],
                       help="Video file extensions to process")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")
    
    # Find all video files
    video_files = []
    for ext in args.video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"**/*{ext}"))
    
    if len(video_files) == 0:
        raise SystemExit(f"No video files found in {video_dir}")
    
    print(f"Found {len(video_files)} video files")
    print(f"Extracting faces to: {output_dir}")
    print(f"Label: {'real' if args.label == 0 else 'fake'}")
    print(f"Frame interval: {args.frame_interval} (every {args.frame_interval} frames)")
    print()
    
    total_faces = 0
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            count = extract_faces_from_video(
                video_path,
                output_dir,
                args.label,
                args.frame_interval,
                args.min_face_size,
            )
            total_faces += count
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    print(f"\nExtraction complete!")
    print(f"Total face crops saved: {total_faces}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()


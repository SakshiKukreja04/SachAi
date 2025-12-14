"""
Prepare YouTube videos for calibration training.

Downloads YouTube videos, extracts frames, detects faces, and organizes them
for calibration training with real videos only.
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

# Add parent directory to path
script_dir = Path(__file__).parent
server_dir = script_dir.parent.parent
sys.path.insert(0, str(server_dir))

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("WARNING: yt-dlp not available. Install with: pip install yt-dlp")


def download_youtube_video(url: str, output_dir: Path) -> Optional[Path]:
    """Download a YouTube video using yt-dlp."""
    if not YT_DLP_AVAILABLE:
        print(f"ERROR: yt-dlp not available. Cannot download {url}")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video.mp4"
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if output_path.exists():
            print(f"[OK] Downloaded: {url} -> {output_path}")
            return output_path
        else:
            print(f"[ERROR] Download failed: {url}")
            return None
    except Exception as e:
        print(f"[ERROR] Error downloading {url}: {e}")
        return None


def detect_face(image: np.ndarray) -> Optional[tuple]:
    """Detect face in image using OpenCV Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try to find cascade file
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_default.xml',
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if os.path.exists(path):
            face_cascade = cv2.CascadeClassifier(path)
            break
    
    if face_cascade is None:
        print("[WARNING] Haar cascade not found, using simple detection")
        # Fallback: use center region
        h, w = gray.shape
        return (w//4, h//4, w//2, h//2)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64)
    )
    
    if len(faces) > 0:
        # Return largest face
        largest = max(faces, key=lambda x: x[2] * x[3])
        return tuple(largest)
    
    return None


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 30,
    min_face_size: int = 64,
) -> int:
    """Extract faces from video frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {fps:.2f} fps, {total_frames} frames")
    
    saved_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_idx % frame_interval == 0:
            face_bbox = detect_face(frame)
            
            if face_bbox:
                x, y, w, h = face_bbox
                
                # Add margin
                margin = 0.2
                x = max(0, int(x - w * margin))
                y = max(0, int(y - h * margin))
                w = int(w * (1 + 2 * margin))
                h = int(h * (1 + 2 * margin))
                
                # Crop and resize to 299x299 (Xception input size)
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (299, 299))
                    
                    # Save face
                    face_filename = f"face_{saved_count:05d}.jpg"
                    face_path = output_dir / face_filename
                    cv2.imwrite(str(face_path), face_resized)
                    saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"  Extracted {saved_count} faces")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YouTube videos for calibration training"
    )
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        required=True,
        help="YouTube video URLs (5-10 recommended)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./calibration_data",
        help="Output directory for extracted faces"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=30,
        help="Extract every Nth frame (default: 30)"
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download videos, don't extract faces"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    faces_dir = output_dir / "real"  # All YouTube videos are real
    
    print(f"\n{'='*60}")
    print("YouTube Calibration Data Preparation")
    print(f"{'='*60}")
    print(f"URLs: {len(args.urls)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    if not YT_DLP_AVAILABLE:
        print("ERROR: yt-dlp is required. Install with:")
        print("  pip install yt-dlp")
        sys.exit(1)
    
    # Download videos
    print("Step 1: Downloading videos...")
    downloaded_videos = []
    for i, url in enumerate(args.urls, 1):
        print(f"\n[{i}/{len(args.urls)}] Downloading: {url}")
        video_dir = videos_dir / f"video_{i:02d}"
        video_path = download_youtube_video(url, video_dir)
        if video_path:
            downloaded_videos.append(video_path)
    
    print(f"\n[OK] Downloaded {len(downloaded_videos)}/{len(args.urls)} videos")
    
    if args.download_only:
        print("\n[OK] Download only mode - skipping face extraction")
        return
    
    # Extract faces
    print(f"\nStep 2: Extracting faces...")
    faces_dir.mkdir(parents=True, exist_ok=True)
    
    total_faces = 0
    for i, video_path in enumerate(downloaded_videos, 1):
        print(f"\n[{i}/{len(downloaded_videos)}] Processing: {video_path.name}")
        faces_count = extract_faces_from_video(
            video_path,
            faces_dir,
            frame_interval=args.frame_interval
        )
        total_faces += faces_count
    
    print(f"\n{'='*60}")
    print(f"[OK] Calibration data prepared!")
    print(f"  Videos downloaded: {len(downloaded_videos)}")
    print(f"  Faces extracted: {total_faces}")
    print(f"  Output directory: {faces_dir}")
    print(f"{'='*60}\n")
    
    print("Next step: Run calibration training:")
    print(f"  python train/train_calibration.py --data_dir {faces_dir.parent}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Face preprocessing utilities for SachAI.
Produces frames, detects faces using OpenCV Haar cascades,
and writes cropped face images and landmarks JSON.
"""

import cv2
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import io
from PIL import Image
import numpy as np


def extract_frames_in_memory(video_path: str, fps: int = 1):
    """Extract frames from a video at given fps and yield numpy arrays (no disk writes)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(video_fps / float(fps))))

    frames = []
    idx = 0
    out_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append((out_idx, frame))
            out_idx += 1
        idx += 1

    cap.release()
    print(f"[extract_frames_in_memory] Extracted {len(frames)} frames from {video_path}")
    return frames


# Top-level function used by ProcessPoolExecutor
def _detect_and_upload_from_frame(frame_idx: int, img, jobId: str, margin_pct: float = 0.1, backend_url: str = 'http://localhost:3000'):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        frame_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            mx = int(w * margin_pct)
            my = int(h * margin_pct)
            x0 = max(0, x - mx)
            y0 = max(0, y - my)
            x1 = min(img.shape[1], x + w + mx)
            y1 = min(img.shape[0], y + h + my)

            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pil = pil.resize((299, 299))

            bio = io.BytesIO()
            pil.save(bio, format='JPEG')
            bio.seek(0)
            face_fname = f"face_frame_{frame_idx:05d}_{i+1}.jpg"

            # POST to backend internal upload
            try:
                files = {'file': (face_fname, bio, 'image/jpeg')}
                data = {'jobId': jobId, 'filename': face_fname}
                resp = requests.post(f"{backend_url}/internal/upload_face", files=files, data=data, timeout=30)
                if resp.ok:
                    resj = resp.json()
                    uploaded_id = resj.get('id')
                else:
                    uploaded_id = None
            except Exception as e:
                print(f"[upload] failed: {e}")
                uploaded_id = None

            cx = int(x + w / 2)
            cy = int(y + h / 2)
            frame_faces.append({
                "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "center": [cx, cy],
                "uploaded_id": uploaded_id,
                "filename": face_fname
            })

        return {"frame": f"frame_{frame_idx:05d}.jpg", "faces": frame_faces}
    except Exception as e:
        print(f"[process_frame_for_faces] Error processing frame {frame_idx}: {e}")
        return {"frame": f"frame_{frame_idx:05d}.jpg", "faces": []}


def detect_and_upload_faces(video_path: str, jobId: str = "", fps: int = 1, margin_pct: float = 0.1, backend_url: str = 'http://localhost:3000'):
    """Extract frames in-memory, detect faces and upload crops to backend; returns landmarks structure."""
    frames = extract_frames_in_memory(video_path, fps=fps)
    results = {"jobId": jobId, "frames": []}
    if not frames:
        print(f"[detect_and_upload_faces] No frames extracted from {video_path}")
        return results

    print(f"[detect_and_upload_faces] Processing {len(frames)} frames and uploading faces to {backend_url}")
    for (idx, frame) in tqdm(frames):
        res = _detect_and_upload_from_frame(idx, frame, jobId, margin_pct=margin_pct, backend_url=backend_url)
        results['frames'].append(res)

    return results


def validate_sample(video_path: str, tmp_dir: str, jobId: str = 'test', fps: int = 1, sample_n: int = 10, workers: int = 4):
    """Create frames and run face detection on a sample of frames, then show crops for quick verification."""
    job_dir = os.path.join(tmp_dir, jobId)
    frames_dir = os.path.join(job_dir, 'frames')
    faces_dir = os.path.join(job_dir, 'faces')
    landmarks_path = os.path.join(job_dir, 'landmarks.json')

    os.makedirs(job_dir, exist_ok=True)

    frames = extract_frames_in_memory(video_path, fps=fps)
    if not frames:
        print("[validate_sample] No frames extracted")
        return

    sample_frames = frames[:sample_n]
    # Create a temporary frames dir for sampled frames
    sample_frames_dir = os.path.join(job_dir, 'sample_frames')
    os.makedirs(sample_frames_dir, exist_ok=True)
    for f in sample_frames:
        dst = os.path.join(sample_frames_dir, os.path.basename(f))
        if not os.path.exists(dst):
            os.replace(f, dst) if False else __import__('shutil').copyfile(f, dst)

    # Run detection on sampled frames and upload
    detect_and_upload_faces(video_path, jobId=jobId, fps=fps)

    # Print summary and attempt to show first N crops
    # Attempt to read landmarks from uploads is not available; just notify
    data = {'frames': []}

    total_faces = sum(len(f['faces']) for f in data.get('frames', []))
    print(f"[validate_sample] Detected {total_faces} faces in sample {len(data.get('frames', []))} frames")

    # Show first few crops using PIL.Image.show (may open external viewer)
    shown = 0
    for frame_entry in data.get('frames', []):
        for face in frame_entry.get('faces', []):
            face_path = face['face_path']
            try:
                img = Image.open(face_path)
                img.show()
                shown += 1
                if shown >= sample_n:
                    break
            except Exception as e:
                print(f"[validate_sample] Unable to show {face_path}: {e}")
        if shown >= sample_n:
            break

    print(f"[validate_sample] Uploaded face crops for job: {jobId}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face preprocessing utility')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--out', default='./tmp', help='Base output directory (default ./tmp)')
    parser.add_argument('--jobId', default='test', help='Job identifier')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--sample', type=int, default=10, help='Number of sample frames to validate/show')

    args = parser.parse_args()

    job_dir = os.path.join(args.out, args.jobId)
    frames_dir = os.path.join(job_dir, 'frames')
    faces_dir = os.path.join(job_dir, 'faces')
    landmarks_path = os.path.join(job_dir, 'landmarks.json')

    try:
        print(f"[main] video={args.video} out={args.out} jobId={args.jobId} fps={args.fps} workers={args.workers}")
        # New in-memory processing: detect faces and upload crops to backend without writing face files to disk
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:3000')
        res = detect_and_upload_faces(args.video, jobId=args.jobId, fps=args.fps, margin_pct=0.1, backend_url=backend_url)
        print(f"[main] Completed. Uploaded faces for job: {args.jobId}")
    except Exception as e:
        print(f"[main] Error: {e}")

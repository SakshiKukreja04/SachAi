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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import numpy as np


def extract_frames(video_path: str, out_dir: str, fps: int = 1) -> list:
    """Extract frames from a video at given fps into out_dir.
    Returns list of frame file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
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
            fname = f"frame_{out_idx:05d}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, frame)
            frames.append(fpath)
            out_idx += 1
        idx += 1

    cap.release()
    print(f"[extract_frames] Wrote {len(frames)} frames to {out_dir}")
    return frames


# Top-level function used by ProcessPoolExecutor
def _process_frame_for_faces(args):
    frame_path, out_faces_dir, margin_pct = args
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return {"frame": os.path.basename(frame_path), "faces": []}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        frame_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            # expand bbox by margin_pct
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

            face_fname = f"face_{Path(frame_path).stem}_{i+1}.jpg"
            face_path = os.path.join(out_faces_dir, face_fname)
            pil.save(face_path)

            cx = int(x + w / 2)
            cy = int(y + h / 2)
            frame_faces.append({
                "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "center": [cx, cy],
                "face_path": os.path.relpath(face_path)
            })

        return {"frame": os.path.basename(frame_path), "faces": frame_faces}
    except Exception as e:
        print(f"[process_frame_for_faces] Error processing {frame_path}: {e}")
        return {"frame": os.path.basename(frame_path), "faces": []}


def detect_and_crop_faces(frames_dir: str, out_faces_dir: str, landmarks_json_path: str, jobId: str = "", workers: int = 4, margin_pct: float = 0.1):
    """Detect faces in frames and crop them into out_faces_dir.
    Writes a JSON landmarks file at landmarks_json_path.
    """
    os.makedirs(out_faces_dir, exist_ok=True)
    frames = sorted([str(p) for p in Path(frames_dir).glob('frame_*.jpg')])

    results = {"jobId": jobId, "frames": []}
    if not frames:
        print(f"[detect_and_crop_faces] No frames found in {frames_dir}")
        with open(landmarks_json_path, 'w') as fh:
            json.dump(results, fh, indent=2)
        return results

    print(f"[detect_and_crop_faces] Processing {len(frames)} frames with {workers} workers")

    tasks = [(f, out_faces_dir, margin_pct) for f in frames]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_frame_for_faces, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            results['frames'].append(res)

    # Save JSON
    os.makedirs(os.path.dirname(landmarks_json_path), exist_ok=True)
    with open(landmarks_json_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"[detect_and_crop_faces] Wrote landmarks to {landmarks_json_path}")
    return results


def validate_sample(video_path: str, tmp_dir: str, jobId: str = 'test', fps: int = 1, sample_n: int = 10, workers: int = 4):
    """Create frames and run face detection on a sample of frames, then show crops for quick verification."""
    job_dir = os.path.join(tmp_dir, jobId)
    frames_dir = os.path.join(job_dir, 'frames')
    faces_dir = os.path.join(job_dir, 'faces')
    landmarks_path = os.path.join(job_dir, 'landmarks.json')

    os.makedirs(job_dir, exist_ok=True)

    frames = extract_frames(video_path, frames_dir, fps=fps)
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

    # Run detection on sampled frames
    detect_and_crop_faces(sample_frames_dir, faces_dir, landmarks_path, jobId=jobId, workers=workers)

    # Print summary and attempt to show first N crops
    with open(landmarks_path, 'r') as fh:
        data = json.load(fh)

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

    print(f"[validate_sample] Saved faces in: {faces_dir}")
    print(f"[validate_sample] Landmarks JSON: {landmarks_path}")


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
        extract_frames(args.video, frames_dir, fps=args.fps)
        detect_and_crop_faces(frames_dir, faces_dir, landmarks_path, jobId=args.jobId, workers=args.workers)
        print(f"[main] Completed. Faces at: {faces_dir}. Landmarks JSON: {landmarks_path}")
    except Exception as e:
        print(f"[main] Error: {e}")

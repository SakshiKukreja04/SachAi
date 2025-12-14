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

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: mediapipe not available. Install with: pip install mediapipe")


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


def _extract_mouth_landmarks_mediapipe(img, face_bbox):
    """Extract mouth landmarks using MediaPipe."""
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # MediaPipe mouth landmarks indices (inner mouth: 12 points, outer mouth: 20 points)
            # Using outer mouth contour: 61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 13, 82, 81, 80, 78
            # Simplified: use points around mouth (indices 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
            # Actually, MediaPipe provides 468 landmarks. Mouth outer: 61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321
            # Mouth inner: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
            
            # Get mouth region landmarks (indices for outer mouth)
            mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 13, 82, 81, 80, 78]
            
            h, w = img.shape[:2]
            mouth_landmarks = []
            
            for idx in mouth_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    mouth_landmarks.append([x, y])
            
            if len(mouth_landmarks) >= 4:
                return mouth_landmarks
        
        return None
    except Exception as e:
        print(f"[extract_mouth_landmarks] Error: {e}")
        return None


def _extract_mouth_landmarks_heuristic(img, face_bbox):
    """
    Extract mouth landmarks using heuristic method based on face bounding box.
    This is a fallback when MediaPipe is not available.
    
    Estimates mouth position based on typical face proportions:
    - Mouth is typically at 60-70% down the face
    - Mouth width is typically 40-50% of face width
    """
    try:
        x, y, w, h = face_bbox
        
        # Typical face proportions for mouth location
        # Mouth center is approximately 65% down from top of face
        mouth_center_y = int(y + h * 0.65)
        mouth_center_x = int(x + w * 0.5)
        
        # Mouth width is approximately 45% of face width
        mouth_width = int(w * 0.45)
        mouth_height = int(h * 0.12)  # Mouth height is about 12% of face height
        
        # Generate 20 points around mouth (simulating MediaPipe format)
        mouth_landmarks = []
        num_points = 20
        
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            # Create an ellipse shape for mouth
            radius_x = mouth_width / 2
            radius_y = mouth_height / 2
            
            # Add some variation to make it more realistic
            if i < num_points // 2:
                # Upper lip points
                point_x = int(mouth_center_x + radius_x * np.cos(angle) * 0.8)
                point_y = int(mouth_center_y - radius_y * np.sin(angle) * 0.6)
            else:
                # Lower lip points
                point_x = int(mouth_center_x + radius_x * np.cos(angle) * 0.8)
                point_y = int(mouth_center_y + radius_y * np.sin(angle) * 0.6)
            
            mouth_landmarks.append([point_x, point_y])
        
        return mouth_landmarks
    except Exception as e:
        print(f"[extract_mouth_landmarks_heuristic] Error: {e}")
        return None


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
            
            # Extract mouth landmarks
            face_bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            
            # Try MediaPipe first, fallback to heuristic method
            mouth_landmarks = _extract_mouth_landmarks_mediapipe(img, face_bbox)
            if not mouth_landmarks:
                mouth_landmarks = _extract_mouth_landmarks_heuristic(img, face_bbox)
            
            face_data = {
                "bbox": face_bbox,
                "center": [cx, cy],
                "uploaded_id": uploaded_id,
                "filename": face_fname
            }
            
            # Add mouth landmarks if available
            if mouth_landmarks:
                face_data["mouth_landmarks"] = mouth_landmarks
            
            frame_faces.append(face_data)

        return {"frame": f"frame_{frame_idx:05d}.jpg", "faces": frame_faces}
    except Exception as e:
        print(f"[process_frame_for_faces] Error processing frame {frame_idx}: {e}")
        return {"frame": f"frame_{frame_idx:05d}.jpg", "faces": []}


def detect_and_upload_faces(video_path: str, jobId: str = "", fps: int = 1, margin_pct: float = 0.1, 
                           backend_url: str = 'http://localhost:3000', landmarks_path: str = None):
    """
    Extract frames in-memory, detect faces and upload crops to backend; returns landmarks structure.
    
    Args:
        video_path: Path to video file
        jobId: Job identifier
        fps: Frames per second to extract
        margin_pct: Margin percentage for face crops
        backend_url: Backend URL for uploading faces
        landmarks_path: Optional path to save landmarks JSON file
    
    Returns:
        Landmarks dictionary
    """
    frames = extract_frames_in_memory(video_path, fps=fps)
    results = {"jobId": jobId, "frames": []}
    if not frames:
        print(f"[detect_and_upload_faces] No frames extracted from {video_path}")
        return results

    print(f"[detect_and_upload_faces] Processing {len(frames)} frames and uploading faces to {backend_url}")
    for (idx, frame) in tqdm(frames):
        res = _detect_and_upload_from_frame(idx, frame, jobId, margin_pct=margin_pct, backend_url=backend_url)
        results['frames'].append(res)
    
    # Save landmarks to file if path provided
    if landmarks_path:
        try:
            landmarks_dir = os.path.dirname(landmarks_path)
            if landmarks_dir:
                os.makedirs(landmarks_dir, exist_ok=True)
                print(f"[detect_and_upload_faces] Created landmarks directory: {landmarks_dir}")
            
            # Write landmarks JSON
            with open(landmarks_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Verify file was written
            if os.path.exists(landmarks_path):
                file_size = os.path.getsize(landmarks_path)
                print(f"[detect_and_upload_faces] [OK] Saved landmarks to {landmarks_path} ({file_size} bytes)")
                print(f"[detect_and_upload_faces] [OK] Landmarks file verified: {os.path.exists(landmarks_path)}")
                print(f"[detect_and_upload_faces] [OK] Landmarks contains {len(results.get('frames', []))} frames")
            else:
                print(f"[detect_and_upload_faces] [ERROR] Landmarks file not found after writing!")
        except Exception as e:
            print(f"[detect_and_upload_faces] [ERROR] Error saving landmarks: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[detect_and_upload_faces] WARNING: landmarks_path not provided, landmarks will not be saved")

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
    parser.add_argument('--landmarks', type=str, default=None, help='Path to save landmarks JSON (default: <out>/<jobId>/landmarks.json)')

    args = parser.parse_args()

    job_dir = os.path.join(args.out, args.jobId)
    frames_dir = os.path.join(job_dir, 'frames')
    faces_dir = os.path.join(job_dir, 'faces')
    
    # Use provided landmarks path, or default to job_dir
    if args.landmarks:
        landmarks_path = os.path.abspath(args.landmarks)  # Use absolute path
    else:
        landmarks_path = os.path.abspath(os.path.join(job_dir, 'landmarks.json'))
    
    # Ensure landmarks directory exists
    landmarks_dir = os.path.dirname(landmarks_path)
    if landmarks_dir:
        os.makedirs(landmarks_dir, exist_ok=True)

    try:
        print(f"[main] ========================================")
        print(f"[main] Face Preprocessing Configuration")
        print(f"[main] ========================================")
        print(f"[main] video: {args.video}")
        print(f"[main] out base: {args.out}")
        print(f"[main] jobId: {args.jobId}")
        print(f"[main] fps: {args.fps}")
        print(f"[main] workers: {args.workers}")
        print(f"[main] job_dir: {job_dir}")
        print(f"[main] landmarks_path: {landmarks_path}")
        print(f"[main] landmarks_dir: {landmarks_dir}")
        print(f"[main] landmarks_dir exists: {os.path.exists(landmarks_dir)}")
        print(f"[main] ========================================")
        
        # New in-memory processing: detect faces and upload crops to backend without writing face files to disk
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:3000')
        res = detect_and_upload_faces(args.video, jobId=args.jobId, fps=args.fps, margin_pct=0.1, 
                                      backend_url=backend_url, landmarks_path=landmarks_path)
        print(f"[main] [OK] Completed. Uploaded faces for job: {args.jobId}")
        
        # Verify landmarks file was created
        if os.path.exists(landmarks_path):
            file_size = os.path.getsize(landmarks_path)
            print(f"[main] [OK] Landmarks file verified: {landmarks_path} ({file_size} bytes)")
            
            # Quick validation
            try:
                with open(landmarks_path, 'r') as f:
                    landmarks_data = json.load(f)
                print(f"[main] [OK] Landmarks JSON valid: {len(landmarks_data.get('frames', []))} frames")
            except Exception as e:
                print(f"[main] [ERROR] Landmarks JSON invalid: {e}")
        else:
            print(f"[main] [WARNING] Landmarks file not found at {landmarks_path}")
            print(f"[main]   Checking if directory exists: {os.path.exists(landmarks_dir)}")
            if os.path.exists(landmarks_dir):
                print(f"[main]   Directory contents: {os.listdir(landmarks_dir)}")
    except Exception as e:
        print(f"[main] [ERROR] {e}")
        import traceback
        traceback.print_exc()

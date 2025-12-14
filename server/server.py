"""Flask inference server for visual deepfake detection using Xception.

Endpoint: POST /infer_frames
- Accepts JSON: {jobId, faces_folder, batch_size}
- Or multipart ZIP upload with key 'zip' and optional 'jobId' and 'batch_size'

Behavior:
- Loads model at startup
- Uses `model.dataset.FaceFolderDataset` to build DataLoader
- Runs batched inference with no_grad and sigmoid
- Aggregates scores with `model.aggregate` utilities
- Returns structured JSON or error JSON

Notes:
- Uses temporary extraction path: tempfile.gettempdir()/sachai_job/<jobId>/faces
- Allows CORS for localhost
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torch.utils.data import DataLoader

from model.xception_model import load_model, predict_batch
from model.dataset import FaceFolderDataset
from model.aggregate import aggregate_scores, top_k_frames

# Configuration
DEFAULT_BATCH_SIZE = 32
TMP_ROOT = Path(tempfile.gettempdir()) / "sachai_job"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
# Allow CORS - configure for production via CORS_ORIGINS env var
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost,http://localhost:3000").split(",")
CORS(app, resources={r"/*": {"origins": cors_origins}})

# Device (CPU by default, supports CUDA if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load model once at startup
# Check for checkpoint in common locations
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT") or os.getenv("CHECKPOINT_PATH")
if not CHECKPOINT_PATH:
    # Try to find checkpoint in common locations
    possible_paths = [
        "checkpoint.pth",
        "checkpoints/checkpoint.pth",
        "checkpoints/best.pth",
        "model/checkpoint.pth",
        "train/checkpoint.pth",
        "server/checkpoint.pth",
        "../checkpoint.pth",
        os.path.join(os.path.dirname(__file__), "checkpoint.pth"),
        os.path.join(os.path.dirname(__file__), "..", "checkpoint.pth"),
    ]
    for cp_path in possible_paths:
        if os.path.exists(cp_path):
            CHECKPOINT_PATH = cp_path
            break

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading model with checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"Loading model with checkpoint: {CHECKPOINT_PATH or 'None (using untrained model)'}")
    print(f"WARNING: Model without checkpoint will give random/unreliable results!")
MODEL = load_model(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)


def _extract_zip_to(job_id: str, zip_bytes: bytes) -> Path:
    """Extract an in-memory ZIP to a job-specific temporary folder and return the faces folder path."""
    job_tmp = TMP_ROOT / job_id
    faces_dir = job_tmp / "faces"
    # Clean up existing folder for this job
    if job_tmp.exists():
        shutil.rmtree(job_tmp)
    job_tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(job_tmp)
    # If the zip contains a top-level folder, try to locate 'faces' like contents
    # If there's a folder with images, use that; otherwise, use job_tmp.
    for child in job_tmp.iterdir():
        if child.is_dir():
            # Check for image files
            if any(child.glob("**/*.[jJ][pP][gG]")) or any(child.glob("**/*.[pP][nN][gG]")):
                # Move contents into faces_dir
                shutil.move(str(child), str(faces_dir))
                break
    if not faces_dir.exists():
        # If faces_dir not created above, move all images next to job_tmp into faces_dir
        faces_dir.mkdir(parents=True, exist_ok=True)
        for p in job_tmp.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rel = p.relative_to(job_tmp)
                dest = faces_dir / rel.name
                shutil.move(str(p), str(dest))
    return faces_dir


@app.route("/infer_frames", methods=["POST"])
def infer_frames():
    """Handle JSON or multipart ZIP request and return model predictions and aggregation."""
    import json
    import logging
    # #region agent log
    try:
        log_data = {
            "location": "server.py:84",
            "message": "infer_frames endpoint called",
            "data": {
                "hasZip": "zip" in request.files,
                "contentType": request.content_type,
                "checkpointLoaded": CHECKPOINT_PATH is not None,
                "device": DEVICE,
            },
            "timestamp": int(__import__("time").time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }
        with open("c:/SachAi/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except:
        pass
    # #endregion
    # Support multipart ZIP upload
    if "zip" in request.files:
        # Expect optional form fields
        job_id = request.form.get("jobId") or request.form.get("job_id") or "job-unknown"
        batch_size = int(request.form.get("batch_size") or request.form.get("batch") or DEFAULT_BATCH_SIZE)
        zip_file = request.files["zip"].read()
        faces_folder = _extract_zip_to(job_id, zip_file)
    else:
        # Expect JSON body
        try:
            payload = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "Invalid JSON body"}), 400
        if not payload:
            return jsonify({"error": "Empty JSON body"}), 400
        job_id = payload.get("jobId") or payload.get("job_id") or "job-unknown"
        faces_folder = payload.get("faces_folder")
        batch_size = int(payload.get("batch_size") or DEFAULT_BATCH_SIZE)
        if not faces_folder:
            return jsonify({"error": "faces_folder is required"}), 400
        faces_folder = Path(faces_folder)
        if not faces_folder.exists() or not faces_folder.is_dir():
            return jsonify({"error": f"faces_folder not found: {faces_folder}"}), 400

    # Build dataset and dataloader
    try:
        dataset = FaceFolderDataset(str(faces_folder))
    except Exception as exc:
        return jsonify({"error": f"Failed to build dataset: {exc}"}), 500

    if len(dataset) == 0:
        return jsonify({"error": "No images found in faces folder"}), 400

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # Run inference
    frame_scores: List[float] = []
    filenames: List[str] = []
    model = MODEL
    device = torch.device(DEVICE)

    try:
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_tensors, batch_fnames in dataloader:
                # Move to device
                batch_tensors = batch_tensors.to(device)
                probs = predict_batch(model, batch_tensors)
                # Ensure probs on CPU and as flat list
                probs_cpu = probs.detach().cpu().float().view(-1).tolist()
                frame_scores.extend([float(x) for x in probs_cpu])
                filenames.extend([str(f) for f in batch_fnames])
        
        # #region agent log
        try:
            log_data = {
                "location": "server.py:138",
                "message": "Inference completed",
                "data": {
                    "numFrames": len(frame_scores),
                    "frameScores": frame_scores[:5] if len(frame_scores) > 0 else [],  # First 5 scores
                    "minScore": min(frame_scores) if frame_scores else 0,
                    "maxScore": max(frame_scores) if frame_scores else 0,
                    "meanScore": sum(frame_scores) / len(frame_scores) if frame_scores else 0,
                    "checkpointLoaded": CHECKPOINT_PATH is not None,
                },
                "timestamp": int(__import__("time").time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B"
            }
            with open("c:/SachAi/.cursor/debug.log", "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass
        # #endregion
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    # Aggregate and select top-k suspicious frames
    try:
        visual_prob = float(aggregate_scores(frame_scores))
        suspicious = top_k_frames(frame_scores, filenames, k=3)
        visual_scores = [{"file": f, "score": s} for f, s in zip(filenames, frame_scores)]
        
        # #region agent log
        try:
            log_data = {
                "location": "server.py:150",
                "message": "Aggregation completed",
                "data": {
                    "visualProb": visual_prob,
                    "suspiciousFramesCount": len(suspicious),
                    "topScores": [s["score"] for s in suspicious],
                    "checkpointLoaded": CHECKPOINT_PATH is not None,
                },
                "timestamp": int(__import__("time").time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B"
            }
            with open("c:/SachAi/.cursor/debug.log", "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass
        # #endregion

        response = {
            "jobId": job_id,
            "visual_scores": visual_scores,
            "visual_prob": visual_prob,
            "suspicious_frames": [
                {"file": item["file"], "score": item["score"], "rank": item["rank"]} for item in suspicious
            ],
            "meta": {
                "num_frames": len(frame_scores),
                "batch_size": batch_size,
                "model": "xception_v1",
                "checkpoint_loaded": CHECKPOINT_PATH is not None,
                "warning": "Model is untrained - results are unreliable" if CHECKPOINT_PATH is None else None,
            },
        }
        
        # Determine classification based on thresholds
        # Model output: visual_prob = probability of being fake/deepfake (0=authentic, 1=deepfake)
        if visual_prob >= 0.66:
            classification = "DEEPFAKE"
            confidence_level = "HIGH"
        elif visual_prob >= 0.33:
            classification = "SUSPECTED"
            confidence_level = "MEDIUM"
        else:
            classification = "AUTHENTIC"
            confidence_level = "LOW"
        
        # Log final score with threshold interpretation
        print(f"\n{'='*60}")
        print(f"FINAL SCORE (Flask Server):")
        print(f"  visual_prob = {visual_prob:.4f} ({visual_prob*100:.2f}%)")
        print(f"  Classification: {classification} (Confidence: {confidence_level})")
        print(f"  Thresholds: >=0.66=Deepfake, >=0.33=Suspected, <0.33=Authentic")
        print(f"  Job ID: {job_id}")
        print(f"  Number of frames: {len(frame_scores)}")
        print(f"  Suspicious frames found: {len(suspicious)}")
        if suspicious:
            top_scores_str = ', '.join([f"{s['score']:.4f}" for s in suspicious[:3]])
        print(f"  Top suspicious frame scores: [{top_scores_str}]")
        print(f"{'='*60}\n")
        
        return jsonify(response)
    finally:
        # Optionally cleanup temporary extracted files for ZIP uploads
        # If faces_folder is inside TMP_ROOT, remove job folder
        try:
            faces_path = Path(faces_folder)
            if str(faces_path).startswith(str(TMP_ROOT)):
                job_root = faces_path.parent
                if job_root.exists():
                    shutil.rmtree(job_root)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SachAi visual inference server (Flask)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting server on {args.host}:{args.port} using device={DEVICE}")
    app.run(host=args.host, port=args.port, debug=args.debug)

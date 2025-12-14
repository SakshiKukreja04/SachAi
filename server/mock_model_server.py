from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import random

app = FastAPI()

class InferRequest(BaseModel):
    jobId: str
    framesDir: Optional[str] = None
    images: Optional[List[dict]] = None
    audioPath: str

class SuspiciousFrame(BaseModel):
    time: str

class InferResponse(BaseModel):
    score: float
    label: str
    reason: str
    suspiciousFrames: List[SuspiciousFrame] = []

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Mock deepfake detection model server.
    In production, this would run the actual ML model on frames and audio.
    """
    # Simulate model inference with random results
    score = round(random.uniform(0.0, 1.0), 2)
    
    if score > 0.7:
        label = "Deepfake"
        reason = "Simulated: High probability of synthetic face manipulation detected"
    elif score > 0.4:
        label = "Suspicious"
        reason = "Simulated: Some artifacts detected but inconclusive"
    else:
        label = "Real"
        reason = "Simulated: Video appears to be authentic"
    
    # Simulate detecting suspicious frames
    suspicious_frames = []
    if score > 0.5:
        suspicious_frames = [
            {"time": "00:05"},
            {"time": "00:12"},
        ]
    
    return InferResponse(
        score=score,
        label=label,
        reason=reason,
        suspiciousFrames=suspicious_frames
    )


@app.post("/infer_frames", response_model=InferResponse)
async def infer_frames(payload: dict):
    """Accepts `images` as array of {filename, b64} for testing."""
    # For the mock, behave same as /infer
    score = round(random.uniform(0.0, 1.0), 2)
    if score > 0.7:
        label = "Deepfake"
        reason = "Simulated: High probability of synthetic face manipulation detected"
    elif score > 0.4:
        label = "Suspicious"
        reason = "Simulated: Some artifacts detected but inconclusive"
    else:
        label = "Real"
        reason = "Simulated: Video appears to be authentic"

    suspicious_frames = []
    if score > 0.5:
        suspicious_frames = [
            {"time": "00:05", "filename": (payload.get('images') or [])[0].get('filename') if payload.get('images') else None},
        ]

    return InferResponse(
        score=score,
        label=label,
        reason=reason,
        suspiciousFrames=suspicious_frames
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

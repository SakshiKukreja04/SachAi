# Architecture Overview

## Two-Server Architecture

Your application uses a **microservices architecture** with two separate servers:

### 1. Node.js/Express Server (Port 3000)
**Role:** Main API Backend

**Responsibilities:**
- ✅ Receives HTTP requests from frontend (React app)
- ✅ Handles video file uploads
- ✅ Manages job queue (in-memory)
- ✅ Processes videos (extracts frames, audio using FFmpeg)
- ✅ Calls Flask server for ML inference
- ✅ Stores results in MongoDB
- ✅ Serves API endpoints (`/api/analyze`, `/api/status`, etc.)

**Technology:** Node.js, Express, TypeScript

### 2. Flask/Python Server (Port 8000)
**Role:** ML Model Inference Server

**Responsibilities:**
- ✅ Loads the trained Xception deepfake detection model
- ✅ Receives face images from Node server
- ✅ Runs deepfake detection inference
- ✅ Returns probability scores and predictions
- ✅ Handles batch processing of face images

**Technology:** Python, Flask, PyTorch, Xception model

## How They Work Together

```
Frontend (React)
    ↓ HTTP Request
Node.js Server (Port 3000)
    ├─ Receives video upload
    ├─ Extracts frames & faces
    ├─ Stores faces in MongoDB
    └─ HTTP Request → Flask Server (Port 8000)
                        ├─ Loads face images
                        ├─ Runs ML model inference
                        └─ Returns predictions
    ├─ Receives predictions
    ├─ Stores results in MongoDB
    └─ Returns response to Frontend
```

## Why Two Servers?

1. **Separation of Concerns:**
   - Node.js handles web API, file management, database
   - Python handles ML model inference (PyTorch ecosystem)

2. **Technology Stack:**
   - Node.js is better for web APIs, async operations
   - Python is standard for ML/AI (PyTorch, TensorFlow)

3. **Scalability:**
   - Can scale ML server independently
   - Can run multiple ML instances for load balancing

4. **Resource Management:**
   - ML server can use GPU if available
   - Node server stays lightweight

## Communication

The Node server calls the Flask server via HTTP:
- **Endpoint:** `POST http://localhost:8000/infer_frames`
- **Payload:** `{ jobId, faces_folder, batch_size }`
- **Response:** `{ visual_prob, visual_scores, suspicious_frames }`

Configured via `MODEL_SERVER_URL` environment variable.


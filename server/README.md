# SachAi Backend - Deepfake Detection API

A Node.js/TypeScript backend for deepfake detection that integrates with a model server, supports video file uploads and URL processing, and uses an in-memory job queue for asynchronous processing.

## Tech Stack

- **Framework**: Express.js with TypeScript
- **Database**: MongoDB (Mongoose) - optional for local dev
- **Queue**: In-memory job queue (no Redis required)
- **File Upload**: Multer
- **Media Processing**: FFmpeg
- **External Model**: HTTP calls to a FastAPI model server (mock provided)

## Project Structure

```
src/
├── index.ts              # Express app bootstrap
├── models/
│   └── Analysis.ts       # MongoDB schema for analysis jobs
├── routes/
│   ├── analyze.ts        # POST /api/analyze endpoint
│   ├── status.ts         # GET /api/status/:jobId endpoint
│   └── history.ts        # GET /api/history endpoint
├── services/
│   ├── processor.ts      # Job processor
│   └── ffmpegService.ts  # FFmpeg wrapper
├── queues/
│   └── jobQueue.ts       # In-memory job queue
└── utils/
    ├── logger.ts         # Simple logger utility
    └── temp.ts           # Temp directory helper
```

## Installation

### Prerequisites

- Node.js 18+
- FFmpeg (for video frame/audio extraction)
- yt-dlp (optional, for downloading videos from URLs)
- MongoDB (optional, for persistent storage; not needed for basic local dev)

### Local Setup

1. Clone the repository and navigate to server folder:
   ```bash
   cd server
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

   By default, `SKIP_MONGODB=true` allows the server to run without MongoDB. Set it to `false` if you have MongoDB running.

4. Start the backend:
   ```bash
   npm run dev
   ```

   The server will start on `http://localhost:3000` and use an in-memory job queue.


## Running the Backend

### Development Mode (Local - No Dependencies)

Simply run:
```bash
npm run dev
```

The server will start on `http://localhost:3000` with:
- In-memory job queue (no Redis needed)
- Optional MongoDB (can be skipped with `SKIP_MONGODB=true` in `.env`)
- Temporary files stored in `./tmp`

### Production Build

```bash
npm run build
npm start
```

Note: For production with persistent data, set up MongoDB separately and update `.env`.


## API Endpoints

### 1. POST /api/analyze

Upload a video file or provide a URL for analysis.

**Request (File Upload):**
```bash
curl -X POST http://localhost:3000/api/analyze \
  -F "video=@/path/to/video.mp4"
```

**Request (Video URL):**
```bash
curl -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "videoUrl": "https://example.com/video.mp4"
  }'
```

**Response (202 Accepted):**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### 2. GET /api/status/:jobId

Get the status and results of an analysis job.

**Request:**
```bash
curl http://localhost:3000/api/status/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "done",
  "progress": 100,
  "result": {
    "score": 0.82,
    "label": "Deepfake",
    "reason": "High probability of synthetic face manipulation detected",
    "suspiciousFrames": [
      { "time": "00:05" },
      { "time": "00:12" }
    ]
  }
}
```

Possible statuses: `queued`, `processing`, `done`, `failed`

---

### 3. GET /api/history

Get the last 20 analysis jobs.

**Request:**
```bash
curl http://localhost:3000/api/history
```

**Response:**
```json
[
  {
    "_id": "...",
    "jobId": "550e8400-e29b-41d4-a716-446655440000",
    "createdAt": "2024-01-10T12:30:00Z",
    "status": "done",
    "result": {
      "label": "Deepfake",
      "score": 0.82
    }
  }
]
```

---

### 4. GET /health

Health check endpoint.

**Request:**
```bash
curl http://localhost:3000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

## Mock Model Server

The mock FastAPI server simulates a deepfake detection model.

### Running Separately

```bash
pip install fastapi uvicorn
python mock_model_server.py
```

Then set `MODEL_SERVER_URL=http://localhost:8000` in your `.env`.

### Mock Server Endpoints

**POST /infer**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "jobId": "test-job",
    "framesDir": "/path/to/frames",
    "audioPath": "/path/to/audio.wav"
  }'
```

Response:
```json
{
  "score": 0.68,
  "label": "Suspicious",
  "reason": "Simulated: Some artifacts detected but inconclusive",
  "suspiciousFrames": [
    { "time": "00:05" },
    { "time": "00:12" }
  ]
}
```

**GET /health**
```bash
curl http://localhost:8000/health
```

## Job Processing Pipeline

When a job is enqueued, the processor performs these steps:

1. **Download** (if URL): Download video using yt-dlp or HTTP fallback
2. **Extract Media**: Use FFmpeg to extract frames (1 fps) and audio (WAV)
3. **Call Model**: POST to model server with frames and audio paths
4. **Store Result**: Update MongoDB with score, label, and suspicious frames
5. **Cleanup**: Delete temporary files

If any step fails, status is set to `failed` and error is recorded.

## Environment Variables

```env
PORT=3000                          # Server port
MONGO_URI=mongodb://localhost:27017/sachai-backend
SKIP_MONGODB=true                  # Set to false if MongoDB is running
MODEL_SERVER_URL=http://localhost:8000
TEMP_DIR=./tmp                     # Temp directory for uploads
```

## npm Scripts

```bash
npm run dev      # Start with ts-node-dev (watch mode)
npm run build    # Compile TypeScript to JavaScript
npm start        # Run compiled JS (for production)
```

## Example Workflow

```bash
# 1. Upload a video for analysis
JOB_ID=$(curl -s -X POST http://localhost:3000/api/analyze \
  -F "video=@test-video.mp4" | jq -r '.jobId')

# 2. Poll status until done
while true; do
  curl http://localhost:3000/api/status/$JOB_ID
  sleep 2
done

# 3. View analysis history
curl http://localhost:3000/api/history
```

## Notes

- Temporary files are stored in `./tmp` and cleaned up after processing
- The mock model server returns random results; replace with real model in production
- FFmpeg must be installed for frame/audio extraction
- Set `NODE_ENV=production` for production deployments
- MongoDB and Redis must be accessible from the backend

## License

MIT

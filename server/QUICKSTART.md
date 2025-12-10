# Quick Start Guide

## Project Generated Successfully ✓

The **sachai-backend** deepfake detection backend has been fully scaffolded with all required components.

## What's Included

✓ **Express + TypeScript** - REST API framework  
✓ **MongoDB + Mongoose** - Document storage for analysis metadata  
✓ **Bull + Redis** - Job queue for async processing  
✓ **Multer** - File upload handling  
✓ **FFmpeg Integration** - Video frame/audio extraction  
✓ **Mock Model Server** - FastAPI stub for deepfake detection  
✓ **Docker Compose** - Complete local environment setup  

## Project Structure

```
src/
├── index.ts                 # Express app bootstrap
├── models/Analysis.ts       # MongoDB schema
├── routes/
│   ├── analyze.ts           # POST /api/analyze
│   ├── status.ts            # GET /api/status/:jobId
│   └── history.ts           # GET /api/history
├── services/processor.ts    # Job processing pipeline
├── queues/jobQueue.ts       # Bull queue setup
└── utils/
    ├── logger.ts
    └── temp.ts

Docker files:
├── Dockerfile               # Backend container
├── docker-compose.yml       # Full stack (Mongo, Redis, Backend, Model)
└── mock_model_server.py     # FastAPI mock model
```

## Next Steps

### Option A: Run Everything with Docker (Recommended)

```bash
docker-compose up --build
```

Services will be available:
- Backend: http://localhost:3000
- MongoDB: localhost:27017
- Redis: localhost:6379
- Mock Model: http://localhost:8000

### Option B: Run Locally (Requires Redis, MongoDB, FFmpeg)

1. Start MongoDB:
   ```bash
   docker run -d -p 27017:27017 mongo:7.0
   ```

2. Start Redis:
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. Start Mock Model Server (separate terminal):
   ```bash
   pip install fastapi uvicorn
   python mock_model_server.py
   ```

4. Start Backend:
   ```bash
   npm run dev
   ```

## Test the API

Upload a test video:
```bash
curl -X POST http://localhost:3000/api/analyze \
  -F "video=@path/to/video.mp4"
```

You'll get back a `jobId`. Check status:
```bash
curl http://localhost:3000/api/status/{jobId}
```

View history:
```bash
curl http://localhost:3000/api/history
```

## Files Generated

**Backend Code:**
- src/index.ts, src/models/Analysis.ts
- src/routes/{analyze,status,history}.ts
- src/services/processor.ts
- src/queues/jobQueue.ts
- src/utils/{logger,temp}.ts

**Configuration:**
- package.json (with all dependencies)
- tsconfig.json
- .env.example
- Dockerfile
- docker-compose.yml

**Other:**
- mock_model_server.py (FastAPI mock)
- README.md (detailed documentation)
- QUICKSTART.md (this file)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/analyze | Start analysis (upload file or URL) |
| GET | /api/status/:jobId | Get analysis status & results |
| GET | /api/history | Get last 20 analyses |
| GET | /health | Health check |

## Architecture Overview

```
Client Upload/URL
       ↓
POST /api/analyze
       ↓
Validate & Create DB doc → Enqueue to Bull
       ↓
Worker Process (Bull)
       ├→ Download video (if URL)
       ├→ Extract frames + audio (FFmpeg)
       ├→ Call Model Server (HTTP POST)
       ├→ Update DB with results
       └→ Cleanup temp files
       ↓
GET /api/status/:jobId (polling)
```

## Key Features

- **Async Processing**: Bull queue with Redis for handling long operations
- **File Upload**: Multer with temp directory management
- **URL Support**: Download videos from URLs (yt-dlp + HTTP fallback)
- **Model Integration**: HTTP calls to external inference server
- **Media Processing**: FFmpeg for frame extraction (1 fps) and audio extraction
- **Persistent Storage**: MongoDB stores analysis metadata and results
- **Error Handling**: Try/catch with status updates and cleanup on failure
- **Scalable**: Worker processes handled by Bull; can add multiple workers

## Environment Variables

See `.env.example` for full list. Key vars:

```env
PORT=3000
MONGO_URI=mongodb://localhost:27017/sachai-backend
REDIS_URL=redis://localhost:6379
MODEL_SERVER_URL=http://localhost:8000
TEMP_DIR=./tmp
```

## npm Scripts

```bash
npm run dev      # Development with ts-node-dev (watch mode)
npm run build    # Compile TypeScript to dist/
npm start        # Run compiled JS
```

## Notes

- TypeScript compiled successfully ✓
- All dependencies installed ✓
- Ready to run with `npm run dev` or `docker-compose up` ✓
- FFmpeg must be installed for video processing (included in Docker image)
- Temp files automatically cleaned up after processing

---

**Next**: Choose Option A or B above and start the backend!

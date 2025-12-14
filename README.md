# SachAi - Deepfake Detection System

A comprehensive deepfake detection system using Xception neural network, featuring a modern web interface and robust backend infrastructure.

## 🚀 Features

- **Real-time Deepfake Detection**: Analyze videos using trained Xception model
- **Modern Web UI**: React-based frontend with beautiful, responsive design
- **Two-Server Architecture**: Node.js backend + Flask ML inference server
- **Face Extraction**: Automatic face detection and cropping from video frames
- **Batch Processing**: Efficient batch inference for multiple frames
- **Training Pipeline**: Complete workflow for training custom models on FaceForensics dataset
- **Multi-Platform Support**: Analyze videos from YouTube, Instagram Reels, Vimeo, or upload directly

## 📁 Project Structure

```
SachAi/
├── client/          # React frontend application
├── server/          # Node.js backend + Flask ML server
│   ├── src/         # TypeScript backend code
│   ├── model/       # PyTorch model definitions
│   ├── train/       # Training scripts
│   └── scripts/     # Utility scripts
└── README.md        # This file
```

## 🛠️ Tech Stack

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS
- Shadcn/ui components

### Backend
- Node.js + Express (TypeScript)
- Flask (Python)
- PyTorch + Xception model
- MongoDB (optional, with in-memory fallback)

## 📋 Prerequisites

- Node.js 18+
- Python 3.8+
- FFmpeg (for video processing)
- yt-dlp (for downloading videos from YouTube, Instagram, Vimeo)
- MongoDB (optional)

### Installing yt-dlp

**Windows:**
```bash
pip install yt-dlp
```

**macOS:**
```bash
brew install yt-dlp
# or
pip install yt-dlp
```

**Linux:**
```bash
pip install yt-dlp
# or
sudo apt-get install yt-dlp
```

## 🚀 Quick Start

### 1. Install Dependencies

**Frontend:**
```bash
cd client
npm install
```

**Backend:**
```bash
cd server
npm install
pip install -r requirements.txt
```

### 2. Start Servers

**Flask ML Server (Port 8000):**
```bash
cd server
python server.py --port 8000
```

**Node.js Backend (Port 3000):**
```bash
cd server
npm start
```

**Frontend (Port 5173):**
```bash
cd client
npm run dev
```

### 3. Access the Application

Open http://localhost:5173 in your browser.

## 🎓 Training Your Own Model

See [server/scripts/RUN_FACEFORENSICS.md](server/scripts/RUN_FACEFORENSICS.md) for complete training instructions.

Quick training:
```bash
cd server
python train/train_faceforensics.py \
  --data_dir "path/to/training_data/train" \
  --checkpoint_out "checkpoint.pth" \
  --epochs 3 \
  --batch_size 32
```

## 📚 Documentation

- [Architecture Overview](server/ARCHITECTURE.md)
- [Training Guide](server/TRAINING_GUIDE.md)
- [FaceForensics Setup](server/FACEFORENSICS_SETUP.md)
- [Quick Start Guide](server/QUICKSTART.md)

## 🔧 Configuration

Set environment variables:
- `MODEL_CHECKPOINT`: Path to trained model checkpoint
- `MODEL_SERVER_URL`: Flask server URL (default: http://localhost:8000)
- `SKIP_MONGODB`: Set to 'true' to use in-memory storage




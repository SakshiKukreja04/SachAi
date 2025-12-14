# Deployment on Render - Guide & Considerations

## ⚠️ Important: Render Deployment Challenges

Deploying this two-server architecture on Render has **several challenges**:

### Challenges

1. **Two Services Required:**
   - Render typically allows one service per repo
   - You'll need to deploy as **two separate services** or use a monorepo setup

2. **GPU Requirements:**
   - ML inference (Flask server) benefits from GPU
   - Render **does NOT provide GPU instances** (only CPU)
   - Model inference will be **slow on CPU** (10-30 seconds per video)

3. **File Storage:**
   - Videos and face images need persistent storage
   - Render's filesystem is **ephemeral** (resets on restart)
   - Need external storage (S3, Cloudinary, etc.)

4. **Service Communication:**
   - Node server needs to reach Flask server
   - Need to configure internal URLs or public URLs
   - CORS configuration for cross-service calls

5. **Build Requirements:**
   - Python dependencies (PyTorch, etc.) are **large** (~2GB)
   - Long build times on Render
   - May hit memory limits during build

## ✅ Solutions & Workarounds

### Option 1: Deploy as Two Services (Recommended)

**Service 1: Node.js Backend**
- Type: Web Service
- Build Command: `cd server && npm install && npm run build`
- Start Command: `cd server && npm start`
- Environment Variables:
  ```
  PORT=3000
  MONGO_URI=your_mongodb_uri
  MODEL_SERVER_URL=https://your-flask-service.onrender.com
  ```

**Service 2: Flask ML Server**
- Type: Web Service
- Build Command: `cd server && pip install -r requirements.txt`
- Start Command: `cd server && python server.py --host 0.0.0.0 --port $PORT`
- Environment Variables:
  ```
  PORT=8000
  MODEL_CHECKPOINT=checkpoint.pth
  ```
- **Note:** Upload `checkpoint.pth` as a file in Render

### Option 2: Combine into Single Service (Alternative)

Create a single service that runs both servers:

**Create `start.sh`:**
```bash
#!/bin/bash
# Start Flask server in background
python server.py --host 0.0.0.0 --port 8000 &
# Start Node server
cd server && npm start
```

**Or use a process manager like `foreman` or `supervisord`**

### Option 3: Use Different Platform (Better for ML)

Consider platforms with GPU support:
- **AWS EC2** (GPU instances)
- **Google Cloud Run** (with GPU)
- **Azure Container Instances** (GPU support)
- **Railway** (simpler, but still CPU-only)
- **Fly.io** (supports GPU in some regions)

## 🚀 Step-by-Step Render Deployment

### Prerequisites

1. **External Storage Setup:**
   - Set up AWS S3, Cloudinary, or similar
   - Update code to use external storage instead of local filesystem

2. **MongoDB Atlas:**
   - Use MongoDB Atlas (cloud MongoDB)
   - Get connection string

3. **Checkpoint File:**
   - Upload trained `checkpoint.pth` to external storage
   - Or include in repo (if < 100MB)

### Deployment Steps

#### 1. Prepare for Render

**Update `server.py` for Render:**
```python
# Change CORS to allow Render URLs
CORS(app, resources={r"/*": {"origins": ["*"]}})  # Or specific Render URLs
```

**Update file paths to use environment variables:**
```python
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")
```

#### 2. Deploy Flask Service

1. Create new Web Service on Render
2. Connect your GitHub repo
3. Settings:
   - **Name:** `sachai-ml-server`
   - **Root Directory:** `server`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python server.py --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   - `PORT=8000`
   - `MODEL_CHECKPOINT=checkpoint.pth`
5. Deploy

#### 3. Deploy Node Service

1. Create new Web Service on Render
2. Connect same GitHub repo
3. Settings:
   - **Name:** `sachai-backend`
   - **Root Directory:** `server`
   - **Environment:** `Node`
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm start`
4. Add Environment Variables:
   - `PORT=3000`
   - `MONGO_URI=your_mongodb_atlas_uri`
   - `MODEL_SERVER_URL=https://sachai-ml-server.onrender.com`
   - `TEMP_DIR=/tmp`
5. Deploy

#### 4. Update Frontend

Update frontend API URL to point to Render backend:
```typescript
const API_URL = 'https://sachai-backend.onrender.com';
```

## ⚡ Performance Considerations

### CPU-Only Inference

Without GPU, inference will be **slow**:
- **With GPU:** ~1-2 seconds per video
- **Without GPU (CPU):** ~10-30 seconds per video

### Optimizations

1. **Use smaller batch sizes:**
   ```python
   batch_size = 16  # Instead of 32
   ```

2. **Reduce image resolution:**
   ```python
   transforms.Resize(224)  # Instead of 299
   ```

3. **Cache model predictions:**
   - Store results for identical videos

4. **Use async processing:**
   - Return job ID immediately
   - Process in background
   - Frontend polls for results

## 💰 Cost Considerations

Render free tier limitations:
- **750 hours/month** (enough for one service)
- **Two services = need paid plan** (~$7/month each = $14/month)
- **No GPU** (CPU-only, slower)

Alternative: Use Render for Node.js, deploy Flask on cheaper GPU platform.

## 🔧 Recommended Setup for Production

1. **Node.js Backend:** Render (or Railway, Fly.io)
2. **Flask ML Server:** AWS EC2 GPU instance (or similar)
3. **Database:** MongoDB Atlas (free tier available)
4. **File Storage:** AWS S3 or Cloudinary
5. **Frontend:** Vercel, Netlify, or Render

## 📝 Checklist Before Deploying

- [ ] Update CORS settings for production URLs
- [ ] Set up external file storage (S3, etc.)
- [ ] Configure MongoDB Atlas
- [ ] Upload checkpoint.pth to accessible location
- [ ] Update MODEL_SERVER_URL to production URL
- [ ] Test both services locally with production configs
- [ ] Set up environment variables on Render
- [ ] Configure health checks
- [ ] Set up monitoring/logging

## 🆘 Troubleshooting

**Issue: Services can't communicate**
- Check MODEL_SERVER_URL is correct
- Verify CORS settings
- Check Render service URLs

**Issue: Model too slow**
- Expected on CPU
- Consider using GPU platform for ML server
- Optimize batch size and image resolution

**Issue: Files disappear**
- Use external storage (S3, etc.)
- Don't rely on local filesystem

**Issue: Build fails**
- Check Python/Node versions
- Verify all dependencies in requirements.txt
- Check build logs for errors


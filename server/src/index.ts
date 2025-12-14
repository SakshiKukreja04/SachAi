import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import 'dotenv/config';
import { initQueue } from './queues/jobQueue';
import analyzeRoute from './routes/analyze';
import statusRoute from './routes/status';
import historyRoute from './routes/history';
import imagesRoute from './routes/images';
import internalRoute from './routes/internal';
import { logger } from './utils/logger';

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Serve temporary files (frames/faces) so frontend can fetch thumbnails
const tempDir = process.env.TEMP_DIR || './tmp';
import path from 'path';
app.use('/api/tmp', express.static(path.resolve(tempDir)));

// MongoDB connection (optional for local dev)
let mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27017/sachai-backend';
// Fix malformed query parameters (e.g., ?=Cluster0 should be removed or fixed)
if (mongoUri.includes('?=')) {
  // Remove invalid query parameters like ?=value
  mongoUri = mongoUri.replace(/\?=[^&]*(&|$)/g, '?').replace(/\?$/, '');
  // If query string is now empty, remove the ?
  if (mongoUri.endsWith('?')) {
    mongoUri = mongoUri.slice(0, -1);
  }
}
const skipMongoDB = process.env.SKIP_MONGODB === 'true';

// Initialize MongoDB connection before starting server
async function startServer() {
  if (!skipMongoDB) {
    try {
      await mongoose.connect(mongoUri, {
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS: 45000,
        bufferCommands: false, // Disable mongoose buffering - fail immediately if not connected
      });
      logger.info('MongoDB connected');
    } catch (err: any) {
      logger.warn('MongoDB connection failed (proceeding without DB):', err.message);
      // Set SKIP_MONGODB to true to use in-memory fallback
      process.env.SKIP_MONGODB = 'true';
    }
  }

  // Initialize in-memory job queue
  const queue = initQueue();

  // Routes
  app.use('/api', analyzeRoute);
  app.use('/api', statusRoute);
  app.use('/api', historyRoute);
  // Serve images stored in MongoDB
  app.use('/images', imagesRoute);
  // Internal endpoints used by preprocessor
  app.use('/internal', internalRoute);

  // Health check
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', mongoDBSkipped: skipMongoDB });
  });

  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    logger.info(`Server running on port ${PORT}`);
  });
}

// Start the server
startServer().catch(err => {
  logger.error('Failed to start server:', err);
  process.exit(1);
});

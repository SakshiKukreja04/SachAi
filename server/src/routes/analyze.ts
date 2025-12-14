import { Router, Request, Response } from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { Analysis } from '../models/Analysis';
import { getQueue } from '../queues/jobQueue';
import { logger } from '../utils/logger';
import mongoose from 'mongoose';

const router = Router();

// Setup multer for temp storage
const tempDir = process.env.TEMP_DIR || './tmp';
const upload = multer({
  dest: tempDir,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500 MB
});

router.post('/analyze', upload.single('video'), async (req: Request, res: Response) => {
  try {
    const jobId = uuidv4();
    let filePath: string | undefined;
    let videoUrl: string | undefined;

    // Handle file upload
    if (req.file) {
      filePath = req.file.path;
      logger.info(`File uploaded: ${filePath}`);
    }

    // Handle video URL
    if (req.body.videoUrl) {
      videoUrl = req.body.videoUrl;
    }

    // Validate input
    if (!filePath && !videoUrl) {
      return res.status(400).json({ error: 'Either file or videoUrl is required' });
    }

    // Create analysis document
    // Check if MongoDB is connected or if we should use in-memory fallback
    if (mongoose.connection.readyState !== 1 && process.env.SKIP_MONGODB !== 'true') {
      // Force use of in-memory fallback
      process.env.SKIP_MONGODB = 'true';
    }
    
    const analysis = await Analysis.create({
      jobId,
      status: 'queued',
      filePath,
      videoUrl,
      progress: 0,
    });

    // Enqueue job
    const q = getQueue();
    if (q) {
      await q.add('analyze', { jobId, filePath, videoUrl }, { jobId, removeOnComplete: true });
    }

    logger.info(`Job enqueued: ${jobId}`);

    return res.status(202).json({ jobId });
  } catch (error) {
    logger.error('Error in /analyze:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;

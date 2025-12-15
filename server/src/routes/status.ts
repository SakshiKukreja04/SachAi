import { Router, Request, Response } from 'express';
import { Analysis } from '../models/Analysis';
import { logger } from '../utils/logger';
import mongoose from 'mongoose';

const router = Router();

router.get('/status/:jobId', async (req: Request, res: Response) => {
  try {
    const { jobId } = req.params;
    
    // Check if MongoDB is connected or if we should use in-memory fallback
    if (mongoose.connection.readyState !== 1 && process.env.SKIP_MONGODB !== 'true') {
      // Force use of in-memory fallback
      process.env.SKIP_MONGODB = 'true';
    }
    
    const analysis = await Analysis.findOne({ jobId });

    if (!analysis) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Log result data for debugging
    logger.info(`\n${'='.repeat(60)}`);
    logger.info(`Status endpoint called for jobId: ${jobId}`);
    logger.info(`Status: ${analysis.status}, Progress: ${analysis.progress}`);
    
    if (analysis.result) {
      logger.info(`Result structure:`);
      logger.info(`  - visual_prob: ${analysis.result.visual_prob ?? 'undefined'}`);
      logger.info(`  - audio_sync_score: ${(analysis.result as any).audio_sync_score ?? 'undefined'}`);
      logger.info(`  - final_prob: ${(analysis.result as any).final_prob ?? 'undefined'}`);
      logger.info(`  - score (legacy): ${analysis.result.score ?? 'undefined'}`);
      logger.info(`  - suspiciousFrames count: ${analysis.result.suspiciousFrames?.length ?? 0}`);
      if (analysis.result.visual_prob !== undefined) {
        const score = analysis.result.visual_prob;
        logger.info(`  - visual_prob value: ${score.toFixed(4)} (${(score * 100).toFixed(2)}%)`);
      }
      if ((analysis.result as any).audio_sync_score !== undefined && (analysis.result as any).audio_sync_score !== null) {
        const audioScore = (analysis.result as any).audio_sync_score;
        logger.info(`  - audio_sync_score value: ${audioScore.toFixed(4)} (${(audioScore * 100).toFixed(2)}%)`);
      }
      logger.info(`Full result object keys: ${Object.keys(analysis.result).join(', ')}`);
    } else {
      logger.info(`Result: null or undefined`);
    }
    logger.info(`${'='.repeat(60)}\n`);

    return res.json({
      jobId: analysis.jobId,
      status: analysis.status,
      progress: analysis.progress,
      result: analysis.result,
      error: analysis.error,
    });
  } catch (error) {
    logger.error('Error in /status:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;

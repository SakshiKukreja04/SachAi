import { Router, Request, Response } from 'express';
import { Analysis } from '../models/Analysis';
import { logger } from '../utils/logger';

const router = Router();

router.get('/history', async (req: Request, res: Response) => {
  try {
    const analyses = await Analysis.find()
      .sort({ createdAt: -1 })
      .limit(20)
      .select('jobId createdAt result.label result.score status');

    return res.json(analyses);
  } catch (error) {
    logger.error('Error in /history:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;

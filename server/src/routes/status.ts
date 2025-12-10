import { Router, Request, Response } from 'express';
import { Analysis } from '../models/Analysis';
import { logger } from '../utils/logger';

const router = Router();

router.get('/status/:jobId', async (req: Request, res: Response) => {
  try {
    const { jobId } = req.params;
    const analysis = await Analysis.findOne({ jobId });

    if (!analysis) {
      return res.status(404).json({ error: 'Job not found' });
    }

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

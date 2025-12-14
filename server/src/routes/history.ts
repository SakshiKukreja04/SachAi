import { Router, Request, Response } from 'express';
import { Analysis } from '../models/Analysis';
import { logger } from '../utils/logger';

const router = Router();

// Get all analysis history
router.get('/history', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const analyses = await Analysis.find({
      status: 'done'  // Only show completed analyses
    })
      .sort({ createdAt: -1 })
      .limit(limit)
      .select('jobId createdAt updatedAt videoUrl filePath result status');

    // Transform data for frontend
    const history = analyses.map((analysis: any) => {
      const result = analysis.result || {};
      const finalProb = result.final_prob ?? result.visual_prob ?? result.score ?? 0;
      const scorePercent = Math.round(finalProb * 100);
      
      // Determine label from classification or compute from score
      let label: string = result.classification?.toLowerCase() || 'unknown';
      if (label === 'unknown') {
        if (finalProb >= 0.55) {
          label = 'deepfake';
        } else if (finalProb >= 0.30) {
          label = 'suspected';
        } else {
          label = 'authentic';
        }
      }
      
      // Get thumbnail from first suspicious frame if available
      let thumbnail: string | null = null;
      if (result.suspiciousFrames && result.suspiciousFrames.length > 0) {
        const firstFrame = result.suspiciousFrames[0];
        if (firstFrame.id) {
          thumbnail = `${process.env.API_URL || 'http://localhost:3000'}/images/${firstFrame.id}`;
        } else if (firstFrame.file && analysis.jobId) {
          // Fallback: use file path if no image ID
          thumbnail = `${process.env.API_URL || 'http://localhost:3000'}/api/tmp/${analysis.jobId}/faces/${firstFrame.file}`;
        }
      }

      return {
        id: analysis.jobId,
        date: new Date(analysis.createdAt).toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: 'numeric',
          minute: '2-digit',
        }),
        score: scorePercent,
        label: label as 'authentic' | 'suspected' | 'deepfake',
        thumbnail: thumbnail,
        videoUrl: analysis.videoUrl,
        createdAt: analysis.createdAt,
      };
    });

    return res.json(history);
  } catch (error) {
    logger.error('Error in /history:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// Mark report as saved
router.post('/history/:jobId/save', async (req: Request, res: Response) => {
  try {
    const { jobId } = req.params;
    
    // Update analysis to mark as saved (we can add a saved field if needed)
    // For now, we'll just return success since the report is already in the database
    const analysis = await Analysis.findOne({ jobId });
    
    if (!analysis) {
      return res.status(404).json({ error: 'Analysis not found' });
    }

    logger.info(`Report saved: ${jobId}`);
    return res.json({ 
      success: true, 
      message: 'Report saved successfully',
      jobId 
    });
  } catch (error) {
    logger.error('Error saving report:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;

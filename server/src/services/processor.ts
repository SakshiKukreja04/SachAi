import axios from 'axios';
import { execSync, spawn } from 'child_process';
import path from 'path';
import fs from 'fs-extra';
import { Analysis } from '../models/Analysis';
import { logger } from '../utils/logger';
import { getTempJobDir, cleanupTempJobDir } from '../utils/temp';
import http from 'http';

const MODEL_SERVER_URL = process.env.MODEL_SERVER_URL || 'http://localhost:8000';

interface JobData {
  jobId: string;
  filePath?: string;
  videoUrl?: string;
}

interface Job {
  id: string;
  data: JobData;
}

// Simple HTTP GET download fallback
async function downloadVideo(url: string, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(outputPath);
    http
      .get(url, (response) => {
        response.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      })
      .on('error', reject);
  });
}

// Download video if videoUrl is provided
async function ensureVideoFile(jobId: string, filePath?: string, videoUrl?: string): Promise<string> {
  if (filePath && fs.existsSync(filePath)) {
    return filePath;
  }

  if (videoUrl) {
    const jobDir = getTempJobDir(jobId);
    const videoPath = path.join(jobDir, 'video.mp4');

    logger.info(`Downloading video from ${videoUrl}...`);
    
    // Try yt-dlp first
    try {
      execSync(`yt-dlp -f best -o "${videoPath}" "${videoUrl}"`, { stdio: 'inherit' });
      return videoPath;
    } catch (err) {
      logger.warn('yt-dlp failed, falling back to HTTP download');
      try {
        await downloadVideo(videoUrl, videoPath);
        return videoPath;
      } catch (downloadErr) {
        throw new Error(`Failed to download video: ${downloadErr}`);
      }
    }
  }

  throw new Error('No video file or URL provided');
}

// Extract frames and audio using FFmpeg
async function extractMediaFiles(jobId: string, videoPath: string): Promise<{ framesDir: string; audioPath: string }> {
  const jobDir = getTempJobDir(jobId);
  const framesDir = path.join(jobDir, 'frames');
  const audioPath = path.join(jobDir, 'audio.wav');

  fs.ensureDirSync(framesDir);

  logger.info(`Extracting frames from ${videoPath}...`);
  try {
    // Extract frames at 1 fps
    execSync(`ffmpeg -i "${videoPath}" -vf "fps=1" "${path.join(framesDir, 'frame_%04d.png')}"`, {
      stdio: 'inherit',
    });
    logger.info('Frames extracted');
  } catch (err) {
    throw new Error(`FFmpeg frame extraction failed: ${err}`);
  }

  logger.info('Extracting audio...');
  try {
    // Extract audio to WAV
    execSync(`ffmpeg -i "${videoPath}" -q:a 9 -n "${audioPath}"`, { stdio: 'inherit' });
    logger.info('Audio extracted');
  } catch (err) {
    logger.warn(`FFmpeg audio extraction warning: ${err}`);
    // Don't fail if audio extraction fails; audio might not be present
  }

  return { framesDir, audioPath };
}

// Call external model server
async function callModelServer(jobId: string, framesDir: string, audioPath: string): Promise<any> {
  try {
    logger.info(`Calling model server at ${MODEL_SERVER_URL}...`);
    const response = await axios.post(`${MODEL_SERVER_URL}/infer`, {
      jobId,
      framesDir,
      audioPath,
    });
    logger.info('Model server response:', response.data);
    return response.data;
  } catch (error) {
    throw new Error(`Model server call failed: ${error}`);
  }
}

// Main job processor
export async function processAnalysisJob(job: Job): Promise<void> {
  const { jobId, filePath, videoUrl } = job.data;

  try {
    logger.info(`Processing job ${jobId}...`);

    // Update status to processing
    await Analysis.updateOne({ jobId }, { status: 'processing', progress: 10 });

    // Ensure video file exists
    const videoPath = await ensureVideoFile(jobId, filePath, videoUrl);
    await Analysis.updateOne({ jobId }, { progress: 30 });

    // Extract frames and audio
    const { framesDir, audioPath } = await extractMediaFiles(jobId, videoPath);
    await Analysis.updateOne({ jobId }, { progress: 50 });

    // Call model server
    const modelResult = await callModelServer(jobId, framesDir, audioPath);
    await Analysis.updateOne({ jobId }, { progress: 80 });

    // Update analysis with result
    await Analysis.updateOne(
      { jobId },
      {
        status: 'done',
        progress: 100,
        result: {
          score: modelResult.score,
          label: modelResult.label,
          reason: modelResult.reason,
          suspiciousFrames: modelResult.suspiciousFrames,
        },
      }
    );

    logger.info(`Job ${jobId} completed successfully`);

    // Cleanup temp files
    cleanupTempJobDir(jobId);
  } catch (error) {
    logger.error(`Job ${jobId} failed:`, error);
    await Analysis.updateOne(
      { jobId },
      {
        status: 'failed',
        error: (error as Error).message,
      }
    );

    // Cleanup on error
    cleanupTempJobDir(jobId);
  }
}

import axios from 'axios';
import { execSync, spawn } from 'child_process';
import path from 'path';
import fs from 'fs-extra';
import { Analysis } from '../models/Analysis';
import { logger } from '../utils/logger';
import { getTempJobDir, cleanupTempJobDir } from '../utils/temp';
import { extractFrames, extractAudio } from './ffmpegService';
import { getImagesForJob } from './imageService';
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

  try {
    logger.info(`Extracting frames from ${videoPath}...`);
    await extractFrames(videoPath, framesDir, 1); // 1 fps
    logger.info('Frames extracted successfully');
  } catch (err) {
    throw new Error(`Frame extraction failed: ${err}`);
  }

  try {
    logger.info('Extracting audio...');
    await extractAudio(videoPath, audioPath);
    logger.info('Audio extracted successfully');
  } catch (err) {
    logger.warn(`Audio extraction warning (proceeding without audio): ${err}`);
    // Don't fail if audio extraction fails; audio might not be present
  }

  return { framesDir, audioPath };
}

// Spawn the Python preprocessing script to detect/crop faces
async function runPythonPreprocessor(jobId: string, videoPath: string): Promise<{ facesDir: string; landmarksPath: string }> {
  const jobDir = getTempJobDir(jobId);
  const facesDir = path.join(jobDir, 'faces');
  const landmarksPath = path.join(jobDir, 'landmarks.json');

  return new Promise((resolve, reject) => {
    logger.info(`Spawning Python preprocessor for job ${jobId}...`);
    // The Python script expects `--out` to be the base tmp dir (it will create a subdir with jobId).
    // Pass the parent directory of the job dir so the script writes to: <out>/<jobId>/faces
    const outBase = path.dirname(jobDir);
    const py = spawn('python', ['ml/face_preprocess.py', '--video', videoPath, '--out', outBase, '--jobId', jobId, '--fps', '1', '--workers', '2'], { stdio: ['ignore', 'pipe', 'pipe'] });

    let stdout = '';
    let stderr = '';
    if (py.stdout) {
      py.stdout.on('data', (chunk) => {
        stdout += chunk.toString();
        logger.info(chunk.toString().trim());
      });
    }
    if (py.stderr) {
      py.stderr.on('data', (chunk) => {
        stderr += chunk.toString();
        logger.warn(chunk.toString().trim());
      });
    }

    py.on('close', (code) => {
      if (code === 0) {
        // Preprocessor completed successfully. It uploads face crops to MongoDB
        logger.info(`Preprocessor exited successfully for job ${jobId}`);
        resolve({ facesDir, landmarksPath });
      } else {
        reject(new Error(`Preprocessor exited with code ${code}: ${stderr || stdout}`));
      }
    });
  });
}

// Call external model server
async function callModelServer(jobId: string, images: any[], batchSize = 32): Promise<any> {
  const jobDir = getTempJobDir(jobId);
  const facesDir = path.join(jobDir, 'faces');
  
  try {
    // Download images from MongoDB to temp folder
    logger.info(`Downloading ${images.length} images from MongoDB to ${facesDir}...`);
    fs.ensureDirSync(facesDir);
    
    for (const doc of images) {
      const buf = doc.data && doc.data.buffer ? Buffer.from(doc.data.buffer) : Buffer.from(doc.data);
      const filePath = path.join(facesDir, doc.filename);
      await fs.writeFile(filePath, buf);
    }
    
    logger.info(`Calling model server at ${MODEL_SERVER_URL}/infer_frames with faces_folder: ${facesDir}...`);
    
    // #region agent log
    try { const fs = require('fs'); fs.appendFileSync('c:/SachAi/.cursor/debug.log', JSON.stringify({ location: 'processor.ts:155', message: 'Before model server call', data: { jobId, facesDir, imageCount: images.length, modelServerUrl: MODEL_SERVER_URL, imageFilenames: images.slice(0, 3).map((img: any) => img.filename) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run1', hypothesisId: 'A' }) + '\n'); } catch {}
    // #endregion
    
    // Send faces_folder path to Flask server (which expects a directory path)
    const response = await axios.post(`${MODEL_SERVER_URL}/infer_frames`, {
      jobId,
      faces_folder: facesDir,
      batch_size: batchSize,
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // #region agent log
    try { 
      const fs = require('fs');
      const visualScores = response.data?.visual_scores || [];
      const topScores = visualScores.slice(0, 5).map((s: any) => s.score);
      fs.appendFileSync('c:/SachAi/.cursor/debug.log', JSON.stringify({ 
        location: 'processor.ts:170', 
        message: 'Model server response received', 
        data: { 
          jobId, 
          visualProb: response.data?.visual_prob, 
          score: response.data?.score, 
          label: response.data?.label, 
          suspiciousFramesCount: (response.data?.suspicious_frames || response.data?.suspiciousFrames || []).length, 
          modelType: response.data?.meta?.model || 'unknown', 
          hasVisualScores: !!response.data?.visual_scores,
          numFrames: response.data?.meta?.num_frames,
          topFrameScores: topScores,
          minFrameScore: visualScores.length > 0 ? Math.min(...visualScores.map((s: any) => s.score)) : null,
          maxFrameScore: visualScores.length > 0 ? Math.max(...visualScores.map((s: any) => s.score)) : null,
          meanFrameScore: visualScores.length > 0 ? visualScores.reduce((sum: number, s: any) => sum + s.score, 0) / visualScores.length : null,
          responseKeys: Object.keys(response.data || {}) 
        }, 
        timestamp: Date.now(), 
        sessionId: 'debug-session', 
        runId: 'run1', 
        hypothesisId: 'A' 
      }) + '\n'); 
    } catch {}
    // #endregion
    
    logger.info('Model server response received');
    return response.data;
  } catch (error: any) {
    // Enhanced error logging
    const errorMessage = error.response?.data?.error || error.message || 'Unknown error';
    const statusCode = error.response?.status || 'N/A';
    logger.error(`Model server call failed (status ${statusCode}): ${errorMessage}`);
    if (error.response?.data) {
      logger.error('Model server response:', JSON.stringify(error.response.data, null, 2));
    }
    throw new Error(`Model server call failed: ${errorMessage}`);
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

    // Extract audio only (frames will be processed in-memory by preprocessor)
    const jobDir = getTempJobDir(jobId);
    const audioPath = path.join(jobDir, 'audio.wav');
    try {
      await extractAudio(videoPath, audioPath);
      logger.info('Audio extracted successfully');
    } catch (err) {
      logger.warn(`Audio extraction warning (proceeding without audio): ${err}`);
    }
    await Analysis.updateOne({ jobId }, { progress: 50 });

    // Run Python preprocessor to produce face crops (uploader will store them in MongoDB)
    await runPythonPreprocessor(jobId, videoPath);
    await Analysis.updateOne({ jobId }, { progress: 70 });

    // Retrieve images uploaded by preprocessor from MongoDB
    const images = await getImagesForJob(jobId);

    // Call model server with images in-memory
    const modelResult = await callModelServer(jobId, images, 32);
    await Analysis.updateOne({ jobId }, { progress: 90 });

    // Map model result suspicious frames to include DB ids if possible
    const suspicious = modelResult.suspicious_frames || modelResult.suspiciousFrames || [];
    const uploadedFrames = suspicious.map((item: any, idx: number) => {
      // match by filename if provided
      const fname = item.filename || item.file || item.name || null;
      const matched = fname ? images.find((d: any) => d.filename === fname) : images[idx];
      const frameScore = item.score ?? item.prob ?? item.confidence ?? 0;
      // Log each suspicious frame for debugging
      logger.info(`  Suspicious frame ${idx + 1}: file=${fname}, score=${frameScore.toFixed(4)} (${(frameScore * 100).toFixed(2)}%)`);
      return { 
        id: matched?._id?.toString?.() || null, 
        file: fname,  // Use 'file' to match frontend expectation
        filename: matched?.filename || fname, 
        score: frameScore,
        rank: item.rank ?? idx + 1
      };
    });

    // Extract final score
    const finalScore = modelResult.visual_prob || modelResult.score || null;
    
    // Determine classification based on thresholds
    // Model output: visual_prob = probability of being fake/deepfake (0=authentic, 1=deepfake)
    let classification = 'UNKNOWN';
    let confidenceLevel = 'UNKNOWN';
    if (finalScore !== null) {
      if (finalScore >= 0.66) {
        classification = 'DEEPFAKE';
        confidenceLevel = 'HIGH';
      } else if (finalScore >= 0.33) {
        classification = 'SUSPECTED';
        confidenceLevel = 'MEDIUM';
      } else {
        classification = 'AUTHENTIC';
        confidenceLevel = 'LOW';
      }
    }
    
    // Log final score with threshold interpretation
    logger.info(`\n${'='.repeat(60)}`);
    logger.info(`FINAL SCORE (Node.js Backend):`);
    logger.info(`  visual_prob = ${finalScore !== null ? finalScore.toFixed(4) : 'null'} (${finalScore !== null ? (finalScore * 100).toFixed(2) : 'null'}%)`);
    logger.info(`  Classification: ${classification} (Confidence: ${confidenceLevel})`);
    logger.info(`  Thresholds: >=0.66=Deepfake, >=0.33=Suspected, <0.33=Authentic`);
    logger.info(`  Job ID: ${jobId}`);
    logger.info(`  Number of frames: ${modelResult.meta?.num_frames || 'unknown'}`);
    logger.info(`  Suspicious frames: ${suspicious.length}`);
    if (suspicious.length > 0) {
      logger.info(`  Top suspicious frame scores: ${suspicious.slice(0, 3).map((s: any) => (s.score ?? s.prob ?? s.confidence ?? 0).toFixed(4)).join(', ')}`);
    }
    logger.info(`${'='.repeat(60)}\n`);

    // Update analysis with result (map Flask response fields)
    await Analysis.updateOne(
      { jobId },
      {
        status: 'done',
        progress: 100,
        result: {
          visual_prob: finalScore,
          visual_scores: modelResult.visual_scores || null,
          suspiciousFrames: uploadedFrames,
          meta: modelResult.meta || {},
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

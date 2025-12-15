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

// Detect platform from URL
function detectPlatform(url: string): string {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname.toLowerCase();
    
    if (hostname.includes('youtube.com') || hostname.includes('youtu.be')) {
      return 'YouTube';
    }
    if (hostname.includes('instagram.com')) {
      return 'Instagram';
    }
    if (hostname.includes('vimeo.com')) {
      return 'Vimeo';
    }
    return 'Unknown';
  } catch {
    return 'Unknown';
  }
}

// Download video if videoUrl is provided
async function ensureVideoFile(jobId: string, filePath?: string, videoUrl?: string): Promise<string> {
  if (filePath && fs.existsSync(filePath)) {
    return filePath;
  }

  if (videoUrl) {
    const jobDir = getTempJobDir(jobId);
    const videoPath = path.join(jobDir, 'video.mp4');
    const platform = detectPlatform(videoUrl);

    logger.info(`Downloading video from ${platform} (${videoUrl})...`);
    
    // Try multiple methods to run yt-dlp with improved detection
    // Priority: Python module (most reliable) > direct command > executable
    const ytDlpMethods = [
      { cmd: 'python -m yt_dlp', name: 'python -m yt_dlp', test: 'python -m yt_dlp --version' },
      { cmd: 'py -m yt_dlp', name: 'py -m yt_dlp', test: 'py -m yt_dlp --version' },
      { cmd: 'python3 -m yt_dlp', name: 'python3 -m yt_dlp', test: 'python3 -m yt_dlp --version' },
      { cmd: 'yt-dlp', name: 'yt-dlp', test: 'yt-dlp --version' },
      { cmd: 'yt-dlp.exe', name: 'yt-dlp.exe', test: 'yt-dlp.exe --version' },
    ];
    
    let ytDlpExecutable = '';
    let ytDlpVersion = '';
    let pythonCommand: string | null = null; // Store the command that works (e.g., 'python', 'py')
    
    logger.info('Detecting yt-dlp installation...');
    
    // Find Python executable directly using sys.executable (most reliable method)
    // This avoids PATH issues completely
    let pythonExecutable: string | null = null;
    const pythonCommands = ['python', 'py', 'python3'];
    
    logger.info('Finding Python executable using sys.executable...');
    for (const pyCmd of pythonCommands) {
      try {
        const testOptions: any = { 
          stdio: 'pipe',
          encoding: 'utf8' as BufferEncoding,
          timeout: 10000, // 10 second timeout
        };
        
        if (process.platform === 'win32') {
          testOptions.shell = true;
        }
        
        // Get Python executable path directly
        const pythonPath = execSync(`${pyCmd} -c "import sys; print(sys.executable)"`, testOptions).toString().trim();
        
        if (pythonPath && fs.existsSync(pythonPath)) {
          pythonExecutable = pythonPath;
          pythonCommand = pyCmd; // Store the command that works (e.g., 'python')
          logger.info(`[OK] Found Python executable: ${pythonExecutable}`);
          logger.info(`  Python command: ${pythonCommand}`);
          
          // Verify yt_dlp is installed by running --version (this is what actually works)
          // Don't check __version__ attribute as it may not exist in newer versions
          try {
            const versionOutput = execSync(`"${pythonExecutable}" -m yt_dlp --version`, testOptions).toString().trim();
            if (versionOutput) {
              ytDlpVersion = versionOutput;
              ytDlpExecutable = `"${pythonExecutable}" -m yt_dlp`;
              logger.info(`[OK] yt-dlp is installed and working`);
              logger.info(`  Python: ${pythonExecutable}`);
              logger.info(`  Python command: ${pythonCommand}`);
              logger.info(`  yt-dlp version: ${ytDlpVersion}`);
              logger.info(`  Command: ${ytDlpExecutable}`);
              break;
            } else {
              throw new Error('No version output');
            }
          } catch (versionErr: any) {
            logger.warn(`Python found but yt_dlp --version failed for ${pythonExecutable}`);
            logger.debug(`Version check error: ${versionErr.message}`);
            logger.debug(`stderr: ${versionErr.stderr?.toString() || 'none'}`);
            logger.debug(`stdout: ${versionErr.stdout?.toString() || 'none'}`);
            pythonExecutable = null; // Reset to try next Python
            pythonCommand = null;
            continue;
          }
        }
      } catch (err: any) {
        logger.debug(`Python command '${pyCmd}' failed: ${err.message}`);
        continue;
      }
    }
    
    if (!pythonExecutable || !ytDlpExecutable || !pythonCommand) {
      // Provide more helpful error message with installation instructions
      const errorMessage = 
        `yt-dlp is not installed or Python executable could not be found. Please install it to download videos from ${platform}.\n\n` +
        `Installation instructions:\n` +
        `1. Install using pip: pip install yt-dlp\n` +
        `2. Or install using pip with user flag: pip install --user yt-dlp\n` +
        `3. After installation, verify with: python -m yt_dlp --version\n` +
        `4. Make sure Python is installed and accessible\n\n` +
        `Note: If you just installed yt-dlp, you may need to restart the server for it to be detected.`;
      
      logger.error(errorMessage);
      throw new Error(errorMessage);
    }
    
    // Try yt-dlp download (supports YouTube, Instagram, Vimeo, and many others)
    try {
      // Use yt-dlp with better error handling using spawn
      // -f bestvideo+bestaudio/best: Select best quality
      // --merge-output-format mp4: Ensure MP4 output
      // --no-playlist: Download only single video, not playlist
      // --quiet: Reduce output noise
      // --no-warnings: Suppress warnings
      const ytDlpArgs = [
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '--no-playlist',
        '--quiet',
        '--no-warnings',
        '-o', videoPath,
        videoUrl
      ];
      
      logger.info(`Downloading video using: ${ytDlpExecutable}`);
      logger.info(`  URL: ${videoUrl}`);
      logger.info(`  Output: ${videoPath}`);
      
      // Use Python wrapper script with 'python' from PATH (like runPythonPreprocessor does)
      // runPythonPreprocessor successfully uses spawn('python', [...]) - same pattern!
      const downloadResult = await new Promise<string>((resolve, reject) => {
        const pythonPath = pythonExecutable!.replace(/^"|"$/g, ''); // Absolute path for wrapper
        const pyCmd = pythonCommand!; // Command from PATH (e.g., 'python')
        const wrapperScript = path.join(__dirname, '../../scripts/download_video.py');
        const absoluteWrapperScript = path.resolve(wrapperScript);
        
        if (!fs.existsSync(absoluteWrapperScript)) {
          reject(new Error(`Python wrapper script not found: ${absoluteWrapperScript}`));
          return;
        }
        
        logger.info(`Executing yt-dlp via Python wrapper (using 'python' from PATH like preprocessor):`);
        logger.info(`  Python command: ${pyCmd}`);
        logger.info(`  Python path: ${pythonPath}`);
        logger.info(`  Wrapper: ${absoluteWrapperScript}`);
        logger.info(`  Output: ${videoPath}`);
        logger.info(`  URL: ${videoUrl}`);
        
        // Use spawn with 'python' from PATH (same as runPythonPreprocessor)
        // Pass wrapper script and args - Python subprocess handles paths with spaces
        const args = [
          absoluteWrapperScript,
          pythonPath,  // Pass absolute path as argument to wrapper
          videoPath,
          videoUrl
        ];
        
        const spawnOptions: any = {
          cwd: path.join(__dirname, '../../'),  // Same as runPythonPreprocessor
          stdio: ['ignore', 'pipe', 'pipe'],
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUNBUFFERED: '1'
          }
        };
        
        const pythonProcess = spawn(pyCmd, args, spawnOptions);
        
        let stdout = '';
        let stderr = '';
        
        if (pythonProcess.stdout) {
          pythonProcess.stdout.on('data', (data: Buffer) => {
            const text = data.toString();
            stdout += text;
            // Log progress and success messages
            if (text.includes('SUCCESS') || text.includes('%') || text.includes('Downloading')) {
              logger.info(`Python wrapper: ${text.trim()}`);
            }
          });
        }
        
        if (pythonProcess.stderr) {
          pythonProcess.stderr.on('data', (data: Buffer) => {
            const text = data.toString();
            stderr += text;
            // Log all stderr for debugging
            logger.warn(`Python wrapper stderr: ${text.trim()}`);
          });
        }
        
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            // Success - verify file exists
            if (fs.existsSync(videoPath)) {
              const stats = fs.statSync(videoPath);
              logger.info(`[OK] Successfully downloaded ${platform} video: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
              resolve(videoPath);
            } else {
              logger.error(`Python wrapper completed with code 0 but video file not found at: ${videoPath}`);
              logger.error(`stdout: ${stdout}`);
              logger.error(`stderr: ${stderr}`);
              reject(new Error('yt-dlp completed but video file not found'));
            }
          } else {
            // Error - log details
            logger.error(`Python wrapper failed with exit code: ${code}`);
            logger.error(`stdout: ${stdout}`);
            logger.error(`stderr: ${stderr}`);
            logger.error(`Python command used: ${pyCmd}`);
            
            const errorMsg = stderr || stdout || 'Unknown error';
            if (errorMsg.toLowerCase().includes('not found') || errorMsg.toLowerCase().includes('enoent')) {
              reject(new Error(
                `Failed to execute yt-dlp using Python command: ${pyCmd}\n` +
                `This usually means yt-dlp is not installed in this Python environment.\n` +
                `Please install it: ${pyCmd} -m pip install yt-dlp`
              ));
            } else {
              reject(new Error(`yt-dlp download failed: ${errorMsg}`));
            }
          }
        });
        
        pythonProcess.on('error', (err) => {
          logger.error(`Python wrapper spawn error: ${err.message}`);
          logger.error(`Python command used: ${pyCmd}`);
          if (err.message.includes('ENOENT') || err.message.includes('not found')) {
            reject(new Error(
              `Failed to execute Python command: ${pyCmd}\n` +
              `Please verify Python is installed and in your system PATH.`
            ));
          } else {
            reject(new Error(`Python wrapper execution error: ${err.message}`));
          }
        });
      });
      
      return downloadResult;
      
    } catch (err: any) {
      const errorMsg = err.message || String(err);
      logger.error(`yt-dlp failed for ${platform}: ${errorMsg}`);
      
      // If yt-dlp is installed but failed, try HTTP fallback for direct video URLs only
      if (videoUrl.match(/\.(mp4|mov|avi|mkv|webm)(\?|$)/i)) {
        logger.warn('Falling back to HTTP download for direct video URL');
        try {
          await downloadVideo(videoUrl, videoPath);
          if (fs.existsSync(videoPath)) {
            return videoPath;
          }
        } catch (downloadErr) {
          throw new Error(`Failed to download video: ${downloadErr}. yt-dlp error: ${errorMsg}`);
        }
      } else {
        throw new Error(
          `Failed to download from ${platform}. ` +
          `Error: ${errorMsg}. ` +
          `Make sure the URL is valid and accessible.`
        );
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
    // Explicitly pass landmarks path to ensure it's saved
    // The preprocessor expects --out to be the base directory, and it creates <out>/<jobId>
    // So we pass the parent of jobDir as --out
    const outBase = path.dirname(jobDir);
    
    // Use absolute paths to avoid any path resolution issues
    const absoluteVideoPath = path.resolve(videoPath);
    const absoluteOutBase = path.resolve(outBase);
    const absoluteLandmarksPath = path.resolve(landmarksPath);
    
    logger.info(`Preprocessor paths:`);
    logger.info(`  Video: ${absoluteVideoPath}`);
    logger.info(`  Out base: ${absoluteOutBase}`);
    logger.info(`  Job ID: ${jobId}`);
    logger.info(`  Landmarks path: ${absoluteLandmarksPath}`);
    
    const py = spawn('python', [
      'ml/face_preprocess.py', 
      '--video', absoluteVideoPath, 
      '--out', absoluteOutBase, 
      '--jobId', jobId, 
      '--fps', '1', 
      '--workers', '2',
      '--landmarks', absoluteLandmarksPath
    ], { 
      stdio: ['ignore', 'pipe', 'pipe'],
      cwd: path.join(__dirname, '../../'),  // Run from server directory
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',  // Force UTF-8 encoding for Python output
        PYTHONUNBUFFERED: '1'  // Unbuffered output for better logging
      }
    });

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
        
        // Verify landmarks file was created
        if (fs.existsSync(landmarksPath)) {
          const stats = fs.statSync(landmarksPath);
          logger.info(`Landmarks file verified: ${landmarksPath} (${stats.size} bytes)`);
        } else {
          logger.warn(`Landmarks file not found at ${landmarksPath} after preprocessor completion`);
          // Try alternative location
          const altPath = path.join(path.dirname(jobDir), jobId, 'landmarks.json');
          if (fs.existsSync(altPath)) {
            logger.info(`Found landmarks at alternative location: ${altPath}`);
          }
        }
        
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
    
    
    // Get audio path and landmarks path for sync analysis
    // Use absolute paths to avoid path resolution issues
    const audioPath = path.resolve(path.join(jobDir, 'audio.wav'));
    const landmarksPath = path.resolve(path.join(jobDir, 'landmarks.json'));
    const absoluteFacesDir = path.resolve(facesDir);
    
    // Get video FPS (default to 1.0 for extracted frames)
    const videoFps = 1.0; // Since we extract at 1 fps
    
    // Send faces_folder path, audio_path, and landmarks_path to Flask server
    // Use absolute paths to ensure Flask can find the files
    const requestPayload: any = {
      jobId,
      faces_folder: absoluteFacesDir,  // Use absolute path
      batch_size: batchSize,
    };
    
    // Add audio and landmarks if available - with extensive debugging
    logger.info(`Checking for audio and landmarks files...`);
    logger.info(`  Audio path: ${audioPath}`);
    logger.info(`  Landmarks path: ${landmarksPath}`);
    logger.info(`  Audio exists: ${fs.existsSync(audioPath)}`);
    logger.info(`  Landmarks exists: ${fs.existsSync(landmarksPath)}`);
    
    // Check audio file - use absolute paths
    const absoluteAudioPath = path.resolve(audioPath);
    if (fs.existsSync(absoluteAudioPath)) {
      const audioStats = fs.statSync(absoluteAudioPath);
      requestPayload.audio_path = absoluteAudioPath;  // Use absolute path
      logger.info(`[OK] Audio file found: ${absoluteAudioPath} (${(audioStats.size / 1024).toFixed(2)} KB)`);
    } else {
      logger.warn(`[ERROR] Audio file NOT found at ${absoluteAudioPath}`);
      logger.warn(`  This will disable audio sync analysis`);
      // Try alternative locations
      const altAudioPath = path.resolve(path.join(path.dirname(jobDir), jobId, 'audio.wav'));
      if (fs.existsSync(altAudioPath)) {
        requestPayload.audio_path = altAudioPath;
        logger.info(`[OK] Found audio at alternative location: ${altAudioPath}`);
      }
    }
    
    // Check landmarks file - use absolute paths
    const absoluteLandmarksPath = path.resolve(landmarksPath);
    if (fs.existsSync(absoluteLandmarksPath)) {
      const landmarksStats = fs.statSync(absoluteLandmarksPath);
      requestPayload.landmarks_path = absoluteLandmarksPath;  // Use absolute path
      requestPayload.video_fps = videoFps;
      logger.info(`[OK] Landmarks file found: ${absoluteLandmarksPath} (${(landmarksStats.size / 1024).toFixed(2)} KB)`);
    } else {
      logger.warn(`[ERROR] Landmarks file NOT found at ${absoluteLandmarksPath}`);
      logger.warn(`  This will disable audio sync analysis`);
      
      // Try alternative locations where preprocessor might have saved it
      const altPaths = [
        path.resolve(path.join(path.dirname(jobDir), jobId, 'landmarks.json')),
        path.resolve(path.join(jobDir, '..', jobId, 'landmarks.json')),
        path.resolve(path.join(path.dirname(path.dirname(jobDir)), jobId, 'landmarks.json')),
      ];
      
      for (const altPath of altPaths) {
        if (fs.existsSync(altPath)) {
          requestPayload.landmarks_path = altPath;
          requestPayload.video_fps = videoFps;
          logger.info(`[OK] Found landmarks at alternative location: ${altPath}`);
          break;
        }
      }
      
      // If still not found, list directory contents for debugging
      if (!requestPayload.landmarks_path) {
        logger.warn(`  Searching in job directory: ${jobDir}`);
        try {
          const dirContents = fs.readdirSync(jobDir);
          logger.warn(`  Directory contents: ${dirContents.join(', ')}`);
        } catch (e) {
          logger.warn(`  Could not read directory: ${e}`);
        }
      }
    }
    
    // Final check - both must be present for audio sync
    if (requestPayload.audio_path && requestPayload.landmarks_path) {
      logger.info(`[OK] Both audio and landmarks available - audio sync will be computed`);
    } else {
      logger.warn(`[ERROR] Audio sync will be skipped - missing:`);
      if (!requestPayload.audio_path) logger.warn(`  - Audio file`);
      if (!requestPayload.landmarks_path) logger.warn(`  - Landmarks file`);
    }
    
    const response = await axios.post(`${MODEL_SERVER_URL}/infer_frames`, requestPayload, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    
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
      logger.info('[OK] Audio extracted successfully');
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

    // Extract final scores (server now computes final_prob using aggregation)
    const visualScore = modelResult.visual_prob ?? modelResult.score ?? null;
    const audioSyncScore = modelResult.audio_sync_score ?? null;  // Use ?? to handle 0 values
    const finalScore = modelResult.final_prob ?? modelResult.combined_score ?? visualScore;
    const classification = modelResult.classification || 'UNKNOWN';
    const confidenceLevel = modelResult.confidence_level || 'UNKNOWN';
    const explanations = modelResult.explanations || [];
    const audioSyncQuality = modelResult.audio_sync_quality || null;
    
    // Log audio sync score for debugging
    logger.info(`Extracted from model result:`);
    logger.info(`  audio_sync_score: ${audioSyncScore !== null ? audioSyncScore : 'null'}`);
    logger.info(`  audio_sync_quality: ${audioSyncQuality || 'null'}`);
    logger.info(`  Has audio_sync_score in response: ${'audio_sync_score' in modelResult}`);
    
    // Log the aggregation results
    logger.info(`Aggregation results:`);
    logger.info(`  visual_prob = ${visualScore !== null ? visualScore.toFixed(4) : 'null'}`);
    if (audioSyncScore !== null) {
      logger.info(`  audio_sync_score = ${audioSyncScore.toFixed(4)}`);
    }
    logger.info(`  final_prob = ${finalScore !== null ? finalScore.toFixed(4) : 'null'} (alpha=0.8, beta=0.2)`);
    logger.info(`  classification = ${classification} (${confidenceLevel})`);
    if (explanations.length > 0) {
      logger.info(`  explanations: ${explanations.length} items`);
    }
    
    // Log final score with threshold interpretation
    logger.info(`\n${'='.repeat(60)}`);
    logger.info(`FINAL SCORE (Node.js Backend):`);
    logger.info(`  visual_prob = ${visualScore !== null ? visualScore.toFixed(4) : 'null'} (${visualScore !== null ? (visualScore * 100).toFixed(2) : 'null'}%)`);
    if (audioSyncScore !== null) {
      logger.info(`  audio_sync_score = ${audioSyncScore.toFixed(4)} (${(audioSyncScore * 100).toFixed(2)}%)`);
    }
    logger.info(`  final_prob = ${finalScore !== null ? finalScore.toFixed(4) : 'null'} (${finalScore !== null ? (finalScore * 100).toFixed(2) : 'null'}%)`);
    logger.info(`  Classification: ${classification} (Confidence: ${confidenceLevel})`);
    logger.info(`  Thresholds (fake_ratio): <0.3=REAL, 0.3-0.59=SUSPECTED, >=0.6=FAKE`);
    logger.info(`  Job ID: ${jobId}`);
    logger.info(`  Number of frames: ${modelResult.meta?.num_frames || 'unknown'}`);
    logger.info(`  Suspicious frames: ${suspicious.length}`);
    if (suspicious.length > 0) {
      logger.info(`  Top suspicious frame scores: ${suspicious.slice(0, 3).map((s: any) => (s.score ?? s.prob ?? s.confidence ?? 0).toFixed(4)).join(', ')}`);
    }
    if (explanations.length > 0) {
      logger.info(`  Explanations:`);
      explanations.slice(0, 3).forEach((exp: string) => {
        logger.info(`    - ${exp}`);
      });
    }
    logger.info(`${'='.repeat(60)}\n`);

    // Update analysis with result (map Flask response fields)
    await Analysis.updateOne(
      { jobId },
      {
        status: 'done',
        progress: 100,
        result: {
          visual_prob: visualScore,
          audio_sync_score: audioSyncScore,
          final_prob: finalScore,
          classification: classification,
          confidence_level: confidenceLevel,
          explanations: explanations,
          visual_scores: modelResult.visual_scores || null,
          suspiciousFrames: uploadedFrames,
          meta: {
            ...(modelResult.meta || {}),
            has_audio_sync: audioSyncScore !== null,
            aggregation_formula: 'final_prob = 0.8 * visual_prob + 0.2 * (1 - audio_sync_score)',
          },
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

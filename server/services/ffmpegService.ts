import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs-extra';

/**
 * Extracts frames from a video using ffmpeg.
 * @param videoPath Path to the input video file
 * @param outDir Directory where frames will be written
 * @param fps Frames per second to extract (default 1)
 * @returns Promise resolving to array of full paths to frame files
 */
export async function extractFrames(videoPath: string, outDir: string, fps = 1): Promise<string[]> {
  await fs.ensureDir(outDir);

  return new Promise<string[]>((resolve, reject) => {
    const outPattern = path.join(outDir, 'frame_%05d.jpg');
    const args = ['-i', videoPath, '-vf', `fps=${fps}`, outPattern];

    console.log('[ffmpeg] Extracting frames:', args.join(' '));

    const ff = spawn('ffmpeg', args, { stdio: 'inherit' });

    ff.on('error', (err) => {
      console.error('[ffmpeg] spawn error:', err);
      reject(err);
    });

    ff.on('close', async (code) => {
      if (code !== 0) {
        const err = new Error(`ffmpeg exited with code ${code}`);
        console.error('[ffmpeg] error:', err.message);
        return reject(err);
      }

      try {
        const files = (await fs.readdir(outDir))
          .filter((f) => f.match(/^frame_\d{5}\.jpg$/))
          .sort();

        const fullPaths = files.map((f) => path.join(outDir, f));
        console.log(`[ffmpeg] Extracted ${fullPaths.length} frames to ${outDir}`);
        resolve(fullPaths);
      } catch (readErr) {
        console.error('[ffmpeg] read outDir error:', readErr);
        reject(readErr);
      }
    });
  });
}

/**
 * Extracts audio from a video using ffmpeg and writes a WAV 16k mono file.
 * @param videoPath Path to input video
 * @param outAudioPath Path to output wav file
 */
export async function extractAudio(videoPath: string, outAudioPath: string): Promise<void> {
  await fs.ensureDir(path.dirname(outAudioPath));

  return new Promise<void>((resolve, reject) => {
    const args = [
      '-i', videoPath,
      '-ar', '16000', // sample rate
      '-ac', '1',     // mono
      '-f', 'wav',
      '-y',           // overwrite
      outAudioPath,
    ];

    console.log('[ffmpeg] Extracting audio:', args.join(' '));

    const ff = spawn('ffmpeg', args, { stdio: 'inherit' });

    ff.on('error', (err) => {
      console.error('[ffmpeg] spawn error (audio):', err);
      reject(err);
    });

    ff.on('close', (code) => {
      if (code !== 0) {
        const err = new Error(`ffmpeg exited with code ${code} (audio)`);
        console.error('[ffmpeg] audio error:', err.message);
        return reject(err);
      }

      console.log(`[ffmpeg] Extracted audio to ${outAudioPath}`);
      resolve();
    });
  });
}

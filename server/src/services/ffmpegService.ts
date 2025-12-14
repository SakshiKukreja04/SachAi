import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs-extra';

/** Resolve an ffmpeg binary path.
 * Priority: `process.env.FFMPEG_PATH` -> `ffmpeg-static` package -> 'ffmpeg' on PATH
 */
function resolveFfmpegBinary(): string {
  if (process.env.FFMPEG_PATH && process.env.FFMPEG_PATH.length > 0) {
    return process.env.FFMPEG_PATH;
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ff = require('ffmpeg-static');
    if (ff && typeof ff === 'string') return ff;
  } catch (e) {
    // ffmpeg-static not installed, fall through to 'ffmpeg'
  }

  return 'ffmpeg';
}

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

    const ffPath = resolveFfmpegBinary();
    console.log('[ffmpeg] using binary:', ffPath);
    const ff = spawn(ffPath, args, { stdio: 'inherit' });

    ff.on('error', (err) => {
      console.error('[ffmpeg] spawn error:', err);
      if ((err as any).code === 'ENOENT') {
        reject(new Error(`ffmpeg not found (tried '${ffPath}'). Install ffmpeg or set FFMPEG_PATH, or install the 'ffmpeg-static' npm package.`));
      } else {
        reject(err);
      }
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

    const ffPath = resolveFfmpegBinary();
    console.log('[ffmpeg] using binary for audio:', ffPath);
    const ff = spawn(ffPath, args, { stdio: 'inherit' });

    ff.on('error', (err) => {
      console.error('[ffmpeg] spawn error (audio):', err);
      if ((err as any).code === 'ENOENT') {
        reject(new Error(`ffmpeg not found (tried '${ffPath}'). Install ffmpeg or set FFMPEG_PATH, or install the 'ffmpeg-static' npm package.`));
      } else {
        reject(err);
      }
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

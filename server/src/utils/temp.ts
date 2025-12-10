import path from 'path';
import fs from 'fs-extra';

const TEMP_DIR = process.env.TEMP_DIR || './tmp';

export function getTempJobDir(jobId: string): string {
  return path.join(TEMP_DIR, jobId);
}

export function createTempJobDir(jobId: string): string {
  const dir = getTempJobDir(jobId);
  fs.ensureDirSync(dir);
  return dir;
}

export async function cleanupTempJobDir(jobId: string): Promise<void> {
  const dir = getTempJobDir(jobId);
  try {
    if (fs.existsSync(dir)) {
      fs.removeSync(dir);
    }
  } catch (error) {
    console.error(`Failed to cleanup temp directory ${dir}:`, error);
  }
}

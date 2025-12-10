import { EventEmitter } from 'events';
import { processAnalysisJob } from '../services/processor';
import { logger } from '../utils/logger';

interface Job {
  id: string;
  data: any;
}

/**
 * Simple in-memory job queue using Node.js EventEmitter.
 * For local development only.
 */
export class SimpleJobQueue extends EventEmitter {
  private queue: Job[] = [];
  private processing = false;
  private maxWorkers: number;
  private activeWorkers = 0;

  constructor(maxWorkers = 2) {
    super();
    this.maxWorkers = maxWorkers;
  }

  async add(name: string, data: any, options?: any): Promise<void> {
    const jobId = options?.jobId || data.jobId;
    this.queue.push({ id: jobId, data });
    logger.info(`[Queue] Job enqueued: ${jobId}`);
    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    if (this.processing || this.activeWorkers >= this.maxWorkers) {
      return;
    }

    this.processing = true;

    while (this.queue.length > 0 && this.activeWorkers < this.maxWorkers) {
      const job = this.queue.shift();
      if (!job) break;

      this.activeWorkers++;
      logger.info(`[Queue] Processing job ${job.id} (${this.activeWorkers}/${this.maxWorkers} workers)`);

      try {
        await processAnalysisJob(job as any);
        this.emit('completed', job);
      } catch (error) {
        logger.error(`[Queue] Job ${job.id} failed:`, error);
        this.emit('failed', job, error);
      } finally {
        this.activeWorkers--;
        // Continue processing if more jobs in queue
        if (this.queue.length > 0 && this.activeWorkers < this.maxWorkers) {
          setImmediate(() => this.processQueue());
        }
      }
    }

    this.processing = false;
  }
}

let queue: SimpleJobQueue | null = null;

export function initQueue(): SimpleJobQueue {
  queue = new SimpleJobQueue(2); // 2 concurrent workers
  logger.info('[Queue] Initialized in-memory job queue');
  return queue;
}

export function getQueue(): SimpleJobQueue | null {
  return queue;
}

export { Job };


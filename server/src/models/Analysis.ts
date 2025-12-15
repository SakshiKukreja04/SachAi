import mongoose, { Schema, Document } from 'mongoose';

export interface IAnalysis extends Document {
  jobId: string;
  status: 'queued' | 'processing' | 'done' | 'failed';
  filePath?: string;
  videoUrl?: string;
  progress: number;
  result?: {
    // Legacy fields (for backward compatibility)
    score?: number;
    label?: string;
    reason?: string;
    // New fields (actual model output)
    visual_prob?: number;  // Probability of being fake/deepfake (0-1)
    visual_scores?: Array<{ file: string; score: number }>;
    suspiciousFrames?: Array<{
      id?: string;
      file?: string;
      filename?: string;
      score?: number;
      rank?: number;
      time?: string;
      confidence?: number;
    }>;
    meta?: {
      num_frames?: number;
      batch_size?: number;
      model?: string;
      checkpoint_loaded?: boolean;
      warning?: string;
    };
  };
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}

const analysisSchema = new Schema<IAnalysis>(
  {
    jobId: { type: String, required: true, unique: true, index: true },
    status: { type: String, enum: ['queued', 'processing', 'done', 'failed'], default: 'queued' },
    filePath: String,
    videoUrl: String,
    progress: { type: Number, default: 0 },
    result: {
      // Legacy fields
      score: Number,
      label: String,
      reason: String,
      // New fields - use Schema.Types.Mixed for flexibility
      visual_prob: Number,
      audio_sync_score: Number,
      final_prob: Number,
      classification: String,
      confidence_level: String,
      explanations: Schema.Types.Mixed,
      visual_scores: Schema.Types.Mixed,
      suspiciousFrames: Schema.Types.Mixed,
      meta: Schema.Types.Mixed,
    },
    error: String,
  },
  { timestamps: true }
);
// Export a (possibly fallback) Analysis object. Declare exported symbol first so
// TypeScript always recognizes the named export regardless of branch.
export let Analysis: any;

// If SKIP_MONGODB is set, export a lightweight in-memory fallback model
if (process.env.SKIP_MONGODB === 'true') {
  type PlainAnalysis = Partial<IAnalysis> & { jobId: string };
  const store: Map<string, any> = new Map();

  const InMemoryModel = {
    async create(doc: PlainAnalysis) {
      const now = new Date();
      const item = Object.assign({ status: 'queued', progress: 0, createdAt: now, updatedAt: now }, doc);
      store.set(item.jobId, item);
      return item;
    },
    async findOne(query: any) {
      if (!query) return null;
      const jobId = query.jobId;
      if (jobId) return store.get(jobId) || null;
      // basic search by status or other simple props
      for (const v of store.values()) {
        let match = true;
        for (const k of Object.keys(query)) {
          if ((v as any)[k] !== (query as any)[k]) {
            match = false;
            break;
          }
        }
        if (match) return v;
      }
      return null;
    },
    async updateOne(query: any, update: any) {
      const jobId = query.jobId;
      if (!jobId) return { acknowledged: false };
      const prev = store.get(jobId) || { jobId };
      const now = new Date();
      const merged = Object.assign({}, prev, update, { updatedAt: now });
      store.set(jobId, merged);
      return { acknowledged: true };
    },
    async find(query: any = {}, limit = 20) {
      const out: any[] = [];
      for (const v of store.values()) {
        out.push(v);
        if (out.length >= limit) break;
      }
      return out;
    },
  };

  Analysis = InMemoryModel;
} else {
  Analysis = mongoose.model<IAnalysis>('Analysis', analysisSchema);
}

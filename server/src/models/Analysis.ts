import mongoose, { Schema, Document } from 'mongoose';

export interface IAnalysis extends Document {
  jobId: string;
  status: 'queued' | 'processing' | 'done' | 'failed';
  filePath?: string;
  videoUrl?: string;
  progress: number;
  result?: {
    score: number;
    label: string;
    reason: string;
    suspiciousFrames?: Array<{ time: string }>;
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
      score: Number,
      label: String,
      reason: String,
      suspiciousFrames: [{ time: String }],
    },
    error: String,
  },
  { timestamps: true }
);

export const Analysis = mongoose.model<IAnalysis>('Analysis', analysisSchema);

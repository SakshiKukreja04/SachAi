import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import 'dotenv/config';
import { initQueue } from './queues/jobQueue';
import analyzeRoute from './routes/analyze';
import statusRoute from './routes/status';
import historyRoute from './routes/history';
import { logger } from './utils/logger';

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// MongoDB connection (optional for local dev)
const mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27017/sachai-backend';
const skipMongoDB = process.env.SKIP_MONGODB === 'true';

if (!skipMongoDB) {
  mongoose.connect(mongoUri)
    .then(() => logger.info('MongoDB connected'))
    .catch(err => {
      logger.warn('MongoDB connection failed (proceeding without DB):', err.message);
    });
}

// Initialize in-memory job queue
const queue = initQueue();

// Routes
app.use('/api', analyzeRoute);
app.use('/api', statusRoute);
app.use('/api', historyRoute);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', mongoDBSkipped: skipMongoDB });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  logger.info(`Server running on port ${PORT}`);
});

import { MongoClient, Db, Collection } from 'mongodb';
import { logger } from '../utils/logger';

// Get and sanitize MongoDB URI (fix malformed query parameters like ?=Cluster0)
let uri = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/sachai-backend';
if (uri.includes('?=')) {
  // Remove invalid query parameters like ?=value
  uri = uri.replace(/\?=[^&]*(&|$)/g, '?').replace(/\?$/, '');
  // If query string is now empty, remove the ?
  if (uri.endsWith('?')) {
    uri = uri.slice(0, -1);
  }
}

let client: MongoClient | null = null;
let db: Db | null = null;

async function tryConnect(attempt: number, maxAttempts: number, delayMs: number) {
  try {
    client = new MongoClient(uri, { serverSelectionTimeoutMS: 5000 });
    await client.connect();
    db = client.db();
    logger.info('Mongo (native) connected');

    // ensure TTL index on images collection
    const images = db.collection('images');
    try {
      await images.createIndex({ expireAt: 1 }, { expireAfterSeconds: 0 });
    } catch (e) {
      logger.warn('Could not ensure TTL index on images collection', (e as Error).message);
    }

    return { client, db };
  } catch (err) {
    logger.warn(`Mongo connect attempt ${attempt}/${maxAttempts} failed: ${(err as Error).message}`);
    if (attempt >= maxAttempts) throw err;
    await new Promise((r) => setTimeout(r, delayMs));
    return tryConnect(attempt + 1, maxAttempts, Math.min(delayMs * 2, 10000));
  }
}

export async function connectMongo() {
  if (db) return { client: client as MongoClient, db };
  const maxAttempts = Number(process.env.MONGO_CONNECT_ATTEMPTS || 5);
  const initialDelay = Number(process.env.MONGO_CONNECT_DELAY_MS || 1000);
  return tryConnect(1, maxAttempts, initialDelay);
}

export function getImagesCollection(): Collection {
  if (!db) throw new Error('Mongo not connected');
  return db.collection('images');
}

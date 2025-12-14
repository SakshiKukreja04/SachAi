import fs from 'fs';
import path from 'path';
import { ObjectId, Binary } from 'mongodb';
import { connectMongo, getImagesCollection } from '../utils/mongoClient';
import { logger } from '../utils/logger';

export async function saveFileToDb(jobId: string, filename: string, filePath: string, ttlSeconds = 60 * 60) {
  const buf = await fs.promises.readFile(filePath);
  const contentType = mimeFromFilename(filename);
  await connectMongo();
  const images = getImagesCollection();
  const now = new Date();
  const expireAt = new Date(now.getTime() + ttlSeconds * 1000);

  const res = await images.insertOne({
    jobId,
    filename,
    data: new Binary(buf),
    contentType,
    createdAt: now,
    expireAt,
  });

  logger.info(`Saved image ${filename} -> db id=${res.insertedId}`);
  return res.insertedId.toString();
}

export async function getImageDocument(id: string) {
  await connectMongo();
  const images = getImagesCollection();
  const doc = await images.findOne({ _id: new ObjectId(id) });
  return doc as any;
}

export async function saveBufferToDb(jobId: string, filename: string, buf: Buffer, ttlSeconds = 60 * 60) {
  const contentType = mimeFromFilename(filename);
  await connectMongo();
  const images = getImagesCollection();
  const now = new Date();
  const expireAt = new Date(now.getTime() + ttlSeconds * 1000);

  const res = await images.insertOne({
    jobId,
    filename,
    data: new Binary(buf),
    contentType,
    createdAt: now,
    expireAt,
  });
  logger.info(`Saved buffer ${filename} -> db id=${res.insertedId}`);
  return res.insertedId.toString();
}

export async function getImagesForJob(jobId: string) {
  await connectMongo();
  const images = getImagesCollection();
  const docs = await images.find({ jobId }).toArray();
  return docs as any[];
}

function mimeFromFilename(fname: string) {
  const ext = path.extname(fname).toLowerCase();
  if (ext === '.png') return 'image/png';
  return 'image/jpeg';
}

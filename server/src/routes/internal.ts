import { Router } from 'express';
import multer from 'multer';
import { saveBufferToDb } from '../services/imageService';

const upload = multer();
const router = Router();

function paramToString(v: unknown): string | undefined {
  if (typeof v === 'string') return v;
  if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'string') return v[0];
  return undefined;
}

// Internal endpoint used by the Python preprocessor to POST face crops (in-memory)
router.post('/upload_face', upload.single('file'), async (req, res) => {
  try {
    const jobId = paramToString(req.body?.jobId) || paramToString(req.query?.jobId);
    const filename = paramToString(req.body?.filename) || (req.file && req.file.originalname) || `face_${Date.now()}.jpg`;
    const ttl = Number(process.env.IMAGE_TTL_SECONDS || 3600);
    if (!req.file || !jobId) return res.status(400).json({ error: 'missing file or jobId' });

    const id = await saveBufferToDb(jobId, filename, req.file.buffer, ttl);
    res.json({ id });
  } catch (err) {
    res.status(500).json({ error: 'upload failed' });
  }
});

export default router;

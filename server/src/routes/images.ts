import { Router } from 'express';
import { getImageDocument } from '../services/imageService';

const router = Router();

router.get('/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const doc = await getImageDocument(id);
    if (!doc) return res.status(404).send('Not found');

    res.setHeader('Content-Type', doc.contentType || 'application/octet-stream');
    res.setHeader('Cache-Control', 'no-store');
    const buf = doc.data.buffer ? Buffer.from(doc.data.buffer) : Buffer.from(doc.data);
    res.send(buf);
  } catch (err) {
    res.status(500).send('error');
  }
});

export default router;

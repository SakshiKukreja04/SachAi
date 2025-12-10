Python Preprocessing - Usage Snippet

Install Python dependencies:

```bash
pip install opencv-python numpy pillow tqdm
```

Example usage (extract frames, detect faces, save crops and landmarks JSON):

```bash
python ml/face_preprocess.py --video ./tmp/test.mp4 --out ./tmp --jobId test --fps 1 --workers 4
```

Validate sample run using helper script:

```bash
chmod +x scripts/validate_sample.sh
./scripts/validate_sample.sh ./tmp/test.mp4 test
```

Outputs (relative to `./tmp/test`):

- `frames/` : extracted frames (`frame_00001.jpg`, ...)
- `faces/` : cropped face images (`face_frame_00001_1.jpg`, ...)
- `landmarks.json` : JSON with per-frame face bounding boxes and centers

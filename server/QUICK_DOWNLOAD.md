# Quick Download Guide - FaceForensics Dataset

## Important: Use EU2 Server

**Only EU2 server is currently available.** Always include `--server EU2` in your commands.

## Quick Commands

### Download Original (Real) Videos

**High quality (c23) - Limited to 400 videos:**
```powershell
python faceforensics.py C:\FaceForensics -d original -c c23 -t videos --server EU2 -n 400
```

**Lower quality, faster download (c40) - Limited to 400 videos:**
```powershell
python faceforensics.py C:\FaceForensics -d original -c c40 -t videos --server EU2 -n 400
```

### Download Deepfakes (Fake) Videos

**High quality - Limited to 400 videos:**
```powershell
python faceforensics.py C:\FaceForensics -d Deepfakes -c c23 -t videos --server EU2 -n 400
```

**Lower quality - Limited to 400 videos:**
```powershell
python faceforensics.py C:\FaceForensics -d Deepfakes -c c40 -t videos --server EU2 -n 400
```

### Download All Datasets (Limited)

```powershell
# Download all datasets, limited to 400 videos each
python faceforensics.py C:\FaceForensics -d all -c c40 -t videos --server EU2 -n 400
```

## Recommended: Start with c40 (400 videos each)

For faster downloads and testing, start with `-c c40` (lower quality but smaller files) and limit to 400 videos:

```powershell
# Download original videos (real) - 400 videos
python faceforensics.py C:\FaceForensics -d original -c c40 -t videos --server EU2 -n 400

# Download Deepfakes videos (fake) - 400 videos
python faceforensics.py C:\FaceForensics -d Deepfakes -c c40 -t videos --server EU2 -n 400
```

**Note:** Use `-n 400` to limit downloads to 400 videos. Remove `-n` parameter to download all videos.

You can always download c23 (higher quality) later if needed.

## Next Steps After Download

1. Extract faces from videos (see `FACEFORENSICS_SETUP.md`)
2. Organize data for training
3. Train the model

For detailed instructions, see:
- `FACEFORENSICS_SETUP.md` - Complete setup guide
- `RUN_FACEFORENSICS.md` - Step-by-step instructions


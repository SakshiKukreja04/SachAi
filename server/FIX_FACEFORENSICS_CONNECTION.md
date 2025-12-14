# Fix FaceForensics Connection Error

## Problem
The script is trying to connect to `canis.vc.in.tum.de:8100` but getting "Connection refused" error.

## Solutions

### Solution 1: Try Different Server (Recommended)

The FaceForensics script supports multiple servers. Try using a different one:

**Use EU2 server:**
```powershell
python faceforensics.py C:\FaceForensics -d original -c c23 -t videos --server EU2
```

**Use CA (Canada) server:**
```powershell
python faceforensics.py C:\FaceForensics -d original -c c23 -t videos --server CA
```

### Solution 2: Check Server Status

The servers might be temporarily down. Try:
1. Check your internet connection
2. Try again later (servers might be under maintenance)
3. Check FaceForensics GitHub for updates: https://github.com/ondyari/FaceForensics

### Solution 3: Update the Script URL

If the servers have moved, you might need to update the script. Check the latest version from:
https://github.com/ondyari/FaceForensics

### Solution 4: Use Alternative Download Method

If the servers are down, you can:
1. Check if FaceForensics is available on other platforms (Kaggle, etc.)
2. Contact the FaceForensics team for updated download links
3. Use a VPN if your location is blocked

### Solution 5: Manual Download (If Available)

Some datasets are available through direct download links. Check:
- FaceForensics GitHub issues for alternative download methods
- Academic/research portals that host the dataset

## Quick Fix Command

**Use EU2 server (only available server):**

```powershell
python faceforensics.py C:\FaceForensics -d original -c c23 -t videos --server EU2
```

Or with lower quality (faster download):

```powershell
python faceforensics.py C:\FaceForensics -d original -c c40 -t videos --server EU2
```

**Note:** Only EU2 server is currently available. Always use `--server EU2` flag.

## Alternative: Use Pre-extracted Face Data

If you already have FaceForensics videos downloaded elsewhere, you can skip the download step and go directly to face extraction:

```powershell
cd C:\SachAi\server
python scripts/extract_faceforensics_faces.py `
  --video_dir "C:\path\to\your\videos" `
  --output_dir "C:\FaceForensics\faces\real" `
  --label 0
```

## Check Script Version

Make sure you have the latest version of the download script from:
https://github.com/ondyari/FaceForensics

The script might have been updated with new server URLs.


# Audio Sync Debugging Guide

## 🔍 What I've Added

### 1. Enhanced Logging in Node.js (`processor.ts`)
- **Audio extraction**: Logs file size and verifies file exists
- **Path checking**: Logs all paths being checked with existence status
- **Alternative paths**: Checks multiple locations for landmarks file
- **Directory listing**: Shows directory contents if landmarks not found

### 2. Enhanced Logging in Python (`server.py`)
- **Detailed sync analysis section**: Shows exactly what files are being checked
- **File existence verification**: Logs whether each file exists
- **Error details**: Full stack traces for debugging
- **Step-by-step progress**: Shows each stage of audio sync computation

### 3. Enhanced Logging in Preprocessor (`face_preprocess.py`)
- **Configuration dump**: Shows all paths and settings
- **Landmarks verification**: Verifies file was written and is valid JSON
- **Directory checks**: Lists directory contents if file not found
- **Absolute paths**: Uses absolute paths to avoid path resolution issues

## 🔧 How to Debug

### Step 1: Check Node.js Logs
Look for these messages in your Node.js backend logs:

```
Checking for audio and landmarks files...
  Audio path: /path/to/audio.wav
  Landmarks path: /path/to/landmarks.json
  Audio exists: true/false
  Landmarks exists: true/false
✓ Audio file found: ... (XX KB)
✓ Landmarks file found: ... (XX KB)
✓ Both audio and landmarks available - audio sync will be computed
```

**If you see warnings:**
- `✗ Audio file NOT found` → Audio extraction failed
- `✗ Landmarks file NOT found` → Preprocessor didn't save landmarks

### Step 2: Check Python Preprocessor Logs
Look for these messages in preprocessor output:

```
[main] ========================================
[main] Face Preprocessing Configuration
[main] ========================================
[main] landmarks_path: /absolute/path/to/landmarks.json
[main] landmarks_dir exists: true
[main] ========================================
[detect_and_upload_faces] ✓ Saved landmarks to ... (XX bytes)
[main] ✓ Landmarks file verified: ... (XX bytes)
[main] ✓ Landmarks JSON valid: X frames
```

**If you see errors:**
- `✗ WARNING: Landmarks file not found` → Check directory permissions
- `✗ Landmarks JSON invalid` → Check JSON structure

### Step 3: Check Flask Server Logs
Look for this section in Flask server output:

```
============================================================
AUDIO SYNC ANALYSIS
============================================================
Audio path: /path/to/audio.wav
Landmarks path: /path/to/landmarks.json
Audio exists: True/False
Landmarks exists: True/False
✓ Both files found, computing audio sync score...
✓ Landmarks loaded: X frames
✓ Audio sync score computed: 0.XXXX
============================================================
```

**If you see errors:**
- `✗ Files not found` → Check paths match between Node.js and Flask
- `✗ Error loading landmarks` → Check JSON format
- `✗ Error computing audio sync` → Check librosa installation

## 🐛 Common Issues & Fixes

### Issue 1: Audio File Not Found
**Symptoms:**
```
✗ Audio file NOT found at /path/to/audio.wav
```

**Possible Causes:**
1. Video has no audio track
2. FFmpeg extraction failed silently
3. Path mismatch between extraction and Flask server

**Fix:**
- Check if video has audio: `ffmpeg -i video.mp4`
- Check FFmpeg logs for errors
- Verify audio path is absolute

### Issue 2: Landmarks File Not Found
**Symptoms:**
```
✗ Landmarks file NOT found at /path/to/landmarks.json
```

**Possible Causes:**
1. Preprocessor didn't save landmarks
2. Path mismatch (relative vs absolute)
3. Directory permissions issue
4. Preprocessor crashed before saving

**Fix:**
- Check preprocessor logs for errors
- Verify `--landmarks` argument is passed correctly
- Check directory permissions
- Look for landmarks in alternative locations

### Issue 3: Landmarks File Empty/Invalid
**Symptoms:**
```
✗ Landmarks JSON invalid: ...
```

**Possible Causes:**
1. No faces detected in video
2. MediaPipe not available (using heuristic fallback)
3. JSON write interrupted

**Fix:**
- Check if faces were detected: Look for "Uploaded faces" messages
- Verify MediaPipe or heuristic fallback is working
- Check file size (should be > 0 bytes)

### Issue 4: Audio Sync Computation Fails
**Symptoms:**
```
✗ Error computing audio sync: ...
```

**Possible Causes:**
1. `librosa` not installed
2. Audio file corrupted
3. No audio data in file
4. Landmarks structure doesn't match expected format

**Fix:**
- Install librosa: `pip install librosa`
- Check audio file: `ffprobe audio.wav`
- Verify landmarks structure has `frames` array
- Check error traceback for specific issue

## 📋 Quick Checklist

Before running analysis, verify:

- [ ] FFmpeg is installed and accessible
- [ ] Video has audio track
- [ ] Python preprocessor runs without errors
- [ ] Landmarks file is created (check file size > 0)
- [ ] Audio file is created (check file size > 0)
- [ ] Both files exist when Flask server checks
- [ ] `librosa` is installed: `pip install librosa`
- [ ] Paths are absolute (not relative)

## 🚀 Testing

Run a test analysis and check logs in this order:

1. **Node.js logs** → Should show audio extraction and path checking
2. **Preprocessor logs** → Should show landmarks saving
3. **Flask server logs** → Should show audio sync computation

If any step fails, the logs will now show exactly where and why.

## 📝 Next Steps

After running an analysis:

1. **Check all three log outputs** (Node.js, Preprocessor, Flask)
2. **Look for the ✓ and ✗ symbols** - they indicate success/failure
3. **Check file sizes** - 0 bytes means file wasn't created properly
4. **Verify paths match** - absolute paths should be consistent

If audio sync still doesn't work after checking logs, share:
- Node.js logs (audio extraction section)
- Preprocessor logs (landmarks saving section)
- Flask server logs (audio sync analysis section)


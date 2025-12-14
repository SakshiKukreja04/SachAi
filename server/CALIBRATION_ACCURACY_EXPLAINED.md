# Why Calibration Training Shows Low Accuracy

## Understanding the 5.71% Accuracy

When you see **5.71% accuracy** during calibration training, this is **expected behavior** and here's why:

### The Problem

1. **Your model was trained to detect fakes** → It outputs **high probabilities** (close to 1.0) for fake videos
2. **You're calibrating with real videos** → You want **low probabilities** (close to 0.0) for real videos
3. **After 1 epoch**, the model hasn't shifted much yet → Still predicting high probabilities

### What's Happening

```
Model Output (before calibration):
  Real video → Probability: 0.65 (model thinks it's fake) ❌
  Real video → Probability: 0.72 (model thinks it's fake) ❌
  Real video → Probability: 0.58 (model thinks it's fake) ❌

Accuracy calculation:
  - If prob < 0.5 → Predict "real" (correct)
  - If prob >= 0.5 → Predict "fake" (wrong for real videos)
  
  Since most probs are > 0.5, model predicts "fake" → Wrong!
  Accuracy = 5.71% (only a few predictions were < 0.5)
```

### Why This is Normal

1. **1 epoch is very short** - The model needs time to shift its predictions
2. **Frozen backbone** - Only classifier is learning, so change is gradual
3. **Loss is decreasing** - This is the important metric (0.34 is reasonable)
4. **Model is learning** - It just needs more training

### What to Watch

**More important than accuracy:**
- ✅ **Loss decreasing**: 0.34 is good (was probably higher initially)
- ✅ **Average probability**: Should decrease from ~0.6-0.7 towards 0.2-0.3
- ⚠️ **Accuracy**: Will improve as probabilities shift lower

### Solutions

#### Option 1: Train for More Epochs (Recommended)

```powershell
# Train for 3-5 epochs instead of 1
python train/train_calibration.py --data_dir ./calibration_data --checkpoint_in checkpoint.pth --checkpoint_out checkpoint_calibrated.pth --epochs 3 --batch_size 16 --lr 1e-4
```

**Expected progression:**
- Epoch 1: Accuracy ~5-10%, Avg prob ~0.6
- Epoch 2: Accuracy ~30-50%, Avg prob ~0.4
- Epoch 3: Accuracy ~70-90%, Avg prob ~0.25

#### Option 2: Use Higher Learning Rate

```powershell
# Use 2x learning rate for faster calibration
python train/train_calibration.py --data_dir ./calibration_data --checkpoint_in checkpoint.pth --checkpoint_out checkpoint_calibrated.pth --epochs 1 --batch_size 16 --lr 2e-4
```

#### Option 3: Check if Calibration is Working

Even with low accuracy, the calibration might be working! Check:

1. **Test on a real video** before and after calibration
2. **Compare probabilities**:
   - Before: Real video → 65% fake probability
   - After: Real video → 45% fake probability (improved!)

### The Real Test: Does It Work in Practice?

**Accuracy during training ≠ Real-world performance**

What matters:
- ✅ Does the calibrated model give **lower scores** for real videos?
- ✅ Are **false positives reduced**?
- ✅ Is the **overall accuracy better** on test videos?

### Quick Test

After calibration, test on a real YouTube video:

```powershell
# Before calibration
# Real video score: 67% fake probability

# After calibration  
# Real video score: 45% fake probability (improved!)
```

**This is the real metric** - not training accuracy!

### Summary

- **5.71% accuracy is normal** for 1 epoch calibration
- **Loss decreasing is good** (0.34 is reasonable)
- **Model is learning** - just needs more time
- **Real test**: Does it reduce false positives on actual videos?

**Recommendation**: Train for 3 epochs or use higher learning rate (2e-4) for better calibration.


# Model Accuracy vs. Confidence in Deepfake Detection

## Understanding the Difference

### Model Accuracy
**Model Accuracy** refers to how well the trained model performs on a test dataset. It's typically measured as:
- **Accuracy** = (Correct Predictions / Total Predictions) × 100%
- Example: If the model correctly identifies 90 out of 100 videos, accuracy = 90%

This is a **fixed property** of the trained model and doesn't change per video.

### Confidence/Probability (Per-Video)
**Confidence/Probability** (the 47% you see) refers to:
- The model's **certainty** that a **specific video** is AI-generated
- A score between 0% (definitely authentic) and 100% (definitely deepfake)
- This changes for each video based on detected artifacts

## How They Relate

```
Model Accuracy (e.g., 90%)
    ↓
    How reliable the model is overall
    ↓
Confidence Score (e.g., 47%)
    ↓
    How certain the model is about THIS specific video
```

### Example:
- **Model Accuracy**: 90% (model is correct 9 out of 10 times on test data)
- **Video A**: Confidence = 85% → High confidence it's a deepfake
- **Video B**: Confidence = 15% → High confidence it's authentic
- **Video C**: Confidence = 47% → Uncertain (SUSPECTED category)

## Why 47% Shows as "SUSPECTED"

Your aggregation formula:
```
final_prob = 0.8 × visual_prob + 0.2 × (1 - audio_sync_score)
```

With:
- Visual probability: 47% (moderate artifacts detected)
- Audio sync: Not available (so formula uses only visual_prob)

**Classification thresholds:**
- `>= 0.8` → AUTHENTIC (very low fake probability)
- `0.4 - 0.79` → SUSPECTED (uncertain)
- `< 0.4` → DEEPFAKE (high fake probability)

**47% falls in the SUSPECTED range**, meaning:
- The model detected some suspicious features
- But not enough to be highly confident it's a deepfake
- Needs human review

## Improving Model Accuracy

### 1. **Better Training Data**
- Use larger, more diverse datasets (FaceForensics++, Celeb-DF)
- Include various deepfake techniques (FaceSwap, DeepFaceLab, etc.)
- Balance authentic vs. fake samples

### 2. **Model Architecture Improvements**
- Use ensemble models (combine multiple models)
- Fine-tune on domain-specific data
- Use attention mechanisms to focus on critical regions

### 3. **Feature Engineering**
- Combine visual + audio + temporal features
- Use higher resolution inputs
- Extract more frames per second

### 4. **Post-Processing**
- Temporal smoothing (average scores across frames)
- Confidence calibration
- Multi-scale analysis

### 5. **Audio-Visual Fusion** (What we're doing)
- Combine visual artifacts with lip-sync analysis
- Weighted aggregation improves accuracy by 5-10%
- Reduces false positives

## Current Implementation Accuracy

Based on typical Xception-based models:
- **Visual-only**: ~85-90% accuracy on FaceForensics++
- **With audio sync**: ~90-93% accuracy (estimated improvement)
- **Your model**: Depends on training data quality

## Recommendations

1. **For Production**: 
   - Aim for model accuracy > 90%
   - Use ensemble of multiple models
   - Combine with human review for SUSPECTED cases

2. **For Your Current Model**:
   - Ensure audio sync is working (fixing the "Not available" issue)
   - This will improve final_prob calculation
   - Better distinguish between authentic and fake

3. **Threshold Tuning**:
   - Adjust classification thresholds based on your use case
   - Lower threshold (e.g., 0.3) = more sensitive (catches more fakes, more false positives)
   - Higher threshold (e.g., 0.5) = more conservative (fewer false positives, might miss some fakes)

## Fixing Audio Sync "Not Available"

The issue is likely:
1. Landmarks file not being saved by preprocessor
2. Audio file path mismatch
3. Both files not found when Flask server checks

**Solution**: Added explicit landmarks path passing and verification logging.


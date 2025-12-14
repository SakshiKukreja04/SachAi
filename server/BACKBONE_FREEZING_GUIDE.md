# Backbone Freezing Guide

## Current Status

**YES, the backbone is frozen by default** in the training script.

## How It Works

### Default Behavior (Backbone Frozen)
```python
# Lines 204-217 in train_faceforensics.py
if args.train_all_layers:
    print("Training all layers...")
else:
    # Default: Freeze backbone, train only classifier head
    print("Training classifier head only")
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier/FC layers
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ["fc", "classifier", "linear", "head"]):
            param.requires_grad = True
            print(f"  Unfrozen: {name}")
```

### What Gets Trained
- **Frozen**: All Xception backbone layers (feature extractor)
- **Trainable**: Only the final classifier/FC layer (single logit output)

## Training Options

### Option 1: Frozen Backbone (Default - Faster, Less Memory)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/data \
  --checkpoint_out checkpoint.pth \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4
```

**Pros:**
- Faster training (fewer parameters to update)
- Less GPU memory required
- Good for small datasets
- Prevents overfitting on limited data

**Cons:**
- Lower accuracy potential
- Can't adapt backbone features to deepfake detection
- May not capture deepfake-specific patterns

### Option 2: Train All Layers (Better Accuracy)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/data \
  --checkpoint_out checkpoint.pth \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-5 \
  --train_all_layers
```

**Pros:**
- Higher accuracy potential (typically 5-10% improvement)
- Can adapt backbone features to deepfake detection
- Better for larger datasets
- Captures deepfake-specific patterns in early layers

**Cons:**
- Slower training
- More GPU memory required
- Risk of overfitting on small datasets
- Need lower learning rate (1e-5 vs 1e-4)

## Recommended Approach: Progressive Unfreezing

For best results, use a two-stage training approach:

### Stage 1: Train Classifier Only (Fast Initial Training)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/data \
  --checkpoint_out checkpoint_stage1.pth \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-4
```

### Stage 2: Fine-tune All Layers (Better Accuracy)
```bash
python train/train_faceforensics.py \
  --data_dir /path/to/data \
  --checkpoint_in checkpoint_stage1.pth \
  --checkpoint_out checkpoint_final.pth \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-5 \
  --train_all_layers
```

## Impact on Accuracy

### Frozen Backbone
- **Typical Accuracy**: 85-88% on FaceForensics++
- **Training Time**: ~2-4 hours (depending on dataset size)
- **Memory**: ~4-6 GB GPU

### Unfrozen Backbone
- **Typical Accuracy**: 90-93% on FaceForensics++
- **Training Time**: ~4-8 hours
- **Memory**: ~8-12 GB GPU

## How to Check Current Status

To see which layers are trainable, the training script prints:
```
Training classifier head only
  Unfrozen: fc.weight
  Unfrozen: fc.bias
Trainable parameters: 2,049
```

Or if training all layers:
```
Training all layers...
Trainable parameters: 22,910,480
```

## Recommendations

1. **For Quick Experiments**: Use frozen backbone (default)
2. **For Production**: Use progressive unfreezing (Stage 1 + Stage 2)
3. **For Best Accuracy**: Train all layers with lower learning rate
4. **For Limited GPU**: Stick with frozen backbone, use larger batch size

## Modifying the Default

If you want to change the default behavior, edit `train/train_faceforensics.py`:

```python
# Change line 205 from:
if args.train_all_layers:

# To:
if not args.freeze_backbone:  # Invert the logic
```

Then add:
```python
parser.add_argument("--freeze_backbone", action="store_true", 
                   help="Freeze backbone (default: train all layers)")
```


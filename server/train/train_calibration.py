"""
Quick calibration training script for real YouTube videos.

This script:
- Loads existing checkpoint
- Trains only classifier head (frozen backbone)
- Uses 1 epoch for quick calibration
- Prevents overfitting with small dataset
"""
from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path
from typing import Tuple

# Add parent directory to path
script_dir = Path(__file__).parent
server_dir = script_dir.parent
sys.path.insert(0, str(server_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model.xception_model import load_model


class RealVideoDataset(Dataset):
    """Dataset for real YouTube videos (label=0 for all)."""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all images in real/ directory
        real_dir = self.data_dir / "real"
        if not real_dir.exists():
            raise ValueError(f"Real directory not found: {real_dir}")
        
        self.image_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {real_dir}")
        
        print(f"Found {len(self.image_paths)} real images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Label = 0 for real videos
        label = torch.tensor(0.0, dtype=torch.float32)
        
        return image, label


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate accuracy given predictions and labels."""
    probs = torch.sigmoid(outputs.view(-1))
    preds = (probs < threshold).float()  # < 0.5 = real (label 0)
    correct = (preds == labels).float()
    return correct.mean().item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track probabilities for debugging
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().tolist())
        
        total_loss += loss.item() * inputs.size(0)
        batch_acc = calculate_accuracy(outputs, labels)
        total_correct += batch_acc * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Show average probability (should decrease towards 0 for real videos)
        avg_prob = probs.mean().item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_acc:.4f}',
            'avg_prob': f'{avg_prob:.3f}'  # Should be low (< 0.3) for real videos
        })
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    # Print statistics
    if all_probs:
        avg_prob_all = sum(all_probs) / len(all_probs)
        min_prob = min(all_probs)
        max_prob = max(all_probs)
        print(f"\n  Probability statistics:")
        print(f"    Average: {avg_prob_all:.4f} (should be < 0.3 for real videos)")
        print(f"    Range: {min_prob:.4f} - {max_prob:.4f}")
        print(f"    Predictions < 0.3: {sum(1 for p in all_probs if p < 0.3) / len(all_probs) * 100:.1f}%")
    
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(
        description="Quick calibration training with real YouTube videos"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to calibration data directory (should contain 'real/' folder)"
    )
    parser.add_argument(
        "--checkpoint_in",
        type=str,
        default=None,
        help="Input checkpoint path (existing trained model). If not provided, starts from pretrained Xception."
    )
    parser.add_argument(
        "--checkpoint_out",
        type=str,
        default="checkpoint_calibrated.pth",
        help="Output checkpoint path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1 for quick calibration)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("Calibration Training")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Input checkpoint: {args.checkpoint_in}")
    print(f"Output checkpoint: {args.checkpoint_out}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model from checkpoint or start from pretrained
    if args.checkpoint_in:
        print(f"Loading model from checkpoint: {args.checkpoint_in}")
        if not os.path.exists(args.checkpoint_in):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_in}")
        model = load_model(checkpoint_path=args.checkpoint_in, device=str(device))
    else:
        print("No checkpoint provided - starting from pretrained Xception")
        print("WARNING: This will train from scratch. For calibration, use an existing checkpoint.")
        model = load_model(checkpoint_path=None, device=str(device))
    
    model.train()  # Set to training mode
    
    # Freeze backbone, train only classifier
    print("Freezing backbone, training only classifier head...")
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier/FC layers
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ["fc", "classifier", "linear", "head"]):
            param.requires_grad = True
            print(f"  Unfrozen: {name}")
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}\n")
    
    # Prepare dataset
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = RealVideoDataset(args.data_dir, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError("No images found in dataset")
    
    print(f"Dataset size: {len(dataset)} images\n")
    
    # Data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if device.type == "cuda" else 1,
        pin_memory=device.type == "cuda"
    )
    
    # Optimizer and loss
    # Use higher learning rate for calibration to push predictions lower faster
    calibration_lr = args.lr * 2.0  # 2x learning rate for faster calibration
    optimizer = torch.optim.Adam(trainable_params, lr=calibration_lr)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Using learning rate: {calibration_lr:.6f} (2x base rate for calibration)\n")
    
    # Training loop
    print(f"Starting training ({args.epochs} epoch(s))...\n")
    
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model, dataloader, criterion, optimizer, device
        )
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Train Acc:  {train_acc:.4f} ({train_acc*100:.2f}%)")
        
        # Check what the model is actually predicting (diagnostics)
        model.eval()
        sample_probs = []
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= 5:  # Just check first 5 batches
                    break
                inputs = inputs.to(device)
                outputs = model(inputs).view(-1)
                probs = torch.sigmoid(outputs)
                sample_probs.extend(probs.cpu().tolist())
        
        if sample_probs:
            avg_prob = sum(sample_probs) / len(sample_probs)
            low_prob_count = sum(1 for p in sample_probs if p < 0.3)
            print(f"  Average predicted probability: {avg_prob:.4f}")
            print(f"    Predictions < 0.3 (real): {low_prob_count}/{len(sample_probs)} ({low_prob_count/len(sample_probs)*100:.1f}%)")
            print(f"    Note: For real videos, average prob should be < 0.3")
            if avg_prob > 0.5:
                print(f"    WARNING: Model still predicting high probabilities (thinking videos are fake)")
                print(f"    This is normal for 1 epoch - model needs more training to shift predictions")
        
        model.train()
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "calibration": True,
            }
            torch.save(checkpoint, args.checkpoint_out)
            print(f"  [OK] Saved checkpoint to {args.checkpoint_out}")
    
    print(f"\n{'='*60}")
    print("[OK] Calibration training complete!")
    print(f"  Final checkpoint: {args.checkpoint_out}")
    print(f"  Final loss: {train_loss:.6f}")
    print(f"  Final accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    print("Next step: Use the calibrated checkpoint in your server:")
    print(f"  export MODEL_CHECKPOINT={args.checkpoint_out}")
    print("  python server.py")


if __name__ == "__main__":
    main()


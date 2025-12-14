"""Training script for FaceForensics dataset with Xception model.

This script:
- Loads FaceForensics dataset (supports multiple formats)
- Splits into train/validation sets
- Trains Xception model with proper validation
- Saves best checkpoint based on validation loss
- Includes accuracy and other metrics
"""
from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path
from typing import Tuple

# Add parent directory to path so we can import model
script_dir = Path(__file__).parent
server_dir = script_dir.parent
sys.path.insert(0, str(server_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.xception_model import load_model
from model.faceforensics_dataset import FaceForensicsDataset


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate accuracy given predictions and labels."""
    probs = torch.sigmoid(outputs.view(-1))
    preds = (probs >= threshold).float()
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
    total_acc = 0.0
    num_batches = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += calculate_accuracy(outputs, labels)
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, labels)
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train Xception on FaceForensics dataset")
    parser.add_argument("--data_dir", required=True, help="Path to FaceForensics dataset root")
    parser.add_argument("--labels_csv", type=str, default=None, help="Optional CSV with filename,label")
    parser.add_argument("--checkpoint_out", type=str, default="checkpoint.pth", help="Output checkpoint path")
    parser.add_argument("--checkpoint_in", type=str, default=None, help="Input checkpoint path to resume training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if not specified)")
    parser.add_argument("--train_all_layers", action="store_true", help="Train all layers (default: freeze backbone, train only classifier)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = FaceForensicsDataset(
        root=args.data_dir,
        label_csv=args.labels_csv,
    )
    
    if len(dataset) == 0:
        raise SystemExit(f"No images found in {args.data_dir}")
    
    print(f"Total samples: {len(dataset)}")
    
    # Split dataset
    if args.val_split > 0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = None
        print("No validation split (val_split=0)")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == "cuda" else False,
        )
    
    # Load model
    print("Loading model...")
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    if args.checkpoint_in and Path(args.checkpoint_in).exists():
        print(f"Resuming from checkpoint: {args.checkpoint_in}")
        checkpoint = torch.load(args.checkpoint_in, map_location=device)
        model = load_model(checkpoint_path=args.checkpoint_in, device=str(device))
        model.train()  # Set to training mode
        
        # Load training state if available
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]
        if "best_val_acc" in checkpoint:
            best_val_acc = checkpoint["best_val_acc"]
        if "val_loss" in checkpoint:
            best_val_loss = min(best_val_loss, checkpoint["val_loss"])
        if "val_acc" in checkpoint:
            best_val_acc = max(best_val_acc, checkpoint["val_acc"])
    else:
        model = load_model(checkpoint_path=None, device=str(device))
        model.train()  # Set to training mode
        if args.checkpoint_in:
            print(f"Warning: Checkpoint not found: {args.checkpoint_in}, starting from scratch")
    
    # Freeze/unfreeze layers
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
    
    # Optimizer and loss - reinitialize AFTER freezing to only include trainable params
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    # Load optimizer state if resuming
    if args.checkpoint_in and Path(args.checkpoint_in).exists():
        checkpoint = torch.load(args.checkpoint_in, map_location=device)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state from checkpoint")
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load optimizer state (parameter mismatch): {e}")
                print("Continuing with fresh optimizer state")
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    checkpoint_dir = Path(args.checkpoint_out).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoint will be saved to: {args.checkpoint_out}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                torch.save({
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                }, args.checkpoint_out)
                print(f"✓ Saved best checkpoint (val_loss={val_loss:.6f}, val_acc={val_acc:.4f})")
        else:
            # Save every epoch if no validation
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }, args.checkpoint_out)
            print(f"✓ Saved checkpoint (train_loss={train_loss:.6f}, train_acc={train_acc:.4f})")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final checkpoint saved to: {args.checkpoint_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


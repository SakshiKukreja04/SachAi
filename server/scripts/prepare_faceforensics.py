"""Script to prepare FaceForensics dataset for training.

This script helps organize FaceForensics data into the expected format.
It can:
1. Create train/val splits
2. Generate CSV labels
3. Organize data into real/ and fake/ folders
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def organize_by_manipulation(source_dir: Path, target_dir: Path, val_split: float = 0.2):
    """Organize FaceForensics data by manipulation method.
    
    Supports two source structures:
    1. Flat structure:
       source/
         real/
         fake/
    2. Nested structure:
       source/
         manipulation1/
           real/
           fake/
         manipulation2/
           real/
           fake/
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    train_real = target_dir / "train" / "real"
    train_fake = target_dir / "train" / "fake"
    val_real = target_dir / "val" / "real"
    val_fake = target_dir / "val" / "fake"
    
    for folder in [train_real, train_fake, val_real, val_fake]:
        folder.mkdir(parents=True, exist_ok=True)
    
    real_files: List[Path] = []
    fake_files: List[Path] = []
    
    # Check for flat structure first (real/ and fake/ directly in source)
    source_real = source_dir / "real"
    source_fake = source_dir / "fake"
    
    if source_real.exists() and source_fake.exists():
        # Flat structure: collect from real/ and fake/ directly
        real_files.extend(source_real.glob("*.jpg"))
        real_files.extend(source_real.glob("*.jpeg"))
        real_files.extend(source_real.glob("*.png"))
        fake_files.extend(source_fake.glob("*.jpg"))
        fake_files.extend(source_fake.glob("*.jpeg"))
        fake_files.extend(source_fake.glob("*.png"))
    else:
        # Nested structure: look for manipulation subdirectories
        for item in source_dir.iterdir():
            if item.is_dir():
                real_dir = item / "real"
                fake_dir = item / "fake"
                
                if real_dir.exists():
                    real_files.extend(real_dir.glob("*.jpg"))
                    real_files.extend(real_dir.glob("*.jpeg"))
                    real_files.extend(real_dir.glob("*.png"))
                
                if fake_dir.exists():
                    fake_files.extend(fake_dir.glob("*.jpg"))
                    fake_files.extend(fake_dir.glob("*.jpeg"))
                    fake_files.extend(fake_dir.glob("*.png"))
    
    # Shuffle and split
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    val_real_count = int(len(real_files) * val_split)
    val_fake_count = int(len(fake_files) * val_split)
    
    # Copy files
    print(f"Organizing {len(real_files)} real and {len(fake_files)} fake images...")
    
    for i, f in enumerate(real_files):
        dest = val_real if i < val_real_count else train_real
        shutil.copy2(f, dest / f.name)
    
    for i, f in enumerate(fake_files):
        dest = val_fake if i < val_fake_count else train_fake
        shutil.copy2(f, dest / f.name)
    
    print(f"Train: {len(real_files) - val_real_count} real, {len(fake_files) - val_fake_count} fake")
    print(f"Val: {val_real_count} real, {val_fake_count} fake")


def create_csv_labels(data_dir: Path, output_csv: Path):
    """Create CSV file with labels from directory structure."""
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    
    samples: List[Tuple[str, int]] = []
    
    # Check for real/ and fake/ folders
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    
    if real_dir.exists():
        for img in real_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((img.name, 0))
    
    if fake_dir.exists():
        for img in fake_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((img.name, 1))
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for filename, label in samples:
            writer.writerow([filename, label])
    
    print(f"Created {output_csv} with {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Prepare FaceForensics dataset")
    parser.add_argument("--source_dir", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--target_dir", type=str, help="Target directory (for organize mode)")
    parser.add_argument("--output_csv", type=str, help="Output CSV file (for csv mode)")
    parser.add_argument("--mode", type=str, choices=["organize", "csv"], default="csv",
                       help="Mode: organize (create train/val structure) or csv (generate labels CSV)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    source_dir = Path(args.source_dir)
    
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")
    
    if args.mode == "organize":
        if not args.target_dir:
            raise SystemExit("--target_dir required for organize mode")
        organize_by_manipulation(source_dir, Path(args.target_dir), args.val_split)
    elif args.mode == "csv":
        if not args.output_csv:
            raise SystemExit("--output_csv required for csv mode")
        create_csv_labels(source_dir, Path(args.output_csv))


if __name__ == "__main__":
    main()


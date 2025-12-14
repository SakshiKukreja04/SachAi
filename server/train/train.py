"""Skeleton training script for fine-tuning Xception on face crops.

This is a minimal example demonstrating:
- Command-line args
- Dataset loading
- Freezing early layers and training the classifier
- Saving a best checkpoint

Labels: optional CSV mapping filenames to label (0/1). If not provided, the training loop will
raise and offer a placeholder for how to load labels.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path so we can import model
script_dir = Path(__file__).parent
server_dir = script_dir.parent
sys.path.insert(0, str(server_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.xception_model import load_model
from model.dataset import FaceFolderDataset
from model.faceforensics_dataset import FaceForensicsDataset


def load_labels_from_csv(csv_path: str) -> Dict[str, int]:
    mapping = {}
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = row.get("filename") or row.get("file") or row.get("name")
            label = int(row.get("label", 0))
            if fname:
                mapping[fname] = label
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to folder with face images")
    parser.add_argument("--labels_csv", required=False, help="Optional CSV with filename,label")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_out", default="checkpoint.pth")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)

    labels = None
    if args.labels_csv:
        labels = load_labels_from_csv(args.labels_csv)

    dataset = FaceFolderDataset(args.data_dir)
    if len(dataset) == 0:
        raise SystemExit("No images found in data_dir")

    # Wrap dataset to yield (tensor, label) if labels present
    if labels:
        # Simple label wrapper
        class LabeledDataset(torch.utils.data.Dataset):
            def __init__(self, base, mapping):
                self.base = base
                self.mapping = mapping

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                tensor, fname = self.base[idx]
                label = float(self.mapping.get(fname, 0))
                return tensor, torch.tensor(label, dtype=torch.float32)

        train_dataset = LabeledDataset(dataset, labels)
    else:
        raise SystemExit("labels_csv required for training in this skeleton script")

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = load_model(checkpoint_path=None, device=str(device))

    # Freeze all layers except final classifier
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier parameters
    for name, param in model.named_parameters():
        if any(k in name for k in ["fc", "classifier", "linear", "head"]):
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            inputs, labels_batch = batch
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * inputs.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{args.epochs} - avg_loss={avg_loss:.6f}")

        # Save checkpoint if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.checkpoint_out)
            print(f"Saved improved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()

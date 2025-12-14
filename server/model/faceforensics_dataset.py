"""FaceForensics dataset loader for training.

Supports multiple FaceForensics dataset formats:
1. Folder structure: real/ and fake/ subdirectories
2. Folder structure: manipulation_method/real and manipulation_method/fake
3. CSV with labels
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FaceForensicsDataset(Dataset):
    """Dataset loader for FaceForensics format.
    
    Supports multiple directory structures:
    1. Flat structure with real/ and fake/ folders:
       data/
         real/
           image1.jpg
           image2.jpg
         fake/
           image3.jpg
           image4.jpg
    
    2. Manipulation-based structure:
       data/
         Deepfakes/
           real/
           fake/
         FaceSwap/
           real/
           fake/
    
    3. Single folder with CSV labels
    """
    
    def __init__(
        self,
        root: str,
        label_csv: Optional[str] = None,
        transform=None,
        real_label: int = 0,
        fake_label: int = 1,
    ):
        self.root = Path(root).resolve()
        self.real_label = real_label
        self.fake_label = fake_label
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
        
        # Load labels from CSV if provided
        if label_csv and os.path.exists(label_csv):
            self._load_from_csv(label_csv)
        else:
            self._discover_from_structure()
    
    def _load_from_csv(self, csv_path: str):
        """Load labels from CSV file (filename,label format)."""
        import csv
        self.samples: List[Tuple[str, int]] = []
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get('filename') or row.get('file') or row.get('name')
                label = int(row.get('label', 0))
                if fname:
                    full_path = self.root / fname
                    if full_path.exists():
                        self.samples.append((str(full_path), label))
    
    def _discover_from_structure(self):
        """Discover dataset structure automatically."""
        self.samples: List[Tuple[str, int]] = []
        
        # Check for real/ and fake/ folders at root
        real_dir = self.root / "real"
        fake_dir = self.root / "fake"
        
        if real_dir.exists() and fake_dir.exists():
            # Structure 1: real/ and fake/ folders
            self._add_folder(real_dir, self.real_label)
            self._add_folder(fake_dir, self.fake_label)
            return
        
        # Check for manipulation-based structure
        for item in self.root.iterdir():
            if item.is_dir():
                real_subdir = item / "real"
                fake_subdir = item / "fake"
                
                if real_subdir.exists():
                    self._add_folder(real_subdir, self.real_label)
                if fake_subdir.exists():
                    self._add_folder(fake_subdir, self.fake_label)
        
        # If still empty, check if root contains images directly (assume all fake for training)
        if len(self.samples) == 0:
            self._add_folder(self.root, self.fake_label)
    
    def _add_folder(self, folder: Path, label: int):
        """Add all images from a folder with given label."""
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in IMG_EXTS:
                self.samples.append((str(img_file), label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch
        img_path, label = self.samples[idx]
        
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        
        return tensor, torch.tensor(label, dtype=torch.float32)


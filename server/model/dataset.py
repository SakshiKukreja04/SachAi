"""Dataset for reading face crops from a folder for inference.

Yields tuples: (image_tensor, filename)
Sorted lexical order of filenames is preserved for deterministic results.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FaceFolderDataset(Dataset):
    """Simple dataset that reads all images from a folder, sorted lexically.

    Returns:
        (tensor, filename) where filename is the basename (not full path)
    """

    def __init__(self, root: str, transform=None):
        self.root = os.path.abspath(root)
        self.transform = transform or transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._files = self._discover_files()

    def _discover_files(self) -> List[str]:
        files = [f for f in os.listdir(self.root) if os.path.splitext(f)[1].lower() in IMG_EXTS]
        files.sort()  # lexical sort for deterministic ordering
        return files

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int):
        fname = self._files[idx]
        path = os.path.join(self.root, fname)
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, fname

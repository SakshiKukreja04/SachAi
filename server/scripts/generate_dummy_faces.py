"""Generate dummy face-crop images for testing the inference server.

Creates numbered JPEGs (frame_00001.jpg, ...) in the provided folder.
Usage:
    python scripts/generate_dummy_faces.py --out ./examples/faces --count 8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_dummy_image(path: Path, idx: int, size=(299, 299)):
    img = Image.new("RGB", size, (int((idx*37) % 255), int((idx*73) % 255), int((idx*151) % 255)))
    draw = ImageDraw.Draw(img)
    # Draw index text in the center
    text = f"{idx:05d}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Obtain text size in a way compatible with multiple Pillow versions
    try:
        # Pillow >=8.0: textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            # Fallback to font.getsize
            w, h = font.getsize(text) if font is not None else (0, 0)
        except Exception:
            w, h = (0, 0)
    draw.text(((size[0]-w)/2, (size[1]-h)/2), text, fill=(255,255,255), font=font)
    img.save(path, format="JPEG", quality=85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./examples/faces", help="Output directory for generated images")
    parser.add_argument("--count", type=int, default=8, help="Number of images to generate")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, args.count + 1):
        fname = f"frame_{i:05d}.jpg"
        make_dummy_image(out_dir / fname, i)
    print(f"Wrote {args.count} images to {out_dir}")


if __name__ == "__main__":
    main()

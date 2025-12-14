"""Example client to call the Flask `/infer_frames` endpoint.

Modes:
- JSON: send `jobId` and `faces_folder` (server must be able to access the folder path)
- ZIP: create a ZIP from a local faces folder and send multipart with key 'zip'

Usage examples:
    python example_client.py --json --jobId test1 --faces_folder ./examples/faces --url http://127.0.0.1:8000/infer_frames
    python example_client.py --zip --jobId test2 --faces_folder ./examples/faces --url http://127.0.0.1:8000/infer_frames
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests


def zip_folder_bytes(folder: str) -> bytes:
    folder = Path(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                arcname = p.relative_to(folder)
                zf.write(p, arcname)
    buf.seek(0)
    return buf.read()


def send_json(url: str, job_id: str, faces_folder: str, batch_size: int = 32):
    payload = {"jobId": job_id, "faces_folder": os.path.abspath(faces_folder), "batch_size": batch_size}
    r = requests.post(url, json=payload, timeout=120)
    return r


def send_zip(url: str, job_id: str, faces_folder: str, batch_size: int = 32):
    zip_bytes = zip_folder_bytes(faces_folder)
    files = {"zip": (f"{job_id}.zip", zip_bytes, "application/zip")}
    data = {"jobId": job_id, "batch_size": str(batch_size)}
    r = requests.post(url, files=files, data=data, timeout=120)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/infer_frames")
    parser.add_argument("--jobId", default="example-job")
    parser.add_argument("--faces_folder", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", action="store_true")
    group.add_argument("--zip", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.json:
        print(f"Sending JSON request to {args.url}...")
        r = send_json(args.url, args.jobId, args.faces_folder, args.batch_size)
    else:
        print(f"Zipping and sending {args.faces_folder} as ZIP to {args.url}...")
        r = send_zip(args.url, args.jobId, args.faces_folder, args.batch_size)

    try:
        r.raise_for_status()
        data = r.json()
        print(json.dumps(data, indent=2))
    except Exception as exc:
        print("Request failed:", exc)
        try:
            print(r.text)
        except Exception:
            pass


if __name__ == "__main__":
    main()

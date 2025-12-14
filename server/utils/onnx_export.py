"""Utility to export the Xception PyTorch model to ONNX and run a short onnxruntime inference.

Example usage:
    python utils/onnx_export.py --checkpoint checkpoint.pth --out model.onnx
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

from model.xception_model import load_model


def export_to_onnx(checkpoint: str | None, out_path: str):
    device = "cpu"
    model = load_model(checkpoint_path=checkpoint, device=device)
    model.eval()

    dummy = torch.randn(1, 3, 299, 299, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Exported ONNX model to {out_path}")

    # Quick check with onnx
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")

    # Run a sample inference using onnxruntime
    sess = ort.InferenceSession(out_path)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: dummy.numpy()})
    probs = 1.0 / (1.0 + np.exp(-out[0].reshape(-1)))
    print("Sample ONNX inference output shape:", out[0].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Optional PyTorch checkpoint to load", default=None)
    parser.add_argument("--out", help="Output ONNX path", default="xception.onnx")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.out)

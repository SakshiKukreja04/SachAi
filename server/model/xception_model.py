"""Xception model loader and batch prediction utilities.

Uses `timm` if available to instantiate an Xception backbone and replaces the final
classifier with a single-logit output suitable for binary classification.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import timm


def _replace_classifier_with_single_logit(model: nn.Module) -> None:
    """Replace the model classifier/head with a single-output linear layer.

    This handles common attribute names used by timm models: `fc`, `classifier`.
    """
    # Try several common attribute names
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return
    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
        return
    # Fallback: try reset_classifier if available
    if hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes=1)
            return
        except Exception:
            pass
    # As a last resort, try to find a Linear module at the end
    for name, mod in list(model.named_modules())[::-1]:
        if isinstance(mod, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            if parent_name == "":
                setattr(model, name, nn.Linear(mod.in_features, 1))
            else:
                parent = model
                parts = parent_name.split(".")
                for p in parts:
                    parent = getattr(parent, p)
                setattr(parent, name.split(".")[-1], nn.Linear(mod.in_features, 1))
            return
    raise RuntimeError("Could not replace classifier with single-logit layer; unknown architecture")


def load_model(checkpoint_path: Optional[str] = None, device: str = "cpu") -> nn.Module:
    """Instantiate an Xception model and optionally load checkpoint.

    Args:
        checkpoint_path: optional path to a PyTorch checkpoint (`state_dict` or full checkpoint)
        device: "cpu" or "cuda" (if available)

    Returns:
        model in eval() mode on the specified device.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    # Create xception via timm
    # 'xception' is supported in recent timm releases; if unavailable, timm will raise.
    model = timm.create_model("xception", pretrained=True)
    _replace_classifier_with_single_logit(model)

    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=dev)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            print(f"[OK] Successfully loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.to(dev)
    model.eval()
    return model


def predict_batch(model: nn.Module, batch_tensor: torch.Tensor) -> torch.Tensor:
    """Run a forward pass on a batch and return probabilities (sigmoid of logits).

    Args:
        model: PyTorch module (single-logit output)
        batch_tensor: Tensor of shape (B, C, H, W)

    Returns:
        Tensor of shape (B,) with float probabilities in [0,1]
    """
    device = next(model.parameters()).device
    batch_tensor = batch_tensor.to(device)
    with torch.no_grad():
        logits = model(batch_tensor)
        # logits may be shape (B,1) or (B,)
        logits = logits.view(-1)
        probs = torch.sigmoid(logits)
    return probs

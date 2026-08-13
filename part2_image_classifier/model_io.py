"""Part 3's `classify_product_image` tool imports load_model/predict_image directly from
here rather than reimplementing model loading — keep these two signatures stable.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from part2_image_classifier import config
from part2_image_classifier.features import ProductClassifier

# cache so Part 3's agent doesn't reload ~45MB of weights on every tool call
_CACHED_MODEL = None
_CACHED_CLASSES = None
_CACHED_DEVICE = None


def load_model(path: Path = config.MODEL_PATH, device: str | None = None):
    """Rebuild resnet18 + head from the checkpoint, load weights, .eval(), return (model, classes)."""
    device = device or config.get_device()
    checkpoint = torch.load(path, map_location=device)

    model = ProductClassifier()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    classes = checkpoint["classes"]
    return model, classes


def _get_cached_model():
    global _CACHED_MODEL, _CACHED_CLASSES, _CACHED_DEVICE
    if _CACHED_MODEL is None:
        _CACHED_DEVICE = config.get_device()
        _CACHED_MODEL, _CACHED_CLASSES = load_model(device=_CACHED_DEVICE)
    return _CACHED_MODEL, _CACHED_CLASSES


def predict_image(image_path: str | Path, model=None, classes=None) -> dict:
    """Open a PNG, apply the same eval transform used at training time, forward, softmax.

    Returns a dict with label, confidence, class_index, and top3 (class name -> probability).
    """
    if model is None or classes is None:
        cached_model, cached_classes = _get_cached_model()
        model = model if model is not None else cached_model
        classes = classes if classes is not None else cached_classes

    device = next(model.parameters()).device

    img = Image.open(image_path).convert("L")  # ensure single-channel grayscale input
    tensor = config.EVAL_TRANSFORM(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_index = int(probs.argmax())
    top3_indices = probs.argsort()[::-1][:3]
    top3 = {classes[i]: float(probs[i]) for i in top3_indices}

    return {
        "label": classes[class_index],
        "confidence": float(probs[class_index]),
        "class_index": class_index,
        "top3": top3,
    }

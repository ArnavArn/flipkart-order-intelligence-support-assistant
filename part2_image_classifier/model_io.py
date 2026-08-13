"""Task 7 — THE documented snippet. Part 3's `classify_product_image` tool imports this file
and calls exactly these two public functions; it does not re-implement model loading.

    from part2_image_classifier.model_io import load_model, predict_image
    model, classes = load_model()
    result = predict_image("data/sample_images/00_tshirt_top.png", model=model, classes=classes)

The eval transform used below is imported from config.py — the SAME object used at training
time (config.EVAL_TRANSFORM == config.TRAIN_TRANSFORM, no augmentation) — so there is no
train/inference transform mismatch.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from part2_image_classifier import config
from part2_image_classifier.features import ProductClassifier

# Module-level cache so the agent (Part 3) doesn't reload ~45MB of weights on every tool call.
_CACHED_MODEL = None
_CACHED_CLASSES = None
_CACHED_DEVICE = None


def load_model(path: Path = config.MODEL_PATH, device: str | None = None):
    """Rebuild resnet18 + head from the checkpoint config, load weights, .eval(), return (model, classes).

    Returns:
        model: nn.Module (ProductClassifier: backbone + head), already .eval() and on `device`.
        classes: list[str] of the 10 class names, in checkpoint/label order.
    """
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
    """Open a PNG with PIL, apply the SAME eval transform used at training time, forward, softmax.

    Returns:
        dict with keys:
          - "label": str, the predicted class name
          - "confidence": float, softmax probability of the predicted class
          - "class_index": int, index of the predicted class in `classes`
          - "top3": dict mapping the top-3 class names to their softmax probabilities
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

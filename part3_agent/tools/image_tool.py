"""classify_product_image(image_path) -> dict. Calls Part 2's load_model/predict_image directly
rather than reimplementing the transform/forward pass; resolves a bare filename against data/sample_images/.
"""
from __future__ import annotations

from pathlib import Path

from part2_image_classifier.model_io import load_model, predict_image
from part3_agent import config

_MODEL = None
_CLASSES = None


def _load():
    global _MODEL, _CLASSES
    if _MODEL is None:
        _MODEL, _CLASSES = load_model()
    return _MODEL, _CLASSES


def _resolve_path(image_path: str) -> Path:
    p = Path(image_path)
    if p.exists():
        return p
    candidate = config.SAMPLE_IMAGES_DIR / p.name
    return candidate


def classify_product_image(image_path: str) -> dict:
    if not image_path or not image_path.strip():
        return {"error": "no image path given", "category": None, "confidence": 0.0}

    p = _resolve_path(image_path)
    if not p.exists() or p.is_dir():
        return {"error": f"image not found: {image_path}", "category": None, "confidence": 0.0}

    model, classes = _load()
    out = predict_image(p, model=model, classes=classes)
    try:
        display_path = str(p.resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        display_path = str(p)
    return {
        "category": out["label"],
        "confidence": round(out["confidence"], 4),
        "image_path": display_path,
        "top3": {k: round(v, 4) for k, v in out["top3"].items()},
        "model": "ResNet-18 transfer learning (Part 2)",
    }


if __name__ == "__main__":
    print(classify_product_image("07_sneaker.png"))

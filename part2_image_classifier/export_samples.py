"""Task 8 — ENTRY POINT: python -m part2_image_classifier.export_samples

Exports the first test-split image found for each of the 10 classes as real PNGs (deterministic,
no randomness) to data/sample_images/, writes labels.json, then runs predict_image on all 10 and
records predicted vs true (appended to reports/03_test_evaluation.md).
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST

from part2_image_classifier import config
from part2_image_classifier.model_io import load_model, predict_image

FILENAME_SLUGS = [
    "00_tshirt_top", "01_trouser", "02_pullover", "03_dress", "04_coat",
    "05_sandal", "06_shirt", "07_sneaker", "08_bag", "09_ankle_boot",
]


def export_samples() -> dict:
    """Export one real PNG per class (first matching index in the test split). Deterministic."""
    test_set = FashionMNIST(root=str(config.DATA_DIR), train=False, download=True)
    labels_np = test_set.targets.numpy()

    labels_json = {}
    for class_idx, slug in enumerate(FILENAME_SLUGS):
        matches = np.where(labels_np == class_idx)[0]
        test_split_index = int(matches[0])  # first test-split index for this class

        img_array = test_set.data[test_split_index].numpy()  # raw uint8 28x28
        filename = f"{slug}.png"
        out_path = config.SAMPLE_IMAGES_DIR / filename
        Image.fromarray(img_array).save(out_path)

        labels_json[filename] = {
            "true_label": config.CLASSES[class_idx],
            "class_index": class_idx,
            "test_split_index": test_split_index,
        }
        print(f"[export] {filename}  <- test_set[{test_split_index}]  true_label={config.CLASSES[class_idx]}")

    labels_path = config.SAMPLE_IMAGES_DIR / "labels.json"
    labels_path.write_text(json.dumps(labels_json, indent=2))
    print(f"[export] wrote {labels_path}")
    return labels_json


def self_check(labels_json: dict) -> list[dict]:
    """Run predict_image on all 10 exported PNGs, print + return predicted vs true."""
    model, classes = load_model()
    results = []
    n_correct = 0
    for filename, info in labels_json.items():
        image_path = config.SAMPLE_IMAGES_DIR / filename
        pred = predict_image(image_path, model=model, classes=classes)
        correct = pred["label"] == info["true_label"]
        n_correct += int(correct)
        results.append({
            "filename": filename,
            "true_label": info["true_label"],
            "predicted_label": pred["label"],
            "confidence": pred["confidence"],
            "correct": correct,
        })
        mark = "OK" if correct else "MISS"
        print(f"[self-check] {filename}: true={info['true_label']!r} pred={pred['label']!r} "
              f"conf={pred['confidence']:.3f} [{mark}]")
    print(f"[self-check] {n_correct}/{len(labels_json)} correct")
    return results


def append_self_check_to_report(results: list[dict]) -> None:
    report_path = config.REPORTS_DIR / "03_test_evaluation.md"
    lines = []
    lines.append("\n## Sample-image self-check (export_samples.py)\n")
    lines.append(
        "Ran `predict_image` (the exact `model_io.py` function Part 3 calls) on the 10 exported "
        "test-split PNGs in `data/sample_images/`.\n"
    )
    lines.append("| File | True label | Predicted label | Confidence | Correct |")
    lines.append("|---|---|---|---|---|")
    n_correct = 0
    for r in results:
        n_correct += int(r["correct"])
        lines.append(
            f"| {r['filename']} | {r['true_label']} | {r['predicted_label']} | "
            f"{r['confidence']:.4f} | {'yes' if r['correct'] else 'no'} |"
        )
    lines.append(f"\n**{n_correct}/{len(results)} correct.**\n")

    if report_path.exists():
        with report_path.open("a") as f:
            f.write("\n".join(lines))
    else:
        report_path.write_text("\n".join(lines))
    print(f"[report] appended sample-image self-check to {report_path}")


def main():
    labels_json = export_samples()
    results = self_check(labels_json)
    append_self_check_to_report(results)


if __name__ == "__main__":
    main()

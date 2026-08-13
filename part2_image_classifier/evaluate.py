"""Evaluate once on the untouched 10,000-image test split; writes reports 03-05 plus
confusion_matrix.csv.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from part2_image_classifier import config

# Hand-written visual-similarity explanations, keyed by unordered class-name pair. Only used
# if that pair actually shows up among the top off-diagonal cells of the real confusion matrix.
KNOWN_PAIR_EXPLANATIONS = {
    frozenset({"Shirt", "T-shirt/top"}): (
        "Shirt and T-shirt/top are both short-sleeved torso garments with the same rectangular "
        "body outline and the same shoulder-to-hem aspect ratio. At 28x28 grayscale the only "
        "cues that would distinguish them — a collar edge or a button placket — are a handful "
        "of pixels wide in the original image; they do not survive the downsampling to 28x28, "
        "nor does upsampling to 224x224 recover detail that was never captured. The model is "
        "left with two near-identical silhouettes and has to guess from faint shading cues."
    ),
    frozenset({"Pullover", "Coat"}): (
        "Pullover and Coat are both long-sleeved outer torso garments that occupy an almost "
        "identical T-shaped silhouette in a 28x28 thumbnail. The feature that actually "
        "separates them — a coat's open front placket — is a thin vertical seam of intensity "
        "change, typically one or two pixels wide at native resolution. That is below the "
        "effective resolution of the sensor once the garment is centered and downsampled, so "
        "the model frequently cannot tell a buttoned coat from a solid pullover."
    ),
    frozenset({"Shirt", "Coat"}): (
        "Shirt sits between the sleeved-torso classes and picks up confusion from both "
        "directions. Against Coat, the difference is again the coat's open-front seam versus "
        "the shirt's closed body outline — a cue that is only a pixel or two wide and is easily "
        "lost at 28x28, so errors between the two are visually plausible rather than random."
    ),
    frozenset({"Shirt", "Pullover"}): (
        "Shirt and Pullover share the same long-torso, set-in-sleeve silhouette; a pullover has "
        "no front opening at all and a shirt's placket is a thin, easily-lost line at this "
        "resolution, so the two classes overlap heavily in pixel space even though they are "
        "visually distinct garments at full resolution."
    ),
    frozenset({"Sandal", "Sneaker"}): (
        "Sandal and Sneaker are both low-profile footwear with a flat sole and a low ankle "
        "line, giving them almost the same overall outline in a 28x28 thumbnail. The feature "
        "that actually separates them is the sandal's strap gaps — thin dark regions inside the "
        "silhouette — which are exactly the kind of small, low-contrast detail that gets "
        "smeared out by downsampling to 28x28, so the model regularly mixes the two up."
    ),
}


def _generic_pair_explanation(class_a: str, class_b: str) -> str:
    return (
        f"{class_a} and {class_b} occupy very similar outlines once reduced to a 28x28 "
        f"grayscale thumbnail. Whatever detail would normally separate them at full "
        f"resolution — trim, fasteners, or thin structural elements — is only a few pixels "
        f"wide in the original image and does not survive the downsampling, so the model is "
        f"working from two silhouettes that are nearly indistinguishable in pixel space."
    )


@torch.no_grad()
def run_test_evaluation(model, device: str, test_dataset,
                         cached_feats: np.ndarray | None = None,
                         cached_labels: np.ndarray | None = None) -> dict:
    """Evaluate `model` once on the test split. If cached feats/labels are given (frozen-backbone
    path) run just the head over them; otherwise (fine-tuned path) run the full model on raw images.
    """
    model.eval()

    if cached_feats is not None and cached_labels is not None:
        print("[evaluate] using cached test features (frozen-backbone path)")
        feats_t = torch.from_numpy(cached_feats).to(device)
        logits = model.head(feats_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = cached_labels
    else:
        print("[evaluate] running full model over raw test images (fine-tuned path)")
        loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
        preds, trues = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)

    accuracy = float((y_pred == y_true).mean())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)

    report_dict = classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), target_names=config.CLASSES,
        output_dict=True, zero_division=0,
    )

    # top off-diagonal cells, found programmatically (not guessed)
    off_diag = []
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if i != j:
                off_diag.append((cm[i, j], i, j))
    off_diag.sort(reverse=True)
    top_pairs = []
    seen_unordered = set()
    for count, i, j in off_diag:
        if count <= 0:
            continue
        key = frozenset({i, j})
        if key in seen_unordered:
            continue  # report each unordered pair once, using its larger direction
        seen_unordered.add(key)
        top_pairs.append({
            "true_idx": i, "pred_idx": j,
            "true_class": config.CLASSES[i], "pred_class": config.CLASSES[j],
            "count": int(count),
            "reverse_count": int(cm[j, i]),
        })
        if len(top_pairs) >= 5:
            break

    return {
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "classification_report": report_dict,
        "top_pairs": top_pairs,
    }


def write_evaluation_reports(result: dict) -> None:
    _write_03_test_evaluation(result)
    _write_04_confusion_matrix(result)
    _write_05_confusion_analysis(result)


def _write_03_test_evaluation(result: dict) -> None:
    acc = result["accuracy"]
    gate = 0.80
    lines = []
    lines.append("# 03 — Test Evaluation\n")
    lines.append(
        "Evaluated **once**, on the untouched 10,000-image Fashion-MNIST test split, after "
        "the feature-extraction / fine-tuning decision in report 02 was finalized.\n"
    )
    lines.append(f"## Overall test accuracy\n")
    lines.append(f"**{acc:.4f}** ({acc * 100:.2f}%)\n")
    lines.append(f"Acceptance gate: >= {gate:.2f}. Result: **{'PASS' if acc >= gate else 'FAIL'}**.\n")

    lines.append("## Per-class precision / recall / f1 / support\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    rep = result["classification_report"]
    for c in config.CLASSES:
        row = rep[c]
        lines.append(
            f"| {c} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | "
            f"{int(row['support'])} |"
        )
    lines.append("")
    lines.append(
        f"| **accuracy** | | | **{rep['accuracy']:.4f}** | {int(rep['macro avg']['support'])} |"
    )
    lines.append(
        f"| **macro avg** | {rep['macro avg']['precision']:.4f} | "
        f"{rep['macro avg']['recall']:.4f} | {rep['macro avg']['f1-score']:.4f} | "
        f"{int(rep['macro avg']['support'])} |"
    )
    lines.append(
        f"| **weighted avg** | {rep['weighted avg']['precision']:.4f} | "
        f"{rep['weighted avg']['recall']:.4f} | {rep['weighted avg']['f1-score']:.4f} | "
        f"{int(rep['weighted avg']['support'])} |"
    )

    (config.REPORTS_DIR / "03_test_evaluation.md").write_text("\n".join(lines))
    print("[report] wrote reports/03_test_evaluation.md")


def _write_04_confusion_matrix(result: dict) -> None:
    cm = result["confusion_matrix"]
    cm_norm = result["confusion_matrix_normalized"]

    # CSV (raw counts)
    csv_lines = ["true_class," + ",".join(config.CLASSES)]
    for i, c in enumerate(config.CLASSES):
        csv_lines.append(c + "," + ",".join(str(int(v)) for v in cm[i]))
    (config.REPORTS_DIR / "confusion_matrix.csv").write_text("\n".join(csv_lines))

    lines = []
    lines.append("# 04 — Confusion Matrix (Test Split)\n")
    lines.append("Rows = true class, columns = predicted class. Raw counts.\n")

    header = "| true \\ pred | " + " | ".join(config.CLASSES) + " |"
    sep = "|---" * (config.NUM_CLASSES + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for i, c in enumerate(config.CLASSES):
        row = " | ".join(str(int(v)) for v in cm[i])
        lines.append(f"| **{c}** | {row} |")

    lines.append("\n## Normalized (row = true class, values sum to 1 per row)\n")
    lines.append(header)
    lines.append(sep)
    for i, c in enumerate(config.CLASSES):
        row = " | ".join(f"{v:.3f}" for v in cm_norm[i])
        lines.append(f"| **{c}** | {row} |")

    lines.append("\nRaw counts also saved to `reports/confusion_matrix.csv`.\n")

    (config.REPORTS_DIR / "04_confusion_matrix.md").write_text("\n".join(lines))
    print("[report] wrote reports/04_confusion_matrix.md")


def _write_05_confusion_analysis(result: dict) -> None:
    top_pairs = result["top_pairs"]
    lines = []
    lines.append("# 05 — Confusion Analysis\n")
    lines.append(
        "The pairs below were found **programmatically** from the real confusion matrix in "
        "report 04 (largest off-diagonal cells, deduplicated by unordered class pair) — they "
        "were not guessed in advance.\n"
    )
    lines.append("## Largest confused pairs (from the actual matrix)\n")
    lines.append("| True class | Predicted class | Count (true->pred) | Count (pred->true) |")
    lines.append("|---|---|---|---|")
    for p in top_pairs[:3]:
        lines.append(
            f"| {p['true_class']} | {p['pred_class']} | {p['count']} | {p['reverse_count']} |"
        )
    lines.append("")

    lines.append("## Why these pairs are visually plausible at 28x28\n")
    for p in top_pairs[:3]:
        explanation = KNOWN_PAIR_EXPLANATIONS.get(
            frozenset({p["true_class"], p["pred_class"]})
        )
        if explanation is None:
            explanation = _generic_pair_explanation(p["true_class"], p["pred_class"])
        lines.append(f"### {p['true_class']} <-> {p['pred_class']}\n")
        lines.append(explanation + "\n")

    lines.append(
        "As the trainer's line goes: an error is not a wrong answer — it is the model telling "
        "you where the visual signal genuinely is ambiguous.\n"
    )

    (config.REPORTS_DIR / "05_confusion_analysis.md").write_text("\n".join(lines))
    print("[report] wrote reports/05_confusion_analysis.md")

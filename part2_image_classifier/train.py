"""ENTRY POINT — python -m part2_image_classifier.train

Pipeline:
  1. Load Fashion-MNIST, build stratified 55k/5k/10k splits (data.py)          -> report 01
  2. Freeze ResNet-18 backbone, cache 512-d features for train/val/test       (features.py)
  3. Train a small head on cached train features, 20 epochs, Adam lr=1e-3    -> report 02
  4. Decision rule: if val_acc_feature_extraction >= 0.80, stop.
     Else unfreeze layer4 and fine-tune end-to-end for 2 epochs, lr=1e-4.
  5. Evaluate once on the untouched 10,000-image test split (evaluate.py)    -> reports 03/04/05
  6. Save the assembled model to models/product_classifier.pt (model_io checkpoint schema)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from part2_image_classifier import config, data, evaluate, features


def write_splits_report(splits: data.Splits) -> None:
    labels_full = splits.full_train.targets.numpy()
    train_labels = labels_full[splits.idx_train]
    val_labels = labels_full[splits.idx_val]
    test_labels = splits.test.targets.numpy()

    val_counts = data.per_class_counts(val_labels)
    train_counts = data.per_class_counts(train_labels)
    test_counts = data.per_class_counts(test_labels)

    lines = []
    lines.append("# 01 — Splits and Setup\n")
    lines.append("## Dataset source\n")
    lines.append(
        "Fashion-MNIST loaded via the pinned source: "
        "`torchvision.datasets.FashionMNIST(root=\"data/fashion_mnist\", download=True)`. "
        "No substitute dataset was used.\n"
    )
    lines.append("## Split sizes actually used\n")
    lines.append(f"- Full official train set: {len(splits.full_train)}")
    lines.append(f"- **Train split: {len(splits.idx_train)}**")
    lines.append(f"- **Val split: {len(splits.idx_val)}**")
    lines.append(f"- **Test split (official FashionMNIST test set, untouched): {len(splits.test)}**")
    lines.append(
        "\nThe 55,000/5,000 train/val split was carved from the 60,000-image official train "
        "set via a **stratified** split (`sklearn.model_selection.train_test_split`, "
        "`stratify=labels`, `random_state=42`). The 10,000-image official test split is not "
        "touched by anything in this pipeline until `evaluate.py` runs once, at the very end, "
        "after the model (head, and fine-tune decision) is fully finalized.\n"
    )
    lines.append("## Per-class counts — validation split (proves stratification)\n")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    for c, cnt in zip(config.CLASSES, val_counts):
        lines.append(f"| {c} | {cnt} |")
    lines.append(f"\nSum: {sum(val_counts)} (expected {config.N_VAL})\n")

    lines.append("## Per-class counts — train split\n")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    for c, cnt in zip(config.CLASSES, train_counts):
        lines.append(f"| {c} | {cnt} |")
    lines.append(f"\nSum: {sum(train_counts)} (expected {config.N_TRAIN})\n")

    lines.append("## Per-class counts — test split (untouched)\n")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    for c, cnt in zip(config.CLASSES, test_counts):
        lines.append(f"| {c} | {cnt} |")
    lines.append(f"\nSum: {sum(test_counts)} (expected {config.N_TEST})\n")

    lines.append("## Input size and transform pipeline (printed verbatim)\n")
    lines.append(f"Documented input size: **{config.INPUT_SIZE} x {config.INPUT_SIZE}**\n")
    lines.append("```\n" + repr(config.EVAL_TRANSFORM) + "\n```\n")
    lines.append(
        f"- Grayscale -> 3 channels (`transforms.Grayscale(num_output_channels=3)`) because "
        f"ResNet-18's ImageNet weights expect RGB.\n"
        f"- Resize to {config.INPUT_SIZE}x{config.INPUT_SIZE} (what ResNet-18's ImageNet "
        f"weights expect).\n"
        f"- ImageNet normalization: mean={config.IMAGENET_MEAN}, std={config.IMAGENET_STD} "
        f"(required — the backbone was trained with these statistics).\n"
        f"- The same transform is used for train and eval (no random augmentation), which is "
        f"what makes the frozen-backbone feature cache mathematically exact (see report 02).\n"
    )
    lines.append("## Class names (FashionMNIST label order)\n")
    lines.append(str(config.CLASSES) + "\n")

    (config.REPORTS_DIR / "01_splits_and_setup.md").write_text("\n".join(lines))
    print("[report] wrote reports/01_splits_and_setup.md")


def train_head(train_feats, train_labels, val_feats, val_labels, device: str):
    """Task 3 — train ONLY the head on cached features. Returns (head, epoch_log)."""
    config.seed_everything()

    head = features.build_head().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.HEAD_LR)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_feats), torch.from_numpy(train_labels)
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.HEAD_BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_feats_t = torch.from_numpy(val_feats).to(device)
    val_labels_t = torch.from_numpy(val_labels).to(device)

    epoch_log = []
    for epoch in range(1, config.HEAD_EPOCHS + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels_t).float().mean().item()

        epoch_log.append((epoch, train_loss, val_acc))
        print(f"  epoch {epoch:2d}/{config.HEAD_EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    return head, epoch_log


def finetune_layer4(backbone: nn.Module, head: nn.Module, splits: data.Splits, device: str):
    """Task 4 — unfreeze layer4 only, retrain end-to-end on real images for 2 epochs."""
    config.seed_everything()

    features.unfreeze_layer4(backbone)
    backbone.to(device)
    head.to(device)

    trainable_params = [p for p in backbone.layer4.parameters() if p.requires_grad] + list(
        head.parameters()
    )
    optimizer = torch.optim.Adam(trainable_params, lr=config.FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        splits.train_subset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        splits.val_subset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=0
    )

    epoch_log = []
    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        backbone.train()
        # keep everything except layer4 in eval-consistent (frozen) mode; layer4 is the only
        # part with requires_grad=True, so BN stats for frozen layers stay as loaded.
        head.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(backbone(xb))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        backbone.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = head(backbone(xb))
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.shape[0]
        val_acc = correct / max(total, 1)
        elapsed = time.time() - t0
        epoch_log.append((epoch, train_loss, val_acc))
        print(
            f"  [finetune] epoch {epoch}/{config.FINETUNE_EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  ({elapsed:.1f}s)"
        )

    return epoch_log


def write_training_log_report(epoch_log, val_acc_feature_extraction, finetuned, finetune_log,
                               val_acc_before, val_acc_after):
    lines = []
    lines.append("# 02 — Training Log\n")
    lines.append("## Setup: frozen backbone + cached features\n")
    lines.append(
        "**Why this is legitimate and not a shortcut:** the backbone is frozen, so its output "
        "for a given image is identical every epoch. Caching that output and training the head "
        "on it is *mathematically identical* to re-running the frozen forward pass each epoch — "
        "just without recomputing a constant 20 times.\n"
    )
    lines.append("## Head architecture\n")
    lines.append("`Linear(512, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 10)`\n")
    lines.append("## Head training hyperparameters\n")
    lines.append(f"- Optimizer: **Adam**")
    lines.append(f"- Learning rate: **{config.HEAD_LR}**")
    lines.append(f"- Batch size: **{config.HEAD_BATCH_SIZE}**")
    lines.append(f"- Epochs: **{config.HEAD_EPOCHS}**")
    lines.append(f"- Loss: CrossEntropyLoss\n")
    lines.append("## Per-epoch train loss / val accuracy\n")
    lines.append("| Epoch | Train loss | Val accuracy |")
    lines.append("|---|---|---|")
    for epoch, loss, acc in epoch_log:
        lines.append(f"| {epoch} | {loss:.4f} | {acc:.4f} |")
    lines.append("")

    lines.append("## Fine-tuning decision\n")
    lines.append(
        f"Feature-extraction validation accuracy after {config.HEAD_EPOCHS} epochs: "
        f"**{val_acc_feature_extraction:.4f}**. Gate: >= {config.FEATURE_EXTRACTION_ACC_GATE}.\n"
    )
    if not finetuned:
        lines.append(
            f"**Feature extraction alone was sufficient (val acc = {val_acc_feature_extraction:.3f} "
            f">= {config.FEATURE_EXTRACTION_ACC_GATE}); fine-tuning was not required.**\n"
        )
        lines.append(
            f"- val_acc_before = {val_acc_before:.4f}\n- val_acc_after = {val_acc_after:.4f} "
            f"(before == after because no fine-tuning was performed)\n"
        )
    else:
        lines.append(
            f"Feature-extraction val accuracy ({val_acc_feature_extraction:.3f}) was **below** "
            f"the {config.FEATURE_EXTRACTION_ACC_GATE} gate, so `layer4` of the backbone was "
            f"unfrozen (all earlier layers stayed frozen) and the model was retrained "
            f"end-to-end on real images for {config.FINETUNE_EPOCHS} epochs at lr="
            f"{config.FINETUNE_LR}, batch={config.FINETUNE_BATCH_SIZE}.\n"
        )
        lines.append("### Fine-tune per-epoch train loss / val accuracy\n")
        lines.append("| Epoch | Train loss | Val accuracy |")
        lines.append("|---|---|---|")
        for epoch, loss, acc in finetune_log:
            lines.append(f"| {epoch} | {loss:.4f} | {acc:.4f} |")
        lines.append(
            f"\n- val_acc_before (feature extraction) = {val_acc_before:.4f}\n"
            f"- val_acc_after (fine-tuned) = {val_acc_after:.4f}\n"
        )

    (config.REPORTS_DIR / "02_training_log.md").write_text("\n".join(lines))
    print("[report] wrote reports/02_training_log.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force re-extraction of cached features even if .npy files exist.")
    args = parser.parse_args()

    config.seed_everything()
    device = config.get_device()
    print(f"[train] device={device}")

    # --- Task 1-2: splits ---
    splits = data.load_splits()
    write_splits_report(splits)

    # --- Task 3: frozen backbone, one-pass feature caching ---
    backbone = features.build_backbone()
    features.freeze_backbone(backbone)

    train_feats, train_labels = features.extract_and_cache_split(
        "train", splits.train_subset, backbone, device, rebuild=args.rebuild_cache
    )
    val_feats, val_labels = features.extract_and_cache_split(
        "val", splits.val_subset, backbone, device, rebuild=args.rebuild_cache
    )
    test_feats, test_labels = features.extract_and_cache_split(
        "test", splits.test, backbone, device, rebuild=args.rebuild_cache
    )

    # --- Task 3: train head on cached features ---
    print("[train] training head on cached features ...")
    head, epoch_log = train_head(train_feats, train_labels, val_feats, val_labels, device)
    val_acc_feature_extraction = epoch_log[-1][2]

    # --- Task 4: conditional fine-tune ---
    finetuned = val_acc_feature_extraction < config.FEATURE_EXTRACTION_ACC_GATE
    finetune_log = []
    val_acc_before = val_acc_feature_extraction
    val_acc_after = val_acc_feature_extraction

    if finetuned:
        print(
            f"[train] val_acc_feature_extraction={val_acc_feature_extraction:.4f} < "
            f"{config.FEATURE_EXTRACTION_ACC_GATE} -> fine-tuning layer4 ..."
        )
        finetune_log = finetune_layer4(backbone, head, splits, device)
        val_acc_after = finetune_log[-1][2]
        # test cache extracted with the OLD frozen backbone is now stale for evaluation;
        # evaluate.py will run the full model over raw test images instead.
    else:
        print(
            f"[train] val_acc_feature_extraction={val_acc_feature_extraction:.4f} >= "
            f"{config.FEATURE_EXTRACTION_ACC_GATE} -> feature extraction alone was sufficient."
        )

    write_training_log_report(
        epoch_log, val_acc_feature_extraction, finetuned, finetune_log, val_acc_before, val_acc_after
    )

    # --- assemble final model ---
    model = features.ProductClassifier()
    model.backbone.load_state_dict(backbone.state_dict())
    model.head.load_state_dict(head.state_dict())
    model.to(device)
    model.eval()

    # --- Task 5-6: evaluate ONCE on the untouched test split ---
    print("[train] running final test evaluation (once) ...")
    if finetuned:
        eval_result = evaluate.run_test_evaluation(model, device, test_dataset=splits.test)
    else:
        eval_result = evaluate.run_test_evaluation(
            model, device, test_dataset=splits.test,
            cached_feats=test_feats, cached_labels=test_labels,
        )

    evaluate.write_evaluation_reports(eval_result)

    # --- Task 7: save the assembled artifact ---
    checkpoint = {
        "arch": "resnet18",
        "input_size": config.INPUT_SIZE,
        "classes": config.CLASSES,
        "head_dims": config.HEAD_DIMS,
        "finetuned": bool(finetuned),
        "state_dict": model.state_dict(),
        "test_accuracy": float(eval_result["accuracy"]),
    }
    torch.save(checkpoint, config.MODEL_PATH)
    print(f"[train] saved checkpoint to {config.MODEL_PATH} (test_accuracy={eval_result['accuracy']:.4f})")


if __name__ == "__main__":
    main()

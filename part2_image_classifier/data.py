"""Download Fashion-MNIST, build the stratified 55k/5k/10k splits. Test split is untouched
until evaluate.py runs at the very end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST

from part2_image_classifier import config


@dataclass
class Splits:
    full_train: FashionMNIST
    test: FashionMNIST
    idx_train: np.ndarray
    idx_val: np.ndarray
    train_subset: Subset
    val_subset: Subset


def load_splits() -> Splits:
    """Download (if needed) and build the stratified 55,000/5,000 train/val split."""
    config.seed_everything()

    full_train = FashionMNIST(
        root=str(config.DATA_DIR), train=True, download=True, transform=config.TRAIN_TRANSFORM
    )
    test = FashionMNIST(
        root=str(config.DATA_DIR), train=False, download=True, transform=config.EVAL_TRANSFORM
    )

    labels = full_train.targets.numpy()
    idx_train, idx_val = train_test_split(
        np.arange(len(full_train)),
        test_size=config.N_VAL,
        stratify=labels,
        random_state=config.SEED,
    )

    train_subset = Subset(full_train, idx_train)
    val_subset = Subset(full_train, idx_val)

    return Splits(
        full_train=full_train,
        test=test,
        idx_train=idx_train,
        idx_val=idx_val,
        train_subset=train_subset,
        val_subset=val_subset,
    )


def per_class_counts(labels: np.ndarray, num_classes: int = config.NUM_CLASSES) -> list[int]:
    return [int((labels == c).sum()) for c in range(num_classes)]

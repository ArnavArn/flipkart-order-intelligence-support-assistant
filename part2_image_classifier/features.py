"""Frozen ResNet-18 backbone, the assembled model, and one-pass feature caching.

The backbone is frozen, so its output per image is constant across epochs — we run one
forward pass per split and cache the 512-d vectors instead of recomputing them every epoch.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from part2_image_classifier import config


def build_backbone() -> nn.Module:
    """ResNet-18 pretrained on ImageNet with the final fc stripped to Identity (512-d output)."""
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    return backbone


def freeze_backbone(backbone: nn.Module) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()


def unfreeze_layer4(backbone: nn.Module) -> None:
    """Fine-tune path: unfreeze only layer4, keep everything earlier frozen."""
    for p in backbone.layer4.parameters():
        p.requires_grad = True


def build_head() -> nn.Sequential:
    d_in, d_hidden, d_out = config.HEAD_DIMS
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(d_hidden, d_out),
    )


class ProductClassifier(nn.Module):
    """The assembled model saved to models/product_classifier.pt: backbone + head."""

    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()
        self.head = build_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def _cache_paths(split: str) -> tuple:
    return (
        config.CACHE_DIR / f"{split}_feats.npy",
        config.CACHE_DIR / f"{split}_labels.npy",
    )


def cache_exists(split: str) -> bool:
    feats_path, labels_path = _cache_paths(split)
    return feats_path.exists() and labels_path.exists()


@torch.no_grad()
def extract_features(dataset, backbone: nn.Module, device: str,
                      batch_size: int = config.FEATURE_EXTRACTION_BATCH_SIZE):
    """One frozen forward pass over `dataset`. Returns (feats float32 (N,512), labels int64 (N,))."""
    backbone = backbone.to(device)
    backbone.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_feats = []
    all_labels = []
    t0 = time.time()
    n_done = 0
    for images, labels in loader:
        images = images.to(device)
        feats = backbone(images)
        all_feats.append(feats.cpu().numpy().astype(np.float32))
        all_labels.append(labels.numpy().astype(np.int64))
        n_done += images.shape[0]
    elapsed = time.time() - t0
    print(f"  extracted features for {n_done} images in {elapsed:.1f}s "
          f"({n_done / max(elapsed, 1e-6):.1f} img/s) on device={device}")

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


def extract_and_cache_split(split: str, dataset, backbone: nn.Module, device: str,
                             rebuild: bool = False):
    feats_path, labels_path = _cache_paths(split)
    if not rebuild and cache_exists(split):
        print(f"[features] cache hit for split={split!r}, loading {feats_path.name}")
        return np.load(feats_path), np.load(labels_path)

    print(f"[features] extracting features for split={split!r} (n={len(dataset)}) ...")
    feats, labels = extract_features(dataset, backbone, device)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(feats_path, feats)
    np.save(labels_path, labels)
    return feats, labels

"""Part 2 — fixed decisions, paths, and the single source of truth for the eval transform.

Nothing here should be re-litigated mid-build (see docs/03_PART2_PLAN.md). In particular:
- the eval transform defined here is imported by BOTH train.py/features.py and model_io.py,
  so training and inference can never silently diverge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths — always resolved relative to the repo root, never hardcoded absolute.
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent          # .../part2_image_classifier
REPO_ROOT = PACKAGE_DIR.parents[0]                       # .../capstone

DATA_DIR = REPO_ROOT / "data" / "fashion_mnist"          # gitignored raw IDX files
CACHE_DIR = PACKAGE_DIR / "cache"                          # gitignored .npy feature cache
REPORTS_DIR = PACKAGE_DIR / "reports"                       # committed markdown reports
MODELS_DIR = REPO_ROOT / "models"                         # committed artifacts
SAMPLE_IMAGES_DIR = REPO_ROOT / "data" / "sample_images"     # committed PNGs + labels.json

MODEL_PATH = MODELS_DIR / "product_classifier.pt"

for _d in (CACHE_DIR, REPORTS_DIR, MODELS_DIR, SAMPLE_IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed decisions
# ---------------------------------------------------------------------------
SEED = 42

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASSES)

# Splits
N_VAL = 5000
N_TRAIN = 60000 - N_VAL  # 55,000
N_TEST = 10000

# Head training (Task 3)
HEAD_LR = 1e-3
HEAD_BATCH_SIZE = 256
HEAD_EPOCHS = 20
HEAD_DIMS = [512, 256, NUM_CLASSES]

# Fine-tune (Task 4, conditional)
FINETUNE_LR = 1e-4
FINETUNE_BATCH_SIZE = 128
FINETUNE_EPOCHS = 2
FEATURE_EXTRACTION_ACC_GATE = 0.80

FEATURE_EXTRACTION_BATCH_SIZE = 256  # used only for the one-pass feature extraction forward


def get_device() -> str:
    """mps if available else cpu — resolved once, used everywhere."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def seed_everything(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# The ONE eval transform. Imported by data.py (for train/val/test datasets),
# features.py (for feature extraction), train.py (fine-tune path) and
# model_io.py (single-image inference). Do not redefine this elsewhere.
# ---------------------------------------------------------------------------
EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Fashion-MNIST is a fixed, non-augmented classification benchmark here — we use the same
# deterministic transform for train and eval (no random augmentation), which keeps the
# "features are a pure function of the frozen backbone" argument in docs/03_PART2_PLAN.md exact.
TRAIN_TRANSFORM = EVAL_TRANSFORM

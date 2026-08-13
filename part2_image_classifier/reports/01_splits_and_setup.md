# 01 — Splits and Setup

## Dataset source

Fashion-MNIST loaded via the pinned source: `torchvision.datasets.FashionMNIST(root="data/fashion_mnist", download=True)`. No substitute dataset was used.

## Split sizes actually used

- Full official train set: 60000
- **Train split: 55000**
- **Val split: 5000**
- **Test split (official FashionMNIST test set, untouched): 10000**

The 55,000/5,000 train/val split was carved from the 60,000-image official train set via a **stratified** split (`sklearn.model_selection.train_test_split`, `stratify=labels`, `random_state=42`). The 10,000-image official test split is not touched by anything in this pipeline until `evaluate.py` runs once, at the very end, after the model (head, and fine-tune decision) is fully finalized.

## Per-class counts — validation split (proves stratification)

| Class | Count |
|---|---|
| T-shirt/top | 500 |
| Trouser | 500 |
| Pullover | 500 |
| Dress | 500 |
| Coat | 500 |
| Sandal | 500 |
| Shirt | 500 |
| Sneaker | 500 |
| Bag | 500 |
| Ankle boot | 500 |

Sum: 5000 (expected 5000)

## Per-class counts — train split

| Class | Count |
|---|---|
| T-shirt/top | 5500 |
| Trouser | 5500 |
| Pullover | 5500 |
| Dress | 5500 |
| Coat | 5500 |
| Sandal | 5500 |
| Shirt | 5500 |
| Sneaker | 5500 |
| Bag | 5500 |
| Ankle boot | 5500 |

Sum: 55000 (expected 55000)

## Per-class counts — test split (untouched)

| Class | Count |
|---|---|
| T-shirt/top | 1000 |
| Trouser | 1000 |
| Pullover | 1000 |
| Dress | 1000 |
| Coat | 1000 |
| Sandal | 1000 |
| Shirt | 1000 |
| Sneaker | 1000 |
| Bag | 1000 |
| Ankle boot | 1000 |

Sum: 10000 (expected 10000)

## Input size and transform pipeline (printed verbatim)

Documented input size: **224 x 224**

```
Compose(
    Grayscale(num_output_channels=3)
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

- Grayscale -> 3 channels (`transforms.Grayscale(num_output_channels=3)`) because ResNet-18's ImageNet weights expect RGB.
- Resize to 224x224 (what ResNet-18's ImageNet weights expect).
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (required — the backbone was trained with these statistics).
- The same transform is used for train and eval (no random augmentation), which is what makes the frozen-backbone feature cache mathematically exact (see report 02).

## Class names (FashionMNIST label order)

['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

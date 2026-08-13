# 03 — Test Evaluation

Evaluated **once**, on the untouched 10,000-image Fashion-MNIST test split, after the feature-extraction / fine-tuning decision in report 02 was finalized.

## Overall test accuracy

**0.9056** (90.56%)

Acceptance gate: >= 0.80. Result: **PASS**.

## Per-class precision / recall / f1 / support

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| T-shirt/top | 0.8099 | 0.8990 | 0.8521 | 1000 |
| Trouser | 0.9889 | 0.9780 | 0.9834 | 1000 |
| Pullover | 0.8477 | 0.8850 | 0.8659 | 1000 |
| Dress | 0.9190 | 0.8740 | 0.8960 | 1000 |
| Coat | 0.8351 | 0.8810 | 0.8574 | 1000 |
| Sandal | 0.9655 | 0.9800 | 0.9727 | 1000 |
| Shirt | 0.7903 | 0.6670 | 0.7234 | 1000 |
| Sneaker | 0.9485 | 0.9570 | 0.9527 | 1000 |
| Bag | 0.9811 | 0.9880 | 0.9846 | 1000 |
| Ankle boot | 0.9703 | 0.9470 | 0.9585 | 1000 |

| **accuracy** | | | **0.9056** | 10000 |
| **macro avg** | 0.9056 | 0.9056 | 0.9047 | 10000 |
| **weighted avg** | 0.9056 | 0.9056 | 0.9047 | 10000 |
## Sample-image self-check (export_samples.py)

Ran `predict_image` (the exact `model_io.py` function Part 3 calls) on the 10 exported test-split PNGs in `data/sample_images/`.

| File | True label | Predicted label | Confidence | Correct |
|---|---|---|---|---|
| 00_tshirt_top.png | T-shirt/top | T-shirt/top | 0.9999 | yes |
| 01_trouser.png | Trouser | Trouser | 1.0000 | yes |
| 02_pullover.png | Pullover | Pullover | 0.9998 | yes |
| 03_dress.png | Dress | Dress | 0.9965 | yes |
| 04_coat.png | Coat | Coat | 0.9562 | yes |
| 05_sandal.png | Sandal | Sandal | 1.0000 | yes |
| 06_shirt.png | Shirt | T-shirt/top | 0.8506 | no |
| 07_sneaker.png | Sneaker | Sneaker | 0.9999 | yes |
| 08_bag.png | Bag | Bag | 1.0000 | yes |
| 09_ankle_boot.png | Ankle boot | Ankle boot | 0.9984 | yes |

**9/10 correct.**

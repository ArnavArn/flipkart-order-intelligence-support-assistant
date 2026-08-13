# 02 — Training Log

## Setup: frozen backbone + cached features

**Why this is legitimate and not a shortcut:** the backbone is frozen, so its output for a given image is identical every epoch. Caching that output and training the head on it is *mathematically identical* to re-running the frozen forward pass each epoch — just without recomputing a constant 20 times.

## Head architecture

`Linear(512, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 10)`

## Head training hyperparameters

- Optimizer: **Adam**
- Learning rate: **0.001**
- Batch size: **256**
- Epochs: **20**
- Loss: CrossEntropyLoss

## Per-epoch train loss / val accuracy

| Epoch | Train loss | Val accuracy |
|---|---|---|
| 1 | 0.5110 | 0.8780 |
| 2 | 0.3521 | 0.8892 |
| 3 | 0.3156 | 0.8942 |
| 4 | 0.2958 | 0.8966 |
| 5 | 0.2794 | 0.9098 |
| 6 | 0.2667 | 0.9080 |
| 7 | 0.2555 | 0.9010 |
| 8 | 0.2489 | 0.9078 |
| 9 | 0.2405 | 0.9074 |
| 10 | 0.2334 | 0.9080 |
| 11 | 0.2255 | 0.9122 |
| 12 | 0.2179 | 0.9088 |
| 13 | 0.2132 | 0.9134 |
| 14 | 0.2087 | 0.9124 |
| 15 | 0.2054 | 0.9146 |
| 16 | 0.1951 | 0.9100 |
| 17 | 0.1899 | 0.9138 |
| 18 | 0.1857 | 0.9146 |
| 19 | 0.1786 | 0.9118 |
| 20 | 0.1753 | 0.9134 |

## Fine-tuning decision

Feature-extraction validation accuracy after 20 epochs: **0.9134**. Gate: >= 0.8.

**Feature extraction alone was sufficient (val acc = 0.913 >= 0.8); fine-tuning was not required.**

- val_acc_before = 0.9134
- val_acc_after = 0.9134 (before == after because no fine-tuning was performed)

# Part 2 — Product Image Categoriser via Transfer Learning (25 marks)

**Entry points:** `python -m part2_image_classifier.train`, then
`python -m part2_image_classifier.export_samples`
**Outputs:** `models/product_classifier.pt`, `data/sample_images/*.png`, 5 reports.

---

## Fixed decisions (write these into `config.py`, don't re-litigate them mid-build)

| Decision | Value | Why |
|---|---|---|
| Dataset | Fashion-MNIST via `torchvision.datasets.FashionMNIST(root="data/fashion_mnist", download=True)` | pinned source is `github.com/zalandoresearch/fashion-mnist`; torchvision pulls the same canonical files, no login, no key |
| Backbone | **ResNet-18**, `weights=ResNet18_Weights.IMAGENET1K_V1` | brief names it; small, fast, 512-d pooled features |
| Input size | **224 × 224** — document this exact number | what ResNet-18's ImageNet weights expect |
| Channels | grayscale → 3 via `transforms.Grayscale(num_output_channels=3)` | backbone expects RGB |
| Normalisation | ImageNet `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` | required — the backbone was trained with these |
| Splits | train **55,000** / val **5,000** (stratified) / test **10,000** | brief requires ≥5,000 val carved from the 60k train; test untouched |
| Optimizer | **Adam** | brief says use Adam |
| Head training | lr `1e-3`, batch `256`, `20` epochs on cached features | seconds, once features exist |
| Fine-tune (if needed) | unfreeze `layer4` only, lr `1e-4`, batch `128`, 2 epochs | standard gradual unfreezing |
| Device | `mps` if available else `cpu` | Apple Silicon |
| Seed | `torch.manual_seed(42)`, `np.random.seed(42)` | reproducibility |

Class names, in FashionMNIST label order — hardcode this list, it is fixed:

```python
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

---

## Task 1–2 — Data and preprocessing (`data.py`) → `reports/01_splits_and_setup.md`

```python
full_train = FashionMNIST(root=DATA_DIR, train=True,  download=True, transform=TFM)
test       = FashionMNIST(root=DATA_DIR, train=False, download=True, transform=TFM)

# stratified 55k / 5k using the labels, seed 42
idx_tr, idx_val = train_test_split(
    np.arange(len(full_train)), test_size=5000,
    stratify=full_train.targets.numpy(), random_state=42)
```

Report the **exact** split sizes actually used, plus the per-class counts in the validation
split (proves it is stratified), plus the transform pipeline printed verbatim, plus the
documented input size.

State in the report: **the 10,000-image test split is not touched until Task 5.**

---

## Task 3 — Transfer learning with cached features (`features.py`, `train.py`)

This is the speed trick the brief tells you to do. **Do it.**

```
1. Build ResNet-18 pretrained. Strip the final fc → nn.Identity().
   Freeze everything: for p in backbone.parameters(): p.requires_grad = False
   backbone.eval()

2. ONE pass over train / val / test with torch.no_grad():
       feats[split] : (N, 512) float32   →  part2_image_classifier/cache/{split}_feats.npy
       labels[split]: (N,)     int64     →  part2_image_classifier/cache/{split}_labels.npy

3. Train ONLY the head on the cached vectors:
       head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10))
       Adam(head.parameters(), lr=1e-3), CrossEntropyLoss, batch 256, 20 epochs
```

**Why this is legitimate and not a shortcut:** the backbone is frozen, so its output for a given
image is identical every epoch. Caching that output and training the head on it is
*mathematically identical* to re-running the frozen forward pass each epoch — just without
recomputing a constant 20 times. Say this sentence in the report.

Runtime on this machine (M-series, MPS, batch 256): feature extraction ≈ 5–12 min for all
70,000 images; head training ≈ 10 seconds total. Skip caching and you are looking at hours.

Cache files are ~150 MB total → **gitignored**. `train.py` must detect existing cache files and
skip re-extraction, but expose `--rebuild-cache`.

Document in `reports/02_training_log.md`: batch size, optimizer, learning rate, epoch count,
and the per-epoch train loss / val accuracy table.

---

## Task 4 — Fine-tune if needed → same report

Rule from the brief: **if feature-extraction val accuracy < 80%, fine-tune.** Either way you
must report the before/after numbers.

Expected: a linear/MLP probe on frozen ImageNet ResNet-18 features gets Fashion-MNIST val
accuracy around **85–89%**, so fine-tuning will most likely **not** be required.

`train.py` must therefore:
- compute `val_acc_feature_extraction`
- if `>= 0.80`: write **"Feature extraction alone was sufficient (val acc = X.XXX ≥ 0.80);
  fine-tuning was not required."** — and still record the number as "after = before".
- if `< 0.80`: unfreeze `layer4`, keep early/middle frozen, retrain end-to-end at lr `1e-4` for
  2 epochs (this path cannot use the cache — it needs real images through the unfrozen
  layers), then report `val_acc_after_finetune`.

Implement **both** branches. The criterion is "states explicitly which happened, with before/
after numbers either way."

---

## Task 5–6 — Evaluate (`evaluate.py`) → reports 03, 04, 05

Run **once**, on the untouched 10,000-image test split.

- `reports/03_test_evaluation.md` — overall test accuracy + `classification_report` as a
  markdown table (per-class precision / recall / f1 / support).
  **Acceptance gate: ≥ 80% test accuracy.**
- `reports/04_confusion_matrix.md` — the full **10 × 10** matrix as a markdown table with class
  names on both axes, plus `confusion_matrix.csv`. Generate a normalised version too (row =
  true class) — it makes the confusion pairs obvious.
- `reports/05_confusion_analysis.md` — **read the top off-diagonal cells programmatically** and
  write the two (or three) largest confused pairs into the report. Then hand-write one paragraph
  per pair.

**Do not guess the pairs.** But for planning, the ones that reliably dominate on Fashion-MNIST
are:

- **Shirt ↔ T-shirt/top** — nearly always the single biggest cell. Both are short-sleeved
  torso garments with the same rectangular body and the same shoulder-to-hem aspect ratio; at
  28×28 grayscale the only distinguishing cues are a collar or button placket a few pixels wide,
  which survive neither the downsampling nor the upsample to 224.
- **Pullover ↔ Coat** — both are long-sleeved outer torso garments with an identical T-shaped
  silhouette. A coat's open front is a thin vertical intensity seam; a pullover is a solid
  block. That difference is one or two pixels of gradient in the original resolution.
- **Shirt ↔ Coat / Shirt ↔ Pullover** — Shirt is the "confusable middle" of the four torso
  classes; its errors spray across all three neighbours.
- **Sandal ↔ Sneaker** — both are low-profile footwear with a flat sole and low ankle line; the
  discriminating feature is the strap gaps in a sandal, which are thin dark regions easily lost.

Write the paragraph in terms of **silhouette and what survives 28×28**, exactly as above. That
framing is what the criterion means by "visually plausible."

The trainer's line, worth echoing in the report: *an error is not a wrong answer — it is the
model telling you where the visual signal genuinely is ambiguous.*

---

## Task 7 — Save the artifact (`model_io.py`)

Save the **assembled** model (backbone + head) so loading is unambiguous:

```python
torch.save({
    "arch": "resnet18",
    "input_size": 224,
    "classes": CLASSES,
    "head_dims": [512, 256, 10],
    "finetuned": bool,                 # whether layer4 was unfrozen
    "state_dict": model.state_dict(),  # full assembled model
    "test_accuracy": float,
}, "models/product_classifier.pt")
```

`model_io.py` exposes exactly two public functions — **this file is the documented snippet the
README shows and the file Part 3 imports:**

```python
def load_model(path: Path = MODEL_PATH, device: str | None = None) -> tuple[nn.Module, list[str]]:
    """Rebuild resnet18 + head from the checkpoint config, load weights, .eval(), return (model, classes)."""

def predict_image(image_path: str | Path, model=None, classes=None) -> dict:
    """Open a PNG with PIL, apply the SAME eval transform, forward, softmax.
    Returns {"label": str, "confidence": float, "class_index": int, "all_probs": {...}}"""
```

Two rules:
- The eval transform lives in **one** place (`config.py` / `data.py`) and both training and
  `predict_image` import it. A transform mismatch between train and inference is the classic
  silent bug here.
- `predict_image` lazily loads and caches the model at module level so the agent doesn't reload
  45 MB on every tool call.

**Checkpoint size:** ResNet-18 state dict ≈ 45 MB. Fine for GitHub (limit is 100 MB/file). If it
somehow exceeds that, save only `head` weights + `finetuned=False` and have `load_model()`
rebuild the frozen pretrained backbone from torchvision — document that in the snippet.

---

## Task 8 — Export real sample PNGs (`export_samples.py`)

The brief is explicit: torchvision stores raw IDX binary; Part 3's
`classify_product_image(image_path)` needs **actual image files**.

Export **10** images — one per class — from the **test split**:

```python
img_array = test_set.data[idx].numpy()          # raw uint8 28×28, NOT the transformed tensor
PIL.Image.fromarray(img_array).save(f"data/sample_images/{label_idx:02d}_{slug}.png")
```

Filenames make the true label obvious:

```
data/sample_images/00_tshirt_top.png
data/sample_images/01_trouser.png
data/sample_images/02_pullover.png
data/sample_images/03_dress.png
data/sample_images/04_coat.png
data/sample_images/05_sandal.png
data/sample_images/06_shirt.png
data/sample_images/07_sneaker.png
data/sample_images/08_bag.png
data/sample_images/09_ankle_boot.png
```

Also write `data/sample_images/labels.json` mapping filename → `{true_label, class_index,
test_split_index}`. Commit all of it.

Pick the **first test-split index for each class** (deterministic, no randomness) so a rerun
produces byte-identical files.

Then, as a self-check the script prints and the report records: run `predict_image` on all 10
and show predicted vs true. Expect 9–10 correct; if one is wrong, keep it — an honest miss on a
Shirt is better evidence the tool is real than a curated 10/10.

---

## Part 2 acceptance self-check

- [ ] Fashion-MNIST from the pinned source, no substitute
- [ ] exact train/val/test sizes reported (55,000 / 5,000 / 10,000); test untouched until final
- [ ] input size documented (224), grayscale→3ch, ImageNet mean/std normalisation
- [ ] backbone frozen, new 10-class head, Adam, batch/lr/epochs documented
- [ ] explicit statement: feature extraction sufficient **or** fine-tuning required, with
      before/after val accuracy either way
- [ ] test accuracy ≥ 80% (or an honest shortfall + attempted fine-tune + diagnosis)
- [ ] real 10×10 confusion matrix from real predictions + per-class precision/recall
- [ ] ≥2 confusion pairs read off the matrix, each with a visual-similarity paragraph
- [ ] `models/product_classifier.pt` exists and loads via the documented `model_io` snippet —
      and Part 3's tool calls that same snippet
- [ ] `data/sample_images/` has ≥5 real PNGs from the test split; Part 3 points at these files

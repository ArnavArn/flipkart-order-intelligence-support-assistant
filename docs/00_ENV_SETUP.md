# Phase 0 — Environment Setup

## Why not the default python3

Your default `python3` is **3.14.6**. PyTorch, faiss-cpu and chromadb do not publish wheels for
3.14. You already have **Python 3.12** via Homebrew at `/opt/homebrew/bin/python3.12` — use it.

Machine: Apple Silicon (arm64), 12 cores, 24 GB RAM. Torch will use the **MPS** backend for
Part 2; CPU fallback is fine because we cache features.

## Commands

```bash
cd /Users/arnav.garg/Developer/projects/capstone

/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import numpy, pandas, sklearn, torch, torchvision, faiss, sentence_transformers, langgraph; print('env ok', torch.__version__, torch.backends.mps.is_available())"
```

Expect ~2.5 GB of downloads (torch + torchvision are the bulk). One-time.

## requirements.txt

Pinned to majors that are known-good together on Python 3.12 / arm64:

```
numpy>=1.26,<3
pandas>=2.2
scikit-learn>=1.5
joblib>=1.4
matplotlib>=3.8
torch>=2.4
torchvision>=0.19
pillow>=10.3
sentence-transformers>=3.0
faiss-cpu>=1.8
langgraph>=0.2
langchain-core>=0.3
```

Notes:
- `shap` is **optional** — the brief allows `.feature_importances_` instead. Skip it; one less
  dependency that can break.
- `chromadb` is not installed. We use **FAISS**, which the brief allows and which is lighter.
- `langchain-core` is needed for `@tool` / message types that `langgraph` expects. No
  `langchain-openai`, no `langchain-community` — we are not calling a hosted LLM.

## One-time network downloads (all free, no account, no API key)

| What | When | Cached where |
|---|---|---|
| Fashion-MNIST IDX files (~30 MB) | Part 2 first run | `data/fashion_mnist/` (gitignored) |
| ResNet-18 ImageNet weights (~45 MB) | Part 2 first run | `~/.cache/torch/hub/` |
| `all-MiniLM-L6-v2` (~90 MB) | Part 3 index build | `~/.cache/huggingface/` |

**Important for the "zero outbound network calls" criterion:** that criterion is about
MOCK_LLM *inference* — no LLM API calls at agent run time. The sentence-transformer weights are
downloaded once at **index-build** time and cached locally; the README must say this plainly.
The FAISS index itself is persisted to `part3_agent/index/` and committed, so a grader can run
the agent without rebuilding.

## .gitignore

```
.venv/
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/

# raw dataset + derived caches (regenerable, large)
data/fashion_mnist/
part2_image_classifier/cache/

# class materials — not part of the deliverable
*.rtf
```

**Do NOT gitignore:** `models/`, `data/sample_images/`, `orders_dataset.csv`,
`part3_agent/index/`, `transcripts/`, `*/reports/`.

## Size sanity check before pushing

```bash
du -sh models data/sample_images orders_dataset.csv part3_agent/index
```

Expected: `models/` ≈ 50–60 MB (ResNet-18 state dict dominates), everything else < 5 MB. Well
under GitHub's 100 MB per-file limit — no Git LFS needed. If `models/product_classifier.pt`
somehow exceeds 100 MB, save only the head + a flag to rebuild the frozen backbone from
torchvision, and document that in the loader snippet.

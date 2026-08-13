# Build Order — step-by-step execution script

This is the checklist for the coding session. Work top to bottom. Each step names its spec doc
and its done-condition. **Do not start a step until the previous step's done-condition holds.**

---

## Step 0 — Environment  ·  spec: `docs/00_ENV_SETUP.md`

1. Write `requirements.txt` and `.gitignore` (contents are in the spec).
2. `/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `git init -b main`, first commit.

**Done when:** the import smoke-test line from the spec prints `env ok <torch version> True`.

---

## Step 1 — Part 1 data  ·  spec: `docs/02_PART1_PLAN.md` Tasks 1–2

1. Save `generate_orders.py` **verbatim** from the brief. Do not edit a single character.
2. `python generate_orders.py`
3. Write `part1_return_risk/config.py` and `data_checks.py`.

**Done when:** `orders_dataset.csv` is 6000×13, printed return rate ∈ [0.18, 0.27], and
`reports/01_data_checks.md` contains the MAR paragraph with the real COD/non-COD gap in pp.

---

## Step 2 — Part 1 models  ·  spec: `docs/02_PART1_PLAN.md` Tasks 3–6

1. `pipeline.py` — the ColumnTransformer. `OneHotEncoder(handle_unknown="ignore")`, **no drop**.
2. `thresholds.py` — `sweep_threshold()`, written **once**, used twice.
3. `models.py` + `train.py` — Dummy → LogReg (+ sweep) → RF GridSearchCV.

**Done when:** Dummy F1(class 1) = 0.0; LogReg test ROC-AUC ≥ 0.58 and F1 ≥ 0.30; RF best CV
ROC-AUC ≥ 0.58 with |test − CV| ≤ 0.05. If any gate fails, stop and diagnose — do not proceed
and do not adjust the reported number.

---

## Step 3 — Part 1 analysis + artifact  ·  spec: `docs/02_PART1_PLAN.md` Tasks 7–9

1. `explain.py` — impurity top-5, then `permutation_importance` on the test split, side by side.
2. `subgroups.py` — recall/precision by `product_category` and by `payment_method` at `t*_rf`.
3. In `train.py`: compute `t*_rf` by calling the **same** `sweep_threshold()` on the RF's
   `predict_proba`; `joblib.dump` the pipeline; write `models/return_risk_meta.json`; reload and
   assert the probability round-trips to 1e-9.

**Done when:** all 7 reports exist, `payment_method` (one-hot, or grouped with both tables
shown) is in the top 5, `delivery_distance_km` is named as collapsing under permutation, and
`models/return_risk_meta.json` has a real `t_star_rf`.

> **Part 1 is now frozen.** Part 3 depends on this pickle and this JSON. Do not retrain later.

---

## Step 4 — Part 2 features + head  ·  spec: `docs/03_PART2_PLAN.md` Tasks 1–4

1. `config.py`, `data.py` — download, stratified 55k/5k/10k, transforms (224, 3ch, ImageNet norm).
2. `features.py` — **cache the frozen ResNet-18 features to `.npy`**. This is the step that
   turns hours into minutes. Verify the cache files exist before training the head.
3. `train.py` — head on cached features, Adam lr 1e-3, batch 256, 20 epochs. Then the branch:
   val ≥ 0.80 → record "feature extraction sufficient"; val < 0.80 → unfreeze `layer4`, lr 1e-4,
   retrain, record before/after.

**Done when:** `reports/02_training_log.md` states explicitly which branch was taken, with both
numbers.

---

## Step 5 — Part 2 evaluation + artifacts  ·  spec: `docs/03_PART2_PLAN.md` Tasks 5–8

1. `evaluate.py` — **one** run on the untouched test split: accuracy, 10×10 confusion matrix,
   per-class precision/recall. Extract the largest off-diagonal cells programmatically.
2. `reports/05_confusion_analysis.md` — one paragraph per confused pair, written from the
   matrix you actually got.
3. `model_io.py` — `load_model()` + `predict_image()`. Shared eval transform with training.
4. `export_samples.py` — 10 real PNGs from the test split + `labels.json`; then run
   `predict_image` over all 10 and record predicted vs true.

**Done when:** test accuracy ≥ 80%, `models/product_classifier.pt` loads via `model_io`, and
`data/sample_images/` has 10 committed PNGs.

> **Part 2 is now frozen.**

---

## Step 6 — Part 3 knowledge base + index  ·  spec: `docs/04_PART3_PLAN.md` Tasks 1–2

1. Author 15 policy docs in `part3_agent/kb/documents/`.
2. `chunking.py` — sentence-wise, every chunk keeps `doc_id`.
3. `kb/eval_queries.json` — 6 queries with document-level relevance.
4. `index_build.py` — MiniLM, `normalize_embeddings=True`, FAISS `IndexFlatIP`, persist to
   `part3_agent/index/`, print the **threshold calibration table** (6 in-scope + 3 out-of-scope).
5. Set `SIM_THRESHOLD` in `config.py` from that measured table.

**Done when:** the calibration table shows a clean gap, and `part3_agent/index/` is committed.

---

## Step 7 — Part 3 tools  ·  spec: `docs/04_PART3_PLAN.md` Tasks 3–4

1. `tools/return_risk_tool.py` — reads `t_star_rf` from the JSON, **never a literal**.
2. `tools/image_tool.py` — imports `part2_image_classifier.model_io.predict_image`.

**Done when:** calling each tool from a Python REPL returns a sensible dict, and
`check_return_risk`'s probability matches a direct `joblib.load(...).predict_proba(...)` call on
the same row exactly.

---

## Step 8 — Part 3 graph  ·  spec: `docs/04_PART3_PLAN.md` Tasks 5–8

Build in this order — each is independently testable:

1. `state.py` — `AgentState`.
2. `guardrails.py` — `check_input()`, `check_groundedness()`.
3. `prompts.py` — 4S-annotated system prompt + 3 few-shot intent examples.
4. `mock_llm.py` — `compose()`, all five branches.
5. `graph.py` — 6 nodes, 2 conditional edges, `MemorySaver` checkpointer.
6. `run_agent.py` — CLI, `--thread-id` flag.

**Done when:** `python -m part3_agent.run_agent` answers a policy question, a risk question, and
an image question correctly, and a `--thread-id` change visibly resets `last_order_id`.

---

## Step 9 — Part 3 transcripts + eval  ·  spec: `docs/04_PART3_PLAN.md` Tasks 9–10

1. `run_transcripts.py` — all 9 files, each printing the node path, the few-shot match line, and
   scores.
2. `eval_retrieval.py` — P@3 / R@3, document-level, per-query arithmetic, plus the calibration
   table appended.

**Done when:** 9 transcripts + `retrieval_eval.md` exist, run with `USE_LIVE_LLM` unset, and
transcripts 05 vs 06 show the same question producing different answers on different threads.

---

## Step 10 — README  ·  spec: `docs/05_GIT_AND_SUBMISSION.md`

Write it last. Every number copied from a generated report.

---

## Step 11 — Git history + push  ·  spec: `docs/05_GIT_AND_SUBMISSION.md`

Feature branch → 3 commits → `git merge --no-ff` → verify with
`git log --graph --all --oneline` → create the **public** repo on github.com/new → push both
branches.

---

## Step 12 — Final audit  ·  spec: `docs/06_ACCEPTANCE_CHECKLIST.md`

Walk all 30 acceptance criteria, ticking each against a file and line. Then do the fresh-clone
smoke test. Then submit the URL.

---

## Rules that apply at every step

- `random_state=42` / `seed=42`, always.
- Every reported number is written to a file **by the code that computed it**.
- If an acceptance gate fails, report the honest shortfall with a diagnosis. The brief says a
  fabricated number is never acceptable — an honest miss plus analysis scores; a fake number is
  the one unrecoverable error.
- No absolute paths. Resolve from `Path(__file__).resolve().parents[1]`.
- Parts 1 and 2 are frozen once their artifacts are saved. Part 3 consumes them; it never
  triggers a retrain.

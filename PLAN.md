# Flipkart Order Intelligence & Support Assistant — Master Build Plan

**One repo. Three parts. Part 3 is the product; Parts 1–2 are the tools it calls.**
Total: 100 marks (P1 = 35, P2 = 25, P3 = 40).

> This file is the map. Detailed, implementation-ready specs live in `docs/`.
> Execute in the order given in [`docs/07_BUILD_ORDER.md`](docs/07_BUILD_ORDER.md).

---

## 0. The one-paragraph version

A customer support agent asks one assistant three kinds of question — *"will this order be
returned?"*, *"what category is this product photo?"*, *"what's the return policy?"* — and a
single LangGraph agent routes each to the right capability: a saved Random Forest
(`models/return_risk_model.pkl`), a saved transfer-learned CNN (`models/product_classifier.pt`),
or a FAISS-indexed policy knowledge base. Nothing is hardcoded; the agent loads and calls the
real artifacts trained in Parts 1 and 2.

```
                       ┌──────────────────────────────┐
  customer question ──▶│   LangGraph support agent    │
                       │  (Part 3 — user-facing)      │
                       └──┬────────┬────────────┬─────┘
              policy intent│  risk intent│   image intent│
                       ┌───▼───┐  ┌───▼─────┐  ┌────▼──────┐
                       │  RAG  │  │ Part 1  │  │  Part 2   │
                       │ FAISS │  │ RF .pkl │  │ CNN .pt   │
                       │  KB   │  │  + t*_rf│  │ + samples │
                       └───────┘  └─────────┘  └───────────┘
                            └──────── grounded, structured JSON answer ───────▶
```

---

## 1. Deliverable

**Exactly one submission: a public GitHub repository URL.** No PDFs, no slides, no screenshots,
no uploads. Every artifact is a committed text/code/data file.

The repo must contain:

| Requirement | Where |
|---|---|
| `generate_orders.py` + `orders_dataset.csv` | repo root |
| Part 1 training/eval code | `part1_return_risk/` |
| `models/return_risk_model.pkl` | committed |
| Part 2 training/eval code + confusion matrix output | `part2_image_classifier/` + `reports/` |
| `models/product_classifier.pt` | committed |
| ≥5 real `.png` files from the test split | `data/sample_images/` |
| KB files, index build code, both tools, LangGraph agent | `part3_agent/` |
| 8+ transcripts | `transcripts/` |
| Retrieval eval numbers (P@3, R@3) | `transcripts/retrieval_eval.md` |
| One root `README.md` tying all three together | `README.md` |
| Feature branch, ≥2 commits, merged to main | git history |

---

## 2. The chain of custody (why this is one system, not three)

These are the **hard couplings**. Break one and Part 3 fails an acceptance criterion:

```
Part 1 ──▶ models/return_risk_model.pkl   ──▶ part3_agent/tools/return_risk_tool.py
       ──▶ models/return_risk_meta.json   ──▶   (t*_rf → risk bucket cut points)
              { "t_star_rf": 0.xx, ... }

Part 2 ──▶ models/product_classifier.pt   ──▶ part2_image_classifier/model_io.py
       ──▶ data/sample_images/*.png       ──▶ part3_agent/tools/image_tool.py
                                                (imports model_io — the SAME documented
                                                 loader snippet the README shows)
```

**Rule:** Part 3's tools never re-implement loading. `image_tool.py` imports
`part2_image_classifier.model_io`. `return_risk_tool.py` reads `t*_rf` from
`models/return_risk_meta.json` — it is never a literal in the tool code.

---

## 3. The graded traps

Read this list before writing any code. These are the specific things the brief and the
walkthrough session single out, and they are where marks are actually lost.

### Part 1
1. **`rating_given` missingness is MAR**, not MCAR, not MNAR. It depends on the *observed*
   `payment_method` column (COD ≈ 22% missing vs non-COD ≈ 6%). You must state the measured
   gap numerically as the evidence.
2. **DummyClassifier F1 for class 1 must come out 0.0**, and the paragraph must name the trap
   as **"high accuracy, zero recall."**
3. **0.5 is not magic.** Sweep the threshold 0.10 → 0.90 in steps of ≤ 0.02. The chosen
   threshold's recall must beat default recall by **≥ 15 percentage points**, and you must
   state the precision drop numerically.
4. **`delivery_distance_km` is a planted decoy.** Look at the generator: `z` does *not* contain
   it. Impurity importance will still rank it high (it is a noisy high-cardinality continuous
   column). Permutation importance on the test split will collapse it. Naming this is a
   required acceptance criterion.
5. **`t*_rf` is computed on the Random Forest's own `predict_proba`** — not reused from the
   Logistic Regression. The saved artifact is the tuned **Random Forest** pipeline.
6. **Subgroup analysis** by `product_category` *and* `payment_method`, with a genuinely weak
   subgroup named and a **specific** fix proposed (a category-specific threshold, a new
   feature — never "collect more data").

### Part 2
7. **Cache the frozen backbone's features.** One forward pass over 70k images, then train the
   head on cached 512-d vectors. This turns hours into minutes. Do not skip it.
8. **The test split stays untouched** until the single final evaluation.
9. **State explicitly** whether feature extraction alone cleared 80%, or fine-tuning was
   required — with before/after validation accuracy **either way**.
10. **Confusion pairs are read off your own matrix**, not guessed. One paragraph per pair on
    the *visual silhouette* similarity.
11. **Export ≥5 real `.png` files** from the test split. The raw IDX binary is not a submission.

### Part 3
12. **Sentence-wise chunking**, with a chunk → parent-document map. P@3/R@3 are scored at the
    **document** level after dedup.
13. **Risk buckets anchor to `t*_rf`**, not to a fixed 0.3/0.6 split.
14. **≥4 nodes, ≥1 conditional edge.** The graph must actually branch by intent.
15. **State vs memory.** One transcript shows a follow-up ("what about *its* delivery?")
    resolving an order ID from two turns earlier. A *separate* transcript shows a fresh
    conversation where that state is correctly **absent**.
16. **System prompt annotated line-by-line against 4S** (Specific, Short, Surround, Single) +
    role prompting. **2 few-shot intent examples** must visibly drive routing in ≥2 transcripts.
17. **MOCK_LLM is the default and is what every graded transcript runs against.** Zero API
    keys, zero outbound calls at run time.
18. **Both guardrails**, each with its own transcript: input-side prompt-injection block, and
    output-side groundedness refusal that **prints the similarity score against the threshold**.

### Repo-wide
19. **Git history** must show a feature branch created, committed to ≥2 times, and merged
    into main. Checked once across the whole repo.

---

## 4. Phases

| Phase | Output | Doc |
|---|---|---|
| **0. Environment** | Python 3.12 venv, `requirements.txt`, `.gitignore`, `git init` | [`docs/00_ENV_SETUP.md`](docs/00_ENV_SETUP.md) |
| **1. Part 1** | `orders_dataset.csv`, `models/return_risk_model.pkl`, `models/return_risk_meta.json`, `part1_return_risk/reports/*.md` | [`docs/02_PART1_PLAN.md`](docs/02_PART1_PLAN.md) |
| **2. Part 2** | `models/product_classifier.pt`, `data/sample_images/*.png`, `part2_image_classifier/reports/*.md` | [`docs/03_PART2_PLAN.md`](docs/03_PART2_PLAN.md) |
| **3. Part 3** | KB + FAISS index, both tools, LangGraph agent, `transcripts/` ×9, retrieval eval | [`docs/04_PART3_PLAN.md`](docs/04_PART3_PLAN.md) |
| **4. README** | Root `README.md` tying it together with run instructions + example transcript | [`docs/05_GIT_AND_SUBMISSION.md`](docs/05_GIT_AND_SUBMISSION.md) |
| **5. Git + push** | feature branch → 2 commits → merge → public GitHub repo | [`docs/05_GIT_AND_SUBMISSION.md`](docs/05_GIT_AND_SUBMISSION.md) |
| **6. Final audit** | Every acceptance criterion ticked with evidence | [`docs/06_ACCEPTANCE_CHECKLIST.md`](docs/06_ACCEPTANCE_CHECKLIST.md) |

Step-by-step execution order for the coding session: [`docs/07_BUILD_ORDER.md`](docs/07_BUILD_ORDER.md).
Full file-by-file map: [`docs/01_REPO_MAP.md`](docs/01_REPO_MAP.md).

---

## 5. Non-negotiable conventions

- **`random_state=42` / `seed=42` everywhere.** Splits, models, GridSearch, permutation
  importance, torch. Reruns must reproduce reported numbers.
- **Every reported number is written to a file by the code that computed it**, into
  `part1_return_risk/reports/` or `part2_image_classifier/reports/` as markdown. Never type a
  number into the README by hand — copy it from a generated report, or have the README link to
  the report. A fabricated number is the one thing the brief says is never acceptable.
- **No absolute paths.** Everything resolves from a repo-root constant
  (`Path(__file__).resolve().parents[1]`).
- **MOCK_LLM default.** `USE_LIVE_LLM` unset must satisfy 100% of acceptance criteria.
- **Nothing large committed** except the two model artifacts, `orders_dataset.csv`, and the
  sample PNGs. Raw Fashion-MNIST IDX and cached feature `.npy` files are gitignored.

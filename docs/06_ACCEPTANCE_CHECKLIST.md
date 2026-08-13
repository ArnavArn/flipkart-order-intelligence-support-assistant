# Final Audit — every acceptance criterion, verbatim, mapped to evidence

Walk this at Step 12. Tick a box **only** after opening the named file and seeing the evidence.
"It should be there" is not a tick.

---

## Submission-level (4)

| # | Criterion | Evidence location | ✓ |
|---|---|---|---|
| S1 | One public GitHub repo URL, nothing else submitted | logged-out browser loads the repo | ☐ |
| S2 | README documents: regenerate Part 1 data+model, run Part 2 train/eval, run Part 3 in mock mode | `README.md` §3, §4, §5 | ☐ |
| S3 | README contains ≥1 full example agent transcript | `README.md` §9 (transcript 05 inlined) | ☐ |
| S4 | Commit history: feature branch, ≥2 commits on it, merged into main | `git log --graph --all --oneline` | ☐ |

---

## Part 1 — Return-Risk Scoring (11)

| # | Criterion | Evidence | ✓ |
|---|---|---|---|
| 1.1 | `orders_dataset.csv` has exactly 6,000 rows and 13 columns | `reports/01_data_checks.md` | ☐ |
| 1.2 | Overall return rate ∈ [18%, 27%] | `reports/01_data_checks.md` | ☐ |
| 1.3 | `rating_given` missing on 8–18% of rows | `reports/01_data_checks.md` | ☐ |
| 1.4 | Missingness named **MAR**, conditional on observed `payment_method`, **with the measured COD vs non-COD gap in pp** as evidence; MCAR and MNAR each explicitly ruled out | `reports/01_data_checks.md` — MAR paragraph | ☐ |
| 1.5 | Dummy `F1(returned=1)` reported as **0.0**; paragraph names **"high accuracy, zero recall"** | `reports/02_baseline_and_logreg.md` | ☐ |
| 1.6 | LogReg @0.5: test **ROC-AUC ≥ 0.58** and **F1(class 1) ≥ 0.30** | `reports/02_baseline_and_logreg.md` | ☐ |
| 1.7 | Threshold swept 0.1→0.9, step ≤ 0.02; chosen t* recall **≥ +15 pp** vs default; precision drop stated **numerically** | `reports/03_threshold_sweep_logreg.md` + `.csv` | ☐ |
| 1.8 | RF GridSearch best **CV ROC-AUC ≥ 0.58**; **|test − CV| ≤ 0.05** | `reports/04_random_forest_gridsearch.md` | ☐ |
| 1.9 | Top-5 impurity importances include `payment_method` (one-hot) **and** ≥2 of {`price_inr`, `customer_tenure_days`, `discount_pct`, `num_previous_returns`}; permutation comparison present; ≥1 top-5 feature named as collapsing; one sentence on why impurity overrates a noisy continuous column | `reports/05_feature_importance.md` | ☐ |
| 1.10 | Subgroup tables for **both** `product_category` and `payment_method`, arithmetic correct, a genuinely weaker subgroup named with a **specific** fix (not "collect more data") | `reports/06_subgroup_analysis.md` | ☐ |
| 1.11 | `models/return_risk_model.pkl` exists, `joblib.load`s, **is the tuned RF** (not LogReg), its `predict_proba` is what Part 3 calls; **`t*_rf` reported**, computed on this model's own probabilities | `reports/07_final_artifact.md` + `models/return_risk_meta.json` | ☐ |

> **Highest-risk item: 1.9.** One-hot dilution can push `payment_method_COD` out of the raw
> top 5. Mitigation is in `docs/02_PART1_PLAN.md` Task 7a — report the raw table **and** a
> grouped-by-parent-feature table, stating which ranking each claim is made against.

---

## Part 2 — Image Categoriser (7)

| # | Criterion | Evidence | ✓ |
|---|---|---|---|
| 2.1 | Fashion-MNIST from the pinned source, no substitute dataset | `reports/01_splits_and_setup.md` | ☐ |
| 2.2 | Exact train/val/test sizes reported (55,000 / 5,000 / 10,000); **test untouched** until final eval | `reports/01_splits_and_setup.md` | ☐ |
| 2.3 | States explicitly whether feature extraction alone sufficed **or** fine-tuning was required, with **before/after val accuracy either way** | `reports/02_training_log.md` | ☐ |
| 2.4 | **≥80% test accuracy** (or an honest shortfall + attempted fine-tune + confusion diagnosis) | `reports/03_test_evaluation.md` | ☐ |
| 2.5 | Confusion matrix from **real** predictions; ≥2 real pairs named with a visual-similarity explanation | `reports/04_confusion_matrix.md`, `reports/05_confusion_analysis.md` | ☐ |
| 2.6 | `models/product_classifier.pt` exists, loadable by the **documented snippet**, and that snippet is what Part 3's tool calls | `part2_image_classifier/model_io.py` ← imported by `part3_agent/tools/image_tool.py` | ☐ |
| 2.7 | `data/sample_images/` has ≥5 real `.png` files exported from the test split; Part 3's tool points at these exact files | `data/sample_images/` (10 PNGs + `labels.json`), transcript 04 | ☐ |

Also documented (task requirements, folded into the reports): input size 224 stated explicitly,
grayscale→3-channel replication, ImageNet mean/std, batch size, Adam, learning rate, epoch count.

---

## Part 3 — Support Agent (10)

| # | Criterion | Evidence | ✓ |
|---|---|---|---|
| 3.1 | ≥12 chunked documents; embeddings from a free local model; index is FAISS or ChromaDB (not account-gated) | `kb/documents/` (15 docs), `index_build.py`, `part3_agent/index/` | ☐ |
| 3.2 | Both tools are **real** calls loading Part 1's and Part 2's actual saved artifacts; spot-checking one tool against running the saved model directly gives the **same number**; `classify_product_image` points at real PNGs | transcript 03 (spot-check block), transcript 04 | ☐ |
| 3.3 | Risk buckets anchored to **`t*_rf`** (not 0.3/0.6, not the LogReg threshold); one-sentence justification states **both** the cut points and the `t*_rf` value | `tools/return_risk_tool.py`, `README.md` §7 | ☐ |
| 3.4 | Graph has **≥4 nodes** and **≥1 conditional edge**; multi-turn transcript shows state carried; a **separate** fresh-conversation transcript shows it correctly absent/reset | `graph.py` (6 nodes, 2 conditional edges), transcripts 05 and 06 | ☐ |
| 3.5 | System prompt annotated against **each** of Specific / Short / Surround / Single **plus role prompting**; ≥2 few-shot intent examples **visibly driving correct routing on ≥2 transcripts** | `prompts.py`, `README.md` §8, transcripts 03, 04, 09 | ☐ |
| 3.6 | MOCK_LLM requires **zero API keys and zero outbound network calls**; all 9 transcripts run in this mode | `config.py` (`MOCK_LLM` default True), transcript headers | ☐ |
| 3.7 | ≥1 transcript is the prompt-injection attempt, **visibly deflected** — agent does not comply | transcript 07 (matched pattern printed) | ☐ |
| 3.8 | ≥1 transcript is the ungrounded question; output-side check **visibly refuses**, printing the retrieved chunk's **similarity score against the stated threshold** | transcript 08 | ☐ |
| 3.9 | P@3 and R@3 at **document level** (chunks mapped to parent docs and **deduplicated**), ≥5 queries, **per-query arithmetic visible** | `transcripts/retrieval_eval.md` | ☐ |
| 3.10 | If the optional live-LLM extension exists, README marks it optional and removing the key still satisfies every criterion via MOCK_LLM | `README.md` §11 | ☐ |

---

## Cross-part integrity checks (do these last)

- [ ] `grep -rn "0\.3\|0\.6" part3_agent/tools/return_risk_tool.py` → **no hardcoded cut points**
- [ ] `grep -rn "t_star\|0\.4[0-9]" part3_agent/` → `t*_rf` only ever arrives from
      `models/return_risk_meta.json`, never as a literal
- [ ] `grep -rn "import" part3_agent/tools/image_tool.py` → imports
      `part2_image_classifier.model_io`, does **not** re-implement loading
- [ ] `grep -rn "/Users/" .` → **zero** absolute paths anywhere in committed code
- [ ] `grep -rln "api_key\|API_KEY\|sk-" part3_agent/` → nothing outside the optional, clearly
      marked live-LLM block
- [ ] Every number in the README appears in some `reports/` or `transcripts/` file
- [ ] Fresh-clone smoke test passes (`docs/05_GIT_AND_SUBMISSION.md`)

---

## Score map — where the 100 marks sit

| Part | Marks | Heaviest sub-items |
|---|---|---|
| 1 | 35 | MAR justification, threshold sweep + trade-off, permutation-vs-impurity, subgroup fix, `t*_rf` artifact |
| 2 | 25 | ≥80% accuracy, explicit FE-vs-fine-tune statement, real confusion-pair analysis, loadable artifact + PNGs |
| 3 | 40 | Real tool calls, `t*_rf`-anchored buckets, state vs memory pair, both guardrail transcripts, doc-level P@3/R@3 |

Part 3 is the largest single block **and** it depends on Parts 1 and 2 shipping clean artifacts.
Budget time accordingly: Parts 1 and 2 are mechanical; Part 3 is where the marks and the risk
concentrate.

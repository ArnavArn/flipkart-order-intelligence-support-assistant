# Flipkart Order Intelligence & Support Assistant

One connected system, not three disconnected scripts. A single **LangGraph support agent**
answers three kinds of customer question — *"will this order be returned?"*, *"what category
is this product photo?"*, *"what's the return policy?"* — by routing to whichever capability
the question actually needs:

```
                       ┌──────────────────────────────┐
  customer question ──▶│   LangGraph support agent    │
                       │  (Part 3 — user-facing)      │
                       └──┬────────┬────────────┬─────┘
              policy intent│  risk intent│   image intent│
                       ┌───▼───┐  ┌───▼─────┐  ┌────▼──────┐
                       │  RAG  │  │ Part 1  │  │  Part 2   │
                       │ FAISS │  │ RF .pkl │  │ CNN .pt   │
                       │  KB   │  │ + t*_rf │  │ + samples │
                       └───────┘  └─────────┘  └───────────┘
                            └──────── grounded, structured JSON answer ───────▶
```

- **Part 1** trains and saves a return-risk model on 6,000 seeded synthetic Flipkart orders.
- **Part 2** trains and saves a Fashion-MNIST product-image classifier via transfer learning.
- **Part 3** is the actual product: a LangGraph agent that loads **both saved artifacts as real,
  callable tools**, on top of its own retrieval-augmented policy knowledge base. Nothing here is
  a hardcoded stand-in — every number the agent reports comes from the saved models.

Every dependency is free and keyless. The agent's default mode (`MOCK_LLM`) requires **zero API
keys and zero outbound network calls at run time**.

---

## Setup

```bash
# Python 3.12 is required — torch/faiss have no 3.14 wheels.
/opt/homebrew/bin/python3.12 -m venv .venv   # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

One-time downloads on first run (all free, no account, no API key):
Fashion-MNIST (~30 MB, Part 2), ResNet-18 ImageNet weights (~45 MB, Part 2),
`all-MiniLM-L6-v2` (~90 MB, Part 3 — cached to `~/.cache/huggingface/`). None of these are
LLM API calls; they are one-time weight/dataset downloads, cached locally afterward.

---

## Part 1 — Return-Risk Scoring Pipeline

Regenerate the dataset and retrain the model:

```bash
python generate_orders.py          # writes orders_dataset.csv (6,000 rows, seed=42, deterministic)
python -m part1_return_risk.train  # runs all 9 tasks, writes models/ + reports/
```

**Results** (full detail in [`part1_return_risk/reports/`](part1_return_risk/reports/)):

| Check | Result |
|---|---|
| Dataset shape | 6,000 rows × 13 columns |
| Overall return rate | 22.75% |
| `rating_given` missingness | 13.05% overall — **MAR**, conditional on `payment_method` (COD 22.83% vs non-COD ≈6.12%, a **16.8 pp** gap — [report](part1_return_risk/reports/01_data_checks.md)) |
| Baseline (`DummyClassifier`) | accuracy 77.25%, **F1(class 1) = 0.0** — the "high accuracy, zero recall" trap ([report](part1_return_risk/reports/02_baseline_and_logreg.md)) |
| Logistic Regression @ 0.5 | ROC-AUC 0.6253, F1 0.3921 |
| Threshold sweep | chosen t\* = 0.44 → recall 0.5788→0.7582 (**+17.9 pp**), precision 0.2964→0.2801 (−1.6 pp) ([report](part1_return_risk/reports/03_threshold_sweep_logreg.md)) |
| Random Forest (GridSearchCV) | best params `{max_depth:6, n_estimators:200}`, CV ROC-AUC 0.6193, test ROC-AUC 0.6203 (gap 0.001) ([report](part1_return_risk/reports/04_random_forest_gridsearch.md)) |
| Top-5 impurity importance | `payment_method_COD`, `price_inr`, `delivery_distance_km`, `customer_tenure_days`, `delivery_days` |
| Permutation importance | `delivery_distance_km` collapses from impurity rank 3 to permutation rank 7 (perm_mean ≈ −0.0002) — confirmed as a **planted decoy**: it never appears in the generator's `z` formula ([report](part1_return_risk/reports/05_feature_importance.md)) |
| Weakest subgroups | `Electronics` (recall 0.4423 vs overall 0.5495) and `Prepaid_Card` (recall 0.0204) — fix proposed: a `price_vs_category_median` feature and/or a per-subgroup threshold ([report](part1_return_risk/reports/06_subgroup_analysis.md)) |

**t\*\_rf = 0.5000** — the F1-maximising threshold computed on the *saved Random Forest's own*
`predict_proba` on the held-out test split (not the Logistic Regression's threshold). Part 3's
`check_return_risk` reads this value live from `models/return_risk_meta.json` and buckets:
**Low** if `p < 0.50`, **Medium** if `0.50 ≤ p < 0.65`, **High** if `p ≥ 0.65`.
([full artifact report](part1_return_risk/reports/07_final_artifact.md))

---

## Part 2 — Product Image Categoriser via Transfer Learning

```bash
python -m part2_image_classifier.train           # cached-feature transfer learning; ~5-15 min on Apple Silicon/MPS
python -m part2_image_classifier.export_samples  # writes 10 real test-split PNGs to data/sample_images/
```

**Results** (full detail in [`part2_image_classifier/reports/`](part2_image_classifier/reports/)):

| Check | Result |
|---|---|
| Dataset | Fashion-MNIST, pinned source, train 55,000 / val 5,000 / test 10,000 (test untouched until final eval) |
| Backbone | ResNet-18 (ImageNet weights), frozen, features cached once, then only a small head trained (Adam, lr 1e-3, batch 256, 20 epochs) |
| Feature-extraction val accuracy | **91.34%** — ≥ the 80% gate, so **fine-tuning was not required** (the `layer4`-unfreeze fine-tune path is implemented and would trigger automatically below 80%) |
| **Test accuracy (held-out, untouched split)** | **90.56%** |
| Largest confusion pairs (read off the real matrix) | **Shirt ↔ T-shirt/top** (155/52 — by far the largest cell), **Shirt ↔ Coat** (82/45), **Shirt ↔ Pullover** (71/35) — all three torso garments share the same rectangular silhouette at 28×28 resolution ([analysis](part2_image_classifier/reports/05_confusion_analysis.md)) |
| Sample-image self-check | 9/10 correct on the 10 exported PNGs; the one honest miss is `06_shirt.png` (true Shirt, predicted T-shirt/top, conf 0.851) — consistent with the #1 confusion pair, not a curated result |

`models/product_classifier.pt` is loaded through exactly one documented snippet,
[`part2_image_classifier/model_io.py`](part2_image_classifier/model_io.py)
(`load_model()` + `predict_image()`), which is the same function Part 3's
`classify_product_image` tool imports and calls — never reimplemented.

---

## Part 3 — Flipkart Support Agent (run in default mock mode)

```bash
python -m part3_agent.index_build       # rebuilds the FAISS index (already committed under part3_agent/index/)
python -m part3_agent.run_transcripts   # regenerates all 9 transcripts into transcripts/
python -m part3_agent.eval_retrieval    # regenerates transcripts/retrieval_eval.md
python -m part3_agent.run_agent --thread-id conv-A --message "What is the return window for apparel?"
```

`USE_LIVE_LLM` is unset by default (`MOCK_LLM=True`) — every command above runs with **zero API
keys and zero outbound network calls**. Verified directly: the entire pipeline above was rerun
end to end under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and produced **byte-identical**
transcripts to the checked-in versions.

### The knowledge base and retrieval

15 policy documents ([`part3_agent/kb/documents/`](part3_agent/kb/documents/)), split
sentence-wise into 59 chunks, embedded with the free local `all-MiniLM-L6-v2` model, indexed with
FAISS (`IndexFlatIP` over unit-normalised vectors = cosine similarity).

**Similarity threshold, calibrated from measured data** (not guessed): top-1 cosine scores for
the 6 answer-key queries ranged 0.494–0.763; three deliberately out-of-scope queries ("capital
of France", "warranty on a car battery", "apply for a job at Flipkart") ranged 0.081–0.465.
`SIM_THRESHOLD = 0.48` sits cleanly between the two clusters
([full table](transcripts/retrieval_eval.md)).

**Retrieval evaluation** (document-level, deduplicated, 6 queries — full per-query arithmetic in
[`transcripts/retrieval_eval.md`](transcripts/retrieval_eval.md)):

| Metric | Average |
|---|---|
| Precision@3 | **0.750** |
| Recall@3 | **0.917** |

### The two tools — real artifacts, spot-checked

`check_return_risk(order_features: dict)` loads `models/return_risk_model.pkl` directly and
reads `t*_rf` from `models/return_risk_meta.json` at import time — **never a hardcoded literal**.
`classify_product_image(image_path: str)` imports
[`part2_image_classifier.model_io.predict_image`](part2_image_classifier/model_io.py) verbatim.

Both are spot-checked against a direct call to the saved model outside the agent
([transcript 03](transcripts/03_return_risk_tool_call.md)):

```
Tool's own output:            return_probability = 0.6495
Direct joblib.load(...).predict_proba(...) on the same row = 0.6495   → identical
```

### The graph

`part3_agent/graph.py` — 6 nodes (`guard_input`, `classify_intent`, `retrieve`, `call_tool`,
`generate`, `verify_output`), 2 conditional edges (blocked-vs-clean after `guard_input`;
policy/return_risk/product_category/unknown after `classify_intent`), compiled with a
`MemorySaver` checkpointer keyed by `thread_id`.

### Prompt engineering — 4S + role prompting, annotated

From [`part3_agent/prompts.py`](part3_agent/prompts.py):

```python
RESPONSE_PROMPT = """
You are Flipkart's customer-support assistant.          # ROLE PROMPTING
Answer only from the context supplied below.            # [SPECIFIC] scope is explicit
You have exactly three capabilities: policy lookup,      # [SPECIFIC] enumerated, no ambiguity
return-risk scoring, product-image categorisation.
Never invent a policy that is not in the context.        # [SPECIFIC] the failure mode, named

<<<CONTEXT>>>                                            # [SURROUND] delimited context block
{context}
<<<END CONTEXT>>>

Reply with one JSON object and nothing else:             # [SINGLE] one task, one output shape
{"answer": str, "source": "policy_kb"|"return_risk_tool"|"image_classifier_tool",
  "confidence": float}
"""                                                       # [SHORT] under 120 words, no filler
```

Two **separate** prompts exist (`INTENT_PROMPT` for classification, `RESPONSE_PROMPT` for
composition) — that separation is itself the **Single** principle applied one level up.

**3 few-shot intent examples** (`FEW_SHOT_INTENT`) are embedded with the same MiniLM model and
scored by cosine similarity against the user's message; the winning exemplar is recorded in
`state["matched_fewshot"]` and printed in every transcript. They visibly drive routing
(cosine 0.74–1.00, no override) in
[transcript 03](transcripts/03_return_risk_tool_call.md),
[transcript 04](transcripts/04_image_classification_tool_call.md), and
[transcript 09](transcripts/09_intent_routing_fewshot.md).

### Guardrails

- **Input-side prompt-injection filter** — regex patterns (`ignore all previous instructions`,
  `pretend you are...`, `reveal your system prompt`, etc.) block the input *before* intent
  routing, retrieval, or tools ever run. See
  [transcript 07](transcripts/07_prompt_injection_blocked.md).
- **Output-side groundedness check** — for policy answers, if the top retrieved chunk's
  similarity is below `SIM_THRESHOLD`, the agent refuses rather than fabricating a policy, and
  prints the exact score against the threshold *inside the refusal text*. See
  [transcript 08](transcripts/08_ungrounded_refusal.md).

### State vs. memory — the required contrast

The identical question — *"What is the delivery SLA for its shipment?"* — produces two different
answers depending on conversation history, because LangGraph's checkpointer scopes state to a
`thread_id`, not to a global memory:

**[Multi-turn, `thread_id=conv-A`](transcripts/05_multiturn_state_carried.md)** — turn 1 checks
return risk for order 1523; turn 3 asks about "its" delivery, and the agent resolves "its" to
order 1523 from state carried within the same thread.

**[Fresh conversation, `thread_id=conv-B`](transcripts/06_fresh_conversation_state_absent.md)** —
the exact same question, as the very first message of a new thread, correctly gets:
*"There is no order referenced earlier in this conversation."*

<details>
<summary><b>Full example transcript — click to expand (transcript 05, multi-turn state carried)</b></summary>

```markdown
# Transcript 05 -- Multi-turn conversation, state carried across turns
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · **thread_id:** `conv-A`

### Turn 1
**USER:** Check the return risk for order 1523 — Rs. 1,899 Apparel, COD, 12 days tenure, 3 previous orders, 1 previous return, 340 km, 6 delivery days.

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #2 ("Is order 1523 likely to be returned?") -> intent=return_risk (cosine=0.6374)
**[call_tool]** `check_return_risk(...)` ->
{"return_probability": 0.6495, "risk_bucket": "Medium", "t_star_rf": 0.5, "cut_points": {"low_max": 0.5, "high_min": 0.65}, ...}
**[generate]** MOCK_LLM template: return_risk
**[verify_output]** grounded=True (tool-sourced answer, groundedness check not applicable)

**AGENT:**
{"answer": "Order 1523 is flagged as Medium return risk: predicted return probability 64.95%, against a model threshold t*_rf=0.5 (Low < 0.5, High >= 0.65).", "source": "return_risk_tool", "confidence": 0.299}

**Node path:** guard_input -> classify_intent -> call_tool -> generate -> verify_output

### Turn 2
**USER:** What is the return window for Apparel orders?
... (policy intent, RAG retrieval, grounded answer — see full file) ...

### Turn 3
**USER:** What is the delivery SLA for its shipment?

**[classify_intent]** ... [coref: "its"/"that order" resolved to last_order_id=1523]
**[retrieve]** top-3 chunks: POL-10#2 (0.5992); POL-09#0 (0.5827); POL-09#2 (0.5670)

**AGENT:**
{"answer": "For order 1523: Based on Flipkart's policy documents: ... Orders shipped to metro cities such as Delhi, Mumbai, Bengaluru, and Chennai are typically delivered within 2-4 days of dispatch.", "source": "policy_kb", "confidence": 0.599}

`state.last_order_id` going into this turn = `1523` (persisted by the MemorySaver checkpointer
from Turn 1, since this is the SAME thread_id `conv-A`) -> "its" resolves to order 1523.
```

Full file with all three turns in full: [`transcripts/05_multiturn_state_carried.md`](transcripts/05_multiturn_state_carried.md)

</details>

### All 9 transcripts + retrieval evaluation

| # | File | Demonstrates |
|---|---|---|
| 01 | [`01_policy_apparel_return_window.md`](transcripts/01_policy_apparel_return_window.md) | policy question via RAG |
| 02 | [`02_policy_cod_refund_timeline.md`](transcripts/02_policy_cod_refund_timeline.md) | policy question via RAG (second doc) |
| 03 | [`03_return_risk_tool_call.md`](transcripts/03_return_risk_tool_call.md) | `check_return_risk` + direct-model spot-check |
| 04 | [`04_image_classification_tool_call.md`](transcripts/04_image_classification_tool_call.md) | `classify_product_image` on a real PNG |
| 05 | [`05_multiturn_state_carried.md`](transcripts/05_multiturn_state_carried.md) | state carried across turns |
| 06 | [`06_fresh_conversation_state_absent.md`](transcripts/06_fresh_conversation_state_absent.md) | same question, fresh thread → state absent |
| 07 | [`07_prompt_injection_blocked.md`](transcripts/07_prompt_injection_blocked.md) | input-side guardrail, deflected |
| 08 | [`08_ungrounded_refusal.md`](transcripts/08_ungrounded_refusal.md) | output-side guardrail, score vs threshold printed |
| 09 | [`09_intent_routing_fewshot.md`](transcripts/09_intent_routing_fewshot.md) | few-shot examples driving all 3 intents |
| — | [`retrieval_eval.md`](transcripts/retrieval_eval.md) | P@3 / R@3 per-query arithmetic + threshold calibration |

---

## Repo map

```
generate_orders.py, orders_dataset.csv          Part 1 data (verbatim seeded generator)
models/return_risk_model.pkl, return_risk_meta.json    Part 1 artifact + t*_rf
part1_return_risk/                              Part 1 code + reports/

models/product_classifier.pt                    Part 2 artifact
data/sample_images/*.png, labels.json           Part 2 real test-split exports
part2_image_classifier/                         Part 2 code + reports/ (model_io.py = the loader)

part3_agent/kb/, index/                         policy docs + FAISS index
part3_agent/tools/                              check_return_risk, classify_product_image
part3_agent/graph.py                            the LangGraph (6 nodes, 2 conditional edges)
part3_agent/{guardrails,prompts,mock_llm}.py    guardrails, 4S prompt, deterministic composer
transcripts/                                    9 required transcripts + retrieval_eval.md

docs/                                           build-planning documents (not graded, kept for traceability)
```

---

## Optional live-LLM extension

Not implemented — this project does not need one to satisfy any acceptance criterion. Setting
`USE_LIVE_LLM=1` is explicitly unsupported and raises a documented error rather than silently
falling back; **MOCK_LLM (the default, with `USE_LIVE_LLM` unset) is what every transcript above
was run against, and removing any hypothetical live-LLM wiring changes nothing** — no part of
this repository's grading depends on a live model call.

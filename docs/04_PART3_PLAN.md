# Part 3 — Flipkart Support Agent (40 marks)

**Entry points:**
`python -m part3_agent.index_build` · `python -m part3_agent.run_transcripts` ·
`python -m part3_agent.eval_retrieval` · `python -m part3_agent.run_agent`

This is the user-facing product. Parts 1 and 2 exist to be called from here.

---

## Task 1 — Policy knowledge base (`kb/documents/`)

**15 documents** (brief requires ≥12), each **2–4 sentences**, one file per doc, front-matter
style header so the loader can read `doc_id` and `title`.

Required coverage: return windows by category (apparel/footwear vs electronics vs home), COD
refund timelines, delivery SLAs, reverse-pickup eligibility. The rest add realistic breadth.

| doc_id | title | covers |
|---|---|---|
| POL-01 | Apparel Return Window | 7-day return, tags intact, size/fit reasons |
| POL-02 | Footwear Return Window | 7-day return, unworn, original box |
| POL-03 | Electronics Return Window | 10-day, replacement-only for functional defects, no change-of-mind |
| POL-04 | Home & Kitchen Return Window | 7-day, unused, packaging intact |
| POL-05 | Beauty & Personal Care Returns | non-returnable on hygiene grounds unless damaged/wrong item |
| POL-06 | COD Refund Timeline | bank transfer after reverse pickup QC, 5–7 business days |
| POL-07 | Prepaid Card Refund Timeline | 3–5 business days back to source card |
| POL-08 | UPI and Wallet Refund Timeline | 24–48 hours to UPI, instant to wallet |
| POL-09 | Delivery SLA — Metro Cities | 2–4 days standard |
| POL-10 | Delivery SLA — Non-Metro and Remote Pincodes | 4–8 days, remote up to 10 |
| POL-11 | Reverse Pickup Eligibility | serviceable pincode, within window, original condition |
| POL-12 | Reverse Pickup Scheduling and Failed Attempts | 3 attempts, self-ship fallback |
| POL-13 | Exchange and Size Replacement | apparel/footwear only, one exchange per order |
| POL-14 | Damaged or Defective on Arrival | report within 48h, unboxing evidence, free pickup |
| POL-15 | Refund Status Tracking and Escalation | reference ID, escalate after SLA breach |

**File format** (`kb/documents/POL-01.md`):
```markdown
---
doc_id: POL-01
title: Apparel Return Window
category: returns
---
Apparel items on Flipkart are eligible for return within 7 days of delivery. The item must be
unused with all original tags and packaging intact. Size and fit issues are accepted as valid
return reasons for apparel. Returns requested after the 7-day window are automatically declined
by the system.
```

### Chunking (`chunking.py`)

**Sentence-wise** — the brief names it as the production-appropriate strategy over fixed-size or
overlapping windows. No NLTK dependency; a regex splitter on `(?<=[.!?])\s+` is sufficient and
deterministic for text you authored.

Every chunk carries its parent:
```python
{"chunk_id": "POL-01#2", "doc_id": "POL-01", "doc_title": "...",
 "sentence_index": 2, "text": "..."}
```

15 docs × ~3.5 sentences ≈ **50 chunks**. Multi-sentence docs each produce >1 chunk, which the
brief explicitly requires.

### Answer key (`kb/eval_queries.json`) — 6 queries (brief requires ≥5)

Relevance is recorded at the **document** level, one or two docs per query:

```json
[
  {"qid": "Q1", "query": "How many days do I have to return a kurta I bought?",
   "relevant_doc_ids": ["POL-01"]},
  {"qid": "Q2", "query": "When will I get my money back for a cash on delivery order?",
   "relevant_doc_ids": ["POL-06"]},
  {"qid": "Q3", "query": "Can I return a laptop if I simply changed my mind?",
   "relevant_doc_ids": ["POL-03"]},
  {"qid": "Q4", "query": "How long does delivery take to a remote pincode?",
   "relevant_doc_ids": ["POL-10", "POL-09"]},
  {"qid": "Q5", "query": "Am I eligible for a reverse pickup and what if the courier misses me?",
   "relevant_doc_ids": ["POL-11", "POL-12"]},
  {"qid": "Q6", "query": "Is a lipstick returnable?",
   "relevant_doc_ids": ["POL-05"]}
]
```

---

## Task 2 — Embed and index (`index_build.py`, `retriever.py`)

- Model: **`sentence-transformers/all-MiniLM-L6-v2`** — free, local, no key. ~90 MB, cached to
  `~/.cache/huggingface/` on first run.
- `encode(..., normalize_embeddings=True)` → unit vectors, so FAISS **`IndexFlatIP`** inner
  product **is** cosine similarity. Do not use L2 distance and then pretend it is similarity.
- Persist `part3_agent/index/faiss.index` + `part3_agent/index/chunks.json` and **commit both**,
  so a grader can run the agent without rebuilding.

`retriever.py`:
```python
def search(query: str, k: int = 3) -> list[dict]:
    """→ [{chunk_id, doc_id, doc_title, text, score}] sorted by score desc."""
```

### Similarity-threshold calibration (do this, don't guess a number)

`index_build.py` ends by printing a calibration table and writing it to
`transcripts/retrieval_eval.md`:

- top-1 cosine score for each of the **6 answer-key queries** (expect ≈ 0.45–0.75)
- top-1 cosine score for **3 deliberately out-of-scope queries**, e.g.
  *"What is the warranty on a car battery?"*, *"How do I apply for a job at Flipkart?"*,
  *"What is the capital of France?"* (expect ≈ 0.05–0.30)

Set `SIM_THRESHOLD` in `config.py` to a value cleanly between the two clusters — start at
**0.40** and adjust based on the measured table. **Record the table**; that is what makes the
Task 9(f) refusal verifiable rather than arbitrary.

---

## Task 3 — `check_return_risk(order_features: dict) -> dict`

`part3_agent/tools/return_risk_tool.py`

```python
_MODEL = None  # module-level lazy cache
_META  = json.load(open(MODELS / "return_risk_meta.json"))
T_STAR = _META["t_star_rf"]        # ← read from Part 1's artifact, NEVER a literal

def check_return_risk(order_features: dict) -> dict:
    model = _load()                             # joblib.load(models/return_risk_model.pkl)
    row   = pd.DataFrame([_normalise(order_features)])   # all training columns, missing → NaN
    p     = float(model.predict_proba(row)[0, 1])
    bucket = ("High"   if p >= T_STAR + 0.15 else
              "Medium" if p >= T_STAR        else "Low")
    return {"return_probability": round(p, 4), "risk_bucket": bucket,
            "t_star_rf": T_STAR,
            "cut_points": {"low_max": T_STAR, "high_min": round(T_STAR + 0.15, 4)},
            "model": "RandomForest (Part 1, GridSearchCV-tuned pipeline)",
            "features_used": row.iloc[0].to_dict()}
```

**Bucket rule, verbatim from the brief:** Low if `p < t*_rf`; High if `p >= t*_rf + 0.15`;
Medium otherwise. Fixed 0.3/0.6 cut points are explicitly wrong — they are not self-calibrating,
and a differently-tuned-but-equally-valid RF could dump every real order into one bucket.

`_normalise` fills any of the 11 training feature columns the caller omitted with `np.nan` — the
pipeline's median/mode imputer handles them, which is exactly why the imputer lives inside the
saved pipeline.

**Spot-check requirement.** The acceptance criteria say running the saved model directly outside
the agent must produce the same number. `run_transcripts.py` transcript 03 prints both:
the tool's output, and a direct `joblib.load(...).predict_proba(...)` on the same row.

The one-sentence justification for the README, filled from the real value:
> `t*_rf = 0.47` (F1-maximising threshold on the saved Random Forest's own `predict_proba` over
> the held-out test split), so the buckets are **Low** `p < 0.47`, **Medium** `0.47 ≤ p < 0.62`,
> **High** `p ≥ 0.62`.

---

## Task 4 — `classify_product_image(image_path: str) -> dict`

`part3_agent/tools/image_tool.py`

```python
from part2_image_classifier.model_io import predict_image   # ← the SAME documented snippet

def classify_product_image(image_path: str) -> dict:
    p = Path(image_path)
    if not p.exists():
        return {"error": f"image not found: {image_path}", "category": None, "confidence": 0.0}
    out = predict_image(p)
    return {"category": out["label"], "confidence": round(out["confidence"], 4),
            "image_path": str(p), "top3": out["top3"],
            "model": "ResNet-18 transfer learning (Part 2)"}
```

Pointed at the real committed PNGs in `data/sample_images/`. No upload, no raw IDX, no hardcoded
label. The agent resolves a bare filename (`07_sneaker.png`) against `data/sample_images/`.

---

## Task 5 — The LangGraph graph (`state.py`, `graph.py`)

### State

```python
class AgentState(TypedDict, total=False):
    user_input: str
    turn_index: int
    history: list[dict]              # [{role, content}] within THIS conversation
    intent: str                      # policy | return_risk | product_category | unknown
    matched_fewshot: str             # which few-shot example the router matched
    injection_blocked: bool
    injection_pattern: str | None
    retrieved: list[dict]
    top_score: float
    tool_name: str | None
    tool_result: dict | None
    # ── the carried state the multi-turn transcript demonstrates ──
    last_order_id: str | None
    last_order_features: dict | None
    last_image_path: str | None
    grounded: bool
    final: dict                      # {answer, source, confidence}
```

### Nodes — 6 (brief requires ≥4)

| # | node | does |
|---|---|---|
| 1 | `guard_input` | prompt-injection regex scan; sets `injection_blocked` |
| 2 | `classify_intent` | few-shot-driven routing + **coreference resolution** ("its", "that order" → `last_order_id` from state) |
| 3 | `retrieve` | FAISS top-3, records `top_score` |
| 4 | `call_tool` | dispatch to `check_return_risk` or `classify_product_image`; writes `last_order_id` / `last_image_path` back into state |
| 5 | `generate` | MOCK_LLM composes the structured JSON |
| 6 | `verify_output` | output-side groundedness check; overwrites `final` with a refusal if `top_score < SIM_THRESHOLD` on a policy answer |

### Edges — 2 conditional (brief requires ≥1)

```
START → guard_input
guard_input  ──conditional──▶  "blocked"  → generate          (skips retrieval and tools entirely)
                               "clean"    → classify_intent
classify_intent ──conditional──▶ "policy"           → retrieve
                                 "return_risk"      → call_tool
                                 "product_category" → call_tool
                                 "unknown"          → retrieve
retrieve  → generate
call_tool → generate
generate  → verify_output
verify_output → END
```

The branching is real: a blocked injection never touches FAISS or a model; a risk question never
runs retrieval. Print the executed node path in every transcript — that is the visible proof.

### State vs memory (this is a graded distinction)

Use LangGraph's `MemorySaver` checkpointer, invoked with a `thread_id`:

```python
graph = builder.compile(checkpointer=MemorySaver())
graph.invoke({"user_input": q}, config={"configurable": {"thread_id": "conv-A"}})
```

- **Same `thread_id`** → `last_order_id` persists across turns → *"what about its delivery?"*
  resolves to the order from two turns ago. → `transcripts/05_multiturn_state_carried.md`
- **New `thread_id`** (`"conv-B"`) → state starts empty → the identical question yields
  *"There is no order referenced earlier in this conversation."* →
  `transcripts/06_fresh_conversation_state_absent.md`

That contrast — same question, different thread, different answer — is the whole demonstration.
It is **state** (scoped to one conversation, cleared on a new one), not **memory** (persisted
across conversations).

---

## Task 6 — Prompt engineering (`prompts.py`)

Two **separate** prompts — that separation is itself the **Single** principle:

`INTENT_PROMPT` (classification only) and `RESPONSE_PROMPT` (composition only).

Annotate each block with an inline comment naming the principle it satisfies:

```python
SYSTEM_PROMPT = """
You are Flipkart's customer-support assistant.          # ROLE PROMPTING
Answer only from the context supplied below.            # [SPECIFIC] scope is explicit
You have exactly three capabilities: policy lookup,     # [SPECIFIC] enumerated, no ambiguity
return-risk scoring, product-image categorisation.
Never invent a policy that is not in the context.       # [SPECIFIC] the failure mode, named

<<<CONTEXT>>>                                           # [SURROUND] delimited context block
{context}
<<<END CONTEXT>>>

Reply with one JSON object and nothing else:            # [SINGLE] one task, one output shape
{{"answer": str, "source": "policy_kb"|"return_risk_tool"|"image_classifier_tool",
  "confidence": float}}
"""                                                     # [SHORT] under 120 words, no filler
```

The README must reproduce this prompt **with the annotations visible** — the criterion is that
it is annotated against each of the 4S plus role prompting.

### Few-shot intent examples (≥2 required; write 3)

```python
FEW_SHOT_INTENT = [
  {"user": "What is the return window for a pair of running shoes?",
   "intent": "policy"},
  {"user": "Is order 1523 likely to be returned?",
   "intent": "return_risk"},
  {"user": "Which category does data/sample_images/07_sneaker.png belong to?",
   "intent": "product_category"},
]
```

**They must visibly drive routing.** In MOCK_LLM mode the router scores the user's input against
each few-shot exemplar (embedding cosine via the same MiniLM model, plus the pattern rules) and
records **which exemplar won** in `state["matched_fewshot"]`. Every transcript prints:

```
[intent] matched few-shot example #2 ("Is order 1523 likely to be returned?") → intent=return_risk
```

That line appearing in transcripts 03, 04 and 09 satisfies "actually driving correct intent
routing on at least 2 of the required transcripts", not merely present in the prompt text.

---

## Task 7 — MOCK_LLM deterministic mode (`mock_llm.py`)

Default and graded mode. Zero network, zero keys, fully deterministic.

```python
def compose(state: AgentState) -> dict:
    """Rule-based/template composition of the final structured answer."""
```

Branches:
- **blocked** → `{"answer": "I can't follow instructions that try to override my configuration. I can help with return policies, return-risk checks, or product-image categorisation.", "source": "policy_kb", "confidence": 1.0}`
- **policy** → stitch the top retrieved chunk(s) into a templated sentence; `source="policy_kb"`;
  `confidence = round(top_score, 3)`
- **return_risk** → template over the tool dict; `source="return_risk_tool"`;
  `confidence` = distance of `p` from `t*_rf`, normalised (deterministic function, documented)
- **product_category** → template over the tool dict; `source="image_classifier_tool"`;
  `confidence` = the model's softmax probability
- **ungrounded** → refusal template (written by `verify_output`, see Task 8)

Config flag:
```python
MOCK_LLM   = os.getenv("USE_LIVE_LLM") != "1"   # default True
```
The optional live path may exist behind `USE_LIVE_LLM=1`, is marked optional in the README, is
never scored, and removing it changes nothing.

---

## Task 8 — Guardrails (`guardrails.py`)

### Input side — prompt-injection filter

```python
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+all\s+rules",
    r"disregard\s+(the\s+)?(above|previous|system)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+",
    r"(reveal|show|print|repeat)\s+(your\s+)?(system\s+)?(prompt|instructions|policies)",
    r"act\s+as\s+(a|an|the)\s+",
    r"developer\s+mode|jailbreak|bypass\s+(your\s+)?(rules|filters|guardrails)",
    r"(new|updated)\s+(system|admin)\s+(prompt|instruction)",
]

def check_input(text: str) -> tuple[bool, str | None]:
    """→ (blocked, matched_pattern)"""
```

Matching is case-insensitive on the raw input. On a hit: `injection_blocked=True`, the graph
routes straight to `generate`, retrieval and tools never execute, and the transcript prints the
matched pattern.

### Output side — groundedness check

```python
def check_groundedness(top_score: float, threshold: float) -> tuple[bool, str]:
    grounded = top_score >= threshold
    msg = (f"top retrieved chunk similarity = {top_score:.4f} "
           f"{'≥' if grounded else '<'} threshold {threshold:.2f}")
    return grounded, msg
```

Applies to **policy** answers only (tool answers are grounded by construction — a model
produced them). If not grounded, `verify_output` **replaces** `final` with:

```json
{"answer": "I don't have a policy document covering that, so I won't guess. Please contact a
 human support agent. (top retrieved chunk similarity = 0.2118 < threshold 0.40)",
 "source": "policy_kb", "confidence": 0.0}
```

The score and the threshold appear **in the answer text itself**, so the refusal is verifiable
straight off the transcript — which is exactly what criterion (f) demands.

---

## Task 9 — 9 transcripts (`run_transcripts.py`) → `transcripts/`

One script regenerates all of them deterministically. Every transcript uses this template:

```markdown
# Transcript 03 — Return-risk question (tool call)
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · **thread_id:** `conv-03`

### Turn 1
**USER:** Is order 1523 likely to be returned? It's a ₹1,899 Apparel item paid by COD ...

**[guard_input]** no injection pattern matched → clean
**[classify_intent]** matched few-shot example #2 (...) → intent=`return_risk`
**[call_tool]** check_return_risk({...})
    → {"return_probability": 0.6431, "risk_bucket": "High", "t_star_rf": 0.47, ...}
**[generate]** MOCK_LLM template: return_risk
**[verify_output]** tool-sourced answer, groundedness check not applicable

**AGENT:**
```json
{"answer": "...", "source": "return_risk_tool", "confidence": 0.83}
```

**Node path:** `guard_input → classify_intent → call_tool → generate → verify_output`

**Spot-check (model called directly, outside the agent):**
`joblib.load("models/return_risk_model.pkl").predict_proba(row)[0,1] = 0.6431` ✓ identical
```

| file | requirement | shows |
|---|---|---|
| `01_policy_apparel_return_window.md` | (a) | RAG policy answer #1 + retrieved chunks with scores |
| `02_policy_cod_refund_timeline.md` | (a) | RAG policy answer #2 |
| `03_return_risk_tool_call.md` | (b) | `check_return_risk` on realistic features + spot-check |
| `04_image_classification_tool_call.md` | (c) | `classify_product_image` on a real PNG |
| `05_multiturn_state_carried.md` | (d) | 3 turns, `thread_id=conv-A`, "its" resolves to order from turn 1 |
| `06_fresh_conversation_state_absent.md` | (d) | same final question, `thread_id=conv-B`, state empty |
| `07_prompt_injection_blocked.md` | (e) | injection deflected, matched pattern printed |
| `08_ungrounded_refusal.md` | (f) | refusal with similarity score vs threshold printed |
| `09_intent_routing_fewshot.md` | prompt criterion | 3 inputs, few-shot match line + route for each |

Transcript 05's script:
1. *"Check the return risk for order 1523 — ₹1,899 Apparel, COD, 12 days tenure, 3 previous orders, 1 previous return, 340 km, 6 delivery days."* → tool call, `last_order_id="1523"` stored
2. *"What is the return policy for that category?"* → policy intent, still same thread
3. *"And what about its delivery timeline?"* → **"its" resolves to order 1523** using
   `last_order_id` from state; the answer references it explicitly

Transcript 06: the exact same turn-3 question, first message of `thread_id=conv-B` → agent
answers that no order is referenced in this conversation. Print `state.last_order_id = None`.

---

## Task 10 — Retrieval evaluation (`eval_retrieval.py`) → `transcripts/retrieval_eval.md`

Document-level, deduplicated, per-query arithmetic shown.

```
For each query:
  chunks   = search(query, k=3)                    # 3 chunks
  ret_docs = dedup([c.doc_id for c in chunks])     # ≤ 3 documents
  hits     = ret_docs ∩ relevant_doc_ids
  P@3      = |hits| / |ret_docs|
  R@3      = |hits| / |relevant_doc_ids|
```

Report per query, showing the fraction **before** the decimal:

```markdown
### Q4 — "How long does delivery take to a remote pincode?"
- relevant docs (answer key): {POL-10, POL-09}
- top-3 chunks: POL-10#1 (0.71), POL-10#3 (0.66), POL-09#2 (0.58)
- retrieved docs after dedup: {POL-10, POL-09}  (3 chunks → 2 unique documents)
- hits: {POL-10, POL-09} → 2
- Precision@3 = 2 / 2 = 1.000
- Recall@3    = 2 / 2 = 1.000
```

Then the two averages across all 6 queries. **Document the dedup choice explicitly**: because
scoring is at the document level, the precision denominator is the number of *unique documents*
returned, not a fixed 3 — state that so the arithmetic is unambiguous to a grader.

Append the similarity-threshold calibration table from Task 2 to the bottom of this file.

---

## Part 3 acceptance self-check

- [ ] ≥12 chunked documents (we have 15 → ~50 chunks), sentence-wise, chunk→doc map exists
- [ ] embeddings from a free local model; index is FAISS
- [ ] both tools load the **real** Part 1 / Part 2 artifacts; spot-check reproduces the number
- [ ] `classify_product_image` points at real PNGs in `data/sample_images/`
- [ ] risk buckets anchored to `t*_rf` (read from `return_risk_meta.json`, not a literal); the
      one-sentence justification states both the cut points and `t*_rf`
- [ ] graph has ≥4 nodes (6) and ≥1 conditional edge (2)
- [ ] multi-turn transcript shows state carried; fresh-conversation transcript shows it absent
- [ ] system prompt annotated against Specific / Short / Surround / Single + role prompting
- [ ] ≥2 few-shot intent examples visibly driving routing in ≥2 transcripts
- [ ] MOCK_LLM default; all 9 transcripts run with `USE_LIVE_LLM` unset; zero keys, zero calls
- [ ] injection transcript present and visibly deflected
- [ ] ungrounded transcript present, refusal prints similarity score vs threshold
- [ ] P@3 / R@3 at document level, deduped, ≥5 queries, per-query arithmetic visible

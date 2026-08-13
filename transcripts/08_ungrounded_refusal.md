# Transcript 08 -- Ungrounded question, output-side refusal
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · **thread_id:** `conv-08`

### Turn 1
**USER:** What is the capital of France?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #3 ("Which category does data/sample_images/07_sneaker.png belong to?") -> intent=product_category (cosine=0.0070) [overridden -> intent=unknown; reason: raw cosine 0.0070 below unknown-confidence threshold 0.2]
**[retrieve]** top-3 chunks: POL-09#0 (0.0811); POL-09#1 (0.0700); POL-09#2 (0.0605)
**[generate]** MOCK_LLM template: unknown
**[verify_output]** grounded=False -- top retrieved chunk similarity = 0.0811 < threshold 0.4800

**AGENT:**
```json
{
  "answer": "I don't have a policy document covering that, so I won't guess. Please contact a human support agent. (top retrieved chunk similarity = 0.0811 < threshold 0.4800)",
  "source": "policy_kb",
  "confidence": 0.0
}
```

**Node path:** `guard_input -> classify_intent -> retrieve -> generate -> verify_output`

`SIM_THRESHOLD` = 0.48 (see `part3_agent/config.py`, calibrated in `transcripts/retrieval_eval.md`). The refusal text above embeds the exact similarity score and threshold so the refusal is verifiable straight off this transcript.

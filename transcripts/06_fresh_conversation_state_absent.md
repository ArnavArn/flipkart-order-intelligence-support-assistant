# Transcript 06 -- Fresh conversation, state correctly absent
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · **thread_id:** `conv-B`

### Turn 1
**USER:** What is the delivery SLA for its shipment?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #3 ("Which category does data/sample_images/07_sneaker.png belong to?") -> intent=product_category (cosine=0.2145) [overridden -> intent=policy; reason: policy-domain keyword present without an order/image signal] [coref: pronoun detected but no last_order_id in state -- unresolved]
**[retrieve]** top-3 chunks: POL-10#2 (0.5992); POL-09#0 (0.5827); POL-09#2 (0.5670)
**[generate]** MOCK_LLM template: policy
**[verify_output]** grounded=True

**AGENT:**
```json
{
  "answer": "There is no order referenced earlier in this conversation.",
  "source": "policy_kb",
  "confidence": 0.0
}
```

**Node path:** `guard_input -> classify_intent -> retrieve -> generate -> verify_output`

`state.last_order_id` = `None` -- this is the FIRST message of a fresh thread_id (`conv-B`), so no order was ever stored in this conversation's state. The identical question that resolved cleanly in transcript 05 (same thread as an earlier return-risk turn) instead gets a direct "no order referenced" answer here.

**Contrast with `transcripts/05_multiturn_state_carried.md` Turn 3:** same exact question ("What is the delivery SLA for its shipment?"), different thread_id, different answer. That is **state** (scoped to one conversation, cleared on a new thread_id), not memory (persisted across conversations).

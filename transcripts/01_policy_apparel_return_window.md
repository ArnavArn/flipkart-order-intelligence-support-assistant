# Transcript 01 -- Policy question: apparel return window (RAG)
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · **thread_id:** `conv-01`

### Turn 1
**USER:** How many days do I have to return a kurta I bought?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #2 ("Is order 1523 likely to be returned?") -> intent=return_risk (cosine=0.2784) [overridden -> intent=policy; reason: policy-domain keyword present without an order/image signal]
**[retrieve]** top-3 chunks: POL-01#0 (0.4941); POL-02#0 (0.4644); POL-04#0 (0.4353)
**[generate]** MOCK_LLM template: policy
**[verify_output]** grounded=True -- top retrieved chunk similarity = 0.4941 >= threshold 0.4800

**AGENT:**
```json
{
  "answer": "Based on Flipkart's policy documents: Apparel items on Flipkart are eligible for return within 7 days of delivery. Footwear purchased on Flipkart can be returned within 7 days of delivery.",
  "source": "policy_kb",
  "confidence": 0.494
}
```

**Node path:** `guard_input -> classify_intent -> retrieve -> generate -> verify_output`

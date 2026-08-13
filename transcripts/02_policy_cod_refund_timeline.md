# Transcript 02 -- Policy question: COD refund timeline (RAG)
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · **thread_id:** `conv-02`

### Turn 1
**USER:** When will I get my money back for a cash on delivery order?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #2 ("Is order 1523 likely to be returned?") -> intent=return_risk (cosine=0.4494) [overridden -> intent=policy; reason: policy-domain keyword present without an order/image signal]
**[retrieve]** top-3 chunks: POL-06#0 (0.5606); POL-08#3 (0.5195); POL-08#0 (0.5136)
**[generate]** MOCK_LLM template: policy
**[verify_output]** grounded=True -- top retrieved chunk similarity = 0.5606 >= threshold 0.4800

**AGENT:**
```json
{
  "answer": "Based on Flipkart's policy documents: For cash-on-delivery orders, refunds are issued as a bank transfer since there is no source payment instrument to credit back. Customers can track either refund from the order details page.",
  "source": "policy_kb",
  "confidence": 0.561
}
```

**Node path:** `guard_input -> classify_intent -> retrieve -> generate -> verify_output`

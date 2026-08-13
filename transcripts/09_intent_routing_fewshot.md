# Transcript 09 -- Few-shot intent routing, 3 inputs
**Mode:** MOCK_LLM (USE_LIVE_LLM unset) · each input runs on its own fresh thread_id so routing is shown in isolation.

Each input below is one of the 3 `FEW_SHOT_INTENT` exemplars verbatim (`part3_agent/prompts.py`), so the router's cosine similarity to that exact exemplar is 1.0000 -- the strongest possible demonstration that the few-shot match is what drives the routing decision, not an incidental correlation.

### Turn 1
**USER:** What is the return window for a pair of running shoes?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #1 ("What is the return window for a pair of running shoes?") -> intent=policy (cosine=1.0000)
**[retrieve]** top-3 chunks: POL-02#1 (0.6155); POL-02#3 (0.5092); POL-02#2 (0.5090)
**[generate]** MOCK_LLM template: policy
**[verify_output]** grounded=True -- top retrieved chunk similarity = 0.6155 >= threshold 0.4800

**AGENT:**
```json
{
  "answer": "Based on Flipkart's policy documents: Shoes must be unworn, show no signs of outdoor use, and be returned in the original box with all accessories. Footwear returned after 7 days, or showing wear on the sole, will not be accepted.",
  "source": "policy_kb",
  "confidence": 0.615
}
```

**Node path:** `guard_input -> classify_intent -> retrieve -> generate -> verify_output`

### Turn 2
**USER:** Is order 1523 likely to be returned?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #2 ("Is order 1523 likely to be returned?") -> intent=return_risk (cosine=1.0000)
**[call_tool]** `check_return_risk(...)` ->
```json
{
  "return_probability": 0.6208,
  "risk_bucket": "Medium",
  "t_star_rf": 0.5,
  "cut_points": {
    "low_max": 0.5,
    "high_min": 0.65
  },
  "model": "RandomForest (Part 1, GridSearchCV-tuned pipeline)",
  "features_used": {
    "price_inr": NaN,
    "discount_pct": NaN,
    "customer_tenure_days": NaN,
    "num_previous_orders": NaN,
    "num_previous_returns": NaN,
    "delivery_distance_km": NaN,
    "delivery_days": NaN,
    "is_weekend_order": NaN,
    "rating_given": NaN,
    "product_category": NaN,
    "payment_method": NaN
  }
}
```
**[generate]** MOCK_LLM template: return_risk
**[verify_output]** grounded=True (tool-sourced answer, groundedness check not applicable)

**AGENT:**
```json
{
  "answer": "Order 1523 is flagged as Medium return risk: predicted return probability 62.08%, against a model threshold t*_rf=0.5 (Low < 0.5, High >= 0.65).",
  "source": "return_risk_tool",
  "confidence": 0.2416
}
```

**Node path:** `guard_input -> classify_intent -> call_tool -> generate -> verify_output`

### Turn 3
**USER:** Which category does data/sample_images/07_sneaker.png belong to?

**[guard_input]** no injection pattern matched -> clean
**[classify_intent]** matched few-shot example #3 ("Which category does data/sample_images/07_sneaker.png belong to?") -> intent=product_category (cosine=1.0000)
**[call_tool]** `classify_product_image(...)` ->
```json
{
  "category": "Sneaker",
  "confidence": 0.9999,
  "image_path": "data/sample_images/07_sneaker.png",
  "top3": {
    "Sneaker": 0.9999,
    "Sandal": 0.0001,
    "Ankle boot": 0.0
  },
  "model": "ResNet-18 transfer learning (Part 2)"
}
```
**[generate]** MOCK_LLM template: product_category
**[verify_output]** grounded=True (tool-sourced answer, groundedness check not applicable)

**AGENT:**
```json
{
  "answer": "This product image is classified as 'Sneaker' with confidence 99.99%.",
  "source": "image_classifier_tool",
  "confidence": 0.9999
}
```

**Node path:** `guard_input -> classify_intent -> call_tool -> generate -> verify_output`

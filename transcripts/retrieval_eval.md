# Retrieval evaluation -- Precision@3 / Recall@3 (document level, deduped)

Scoring is at the **document** level: each query's top-3 chunks are mapped to their parent `doc_id` and deduplicated, so the Precision@3 denominator is the number of *unique documents* retrieved (<= 3), not a fixed 3. Recall@3's denominator is the number of documents in the answer key for that query.

### Q1 -- "How many days do I have to return a kurta I bought?"
- relevant docs (answer key): {POL-01}
- top-3 chunks: POL-01#0 (0.49), POL-02#0 (0.46), POL-04#0 (0.44)
- retrieved docs after dedup: {POL-01, POL-02, POL-04}  (3 chunks -> 3 unique documents)
- hits: {POL-01} -> 1
- Precision@3 = 1 / 3 = 0.333
- Recall@3    = 1 / 1 = 1.000

### Q2 -- "When will I get my money back for a cash on delivery order?"
- relevant docs (answer key): {POL-06}
- top-3 chunks: POL-06#0 (0.56), POL-08#3 (0.52), POL-08#0 (0.51)
- retrieved docs after dedup: {POL-06, POL-08}  (3 chunks -> 2 unique documents)
- hits: {POL-06} -> 1
- Precision@3 = 1 / 2 = 0.500
- Recall@3    = 1 / 1 = 1.000

### Q3 -- "Can I return a laptop if I simply changed my mind?"
- relevant docs (answer key): {POL-03}
- top-3 chunks: POL-03#2 (0.50), POL-03#0 (0.48), POL-03#3 (0.46)
- retrieved docs after dedup: {POL-03}  (3 chunks -> 1 unique documents)
- hits: {POL-03} -> 1
- Precision@3 = 1 / 1 = 1.000
- Recall@3    = 1 / 1 = 1.000

### Q4 -- "How long does delivery take to a remote pincode?"
- relevant docs (answer key): {POL-09, POL-10}
- top-3 chunks: POL-10#1 (0.76), POL-10#3 (0.67), POL-10#0 (0.67)
- retrieved docs after dedup: {POL-10}  (3 chunks -> 1 unique documents)
- hits: {POL-10} -> 1
- Precision@3 = 1 / 1 = 1.000
- Recall@3    = 1 / 2 = 0.500

### Q5 -- "Am I eligible for a reverse pickup and what if the courier misses me?"
- relevant docs (answer key): {POL-11, POL-12}
- top-3 chunks: POL-12#0 (0.72), POL-11#0 (0.68), POL-14#2 (0.63)
- retrieved docs after dedup: {POL-12, POL-11, POL-14}  (3 chunks -> 3 unique documents)
- hits: {POL-11, POL-12} -> 2
- Precision@3 = 2 / 3 = 0.667
- Recall@3    = 2 / 2 = 1.000

### Q6 -- "Is a lipstick returnable?"
- relevant docs (answer key): {POL-05}
- top-3 chunks: POL-05#0 (0.71), POL-05#3 (0.57), POL-05#1 (0.49)
- retrieved docs after dedup: {POL-05}  (3 chunks -> 1 unique documents)
- hits: {POL-05} -> 1
- Precision@3 = 1 / 1 = 1.000
- Recall@3    = 1 / 1 = 1.000

## Averages across all 6 queries
- mean Precision@3 = (0.333 + 0.500 + 1.000 + 1.000 + 0.667 + 1.000) / 6 = 0.750
- mean Recall@3    = (1.000 + 1.000 + 1.000 + 0.500 + 1.000 + 1.000) / 6 = 0.917

---

## Similarity-threshold calibration

Top-1 cosine score (FAISS `IndexFlatIP` over unit-normalised MiniLM embeddings) for the 6 in-scope answer-key queries and 3 deliberately out-of-scope queries. `SIM_THRESHOLD = 0.48` in `config.py` was set by inspecting this table and placing the threshold cleanly between the two clusters.

| query | in-scope? | top-1 doc | top-1 score |
|---|---|---|---|
| How many days do I have to return a kurta I bought? | yes | POL-01 | 0.4941 |
| When will I get my money back for a cash on delivery order? | yes | POL-06 | 0.5606 |
| Can I return a laptop if I simply changed my mind? | yes | POL-03 | 0.5048 |
| How long does delivery take to a remote pincode? | yes | POL-10 | 0.7629 |
| Am I eligible for a reverse pickup and what if the courier misses me? | yes | POL-12 | 0.7244 |
| Is a lipstick returnable? | yes | POL-05 | 0.7094 |
| What is the warranty on a car battery? | no | POL-14 | 0.3198 |
| How do I apply for a job at Flipkart? | no | POL-01 | 0.4650 |
| What is the capital of France? | no | POL-09 | 0.0811 |

In-scope cluster: min=0.4941, max=0.7629. Out-of-scope cluster: min=0.0811, max=0.4650. SIM_THRESHOLD=0.48 sits between 0.4650 (highest out-of-scope score) and 0.4941 (lowest in-scope score).

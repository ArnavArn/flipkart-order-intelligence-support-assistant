# 01 — Data Checks

## Acceptance gates

- **[PASS] Shape 6000 x 13** — actual = 6000 rows x 13 cols
- **[PASS] Return rate in [18%, 27%]** — actual = 22.75%
- **[PASS] rating_given missing in [8%, 18%]** — actual = 13.05%

## 1. Shape

- Total rows: **6000**
- Total columns: **13**

## 2. Overall return rate

- Overall return rate: **22.75%** (1365 of 6000 orders returned)

## 3. Missingness in rating_given

- % missing overall: **13.05%**

## 4. Return rate by product_category

| product_category | n | return_rate_pct |
| --- | --- | --- |
| Apparel | 1979 | 26.43 |
| Beauty | 579 | 20.03 |
| Electronics | 1316 | 18.69 |
| Footwear | 1071 | 25.96 |
| Home | 1055 | 19.15 |

## 5. Return rate by payment_method

| payment_method | n | return_rate_pct |
| --- | --- | --- |
| COD | 2501 | 30.75 |
| Prepaid_Card | 1457 | 16.82 |
| Prepaid_UPI | 1448 | 16.92 |
| Wallet | 594 | 17.85 |

## 6. MAR evidence table

| payment_method | n | n missing rating | % missing |
| --- | --- | --- | --- |
| COD | 2501 | 571 | 22.83 |
| Prepaid_Card | 1457 | 92 | 6.31 |
| Prepaid_UPI | 1448 | 82 | 5.66 |
| Wallet | 594 | 38 | 6.40 |

**COD missing rate − non-COD missing rate = 16.8 pp**

## MAR paragraph

This missingness is **MAR — missing at random, conditional on an observed column.** The observed column it is conditional on is **`payment_method`**. The measured gap between COD and non-COD missing rates is **16.8 percentage points** (22.83% for COD vs an average of roughly 6.12% for the three non-COD methods) — this gap is the evidence.

It is **not MCAR** (missing completely at random), because a genuine, large dependency on `payment_method` exists — a gap of 16.8 pp is far too large to be chance; under MCAR the missing rate would be roughly constant across payment methods.

It is **not MNAR** (missing not at random), because the missingness depends on the *observed* column `payment_method`, not on the unobserved `rating_given` value itself. The generator's mask is built as:

```python
missing_mask = rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
```

which never inspects `rating_given` — it only branches on `payment_method`. That is exactly the textbook definition of MAR, and exactly why it is not MNAR.

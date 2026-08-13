# 05 — Feature Importance: Impurity vs Permutation

- **[PASS] Top-5 impurity includes payment_method (one-hot) + >=2 of {price_inr, customer_tenure_days, discount_pct, num_previous_returns}** — cat__payment_method_COD in raw top5 = True; matches = ['price_inr', 'customer_tenure_days']

## 7a. Impurity importance — raw (post-one-hot) table

| rank | feature | importance |
| --- | --- | --- |
| 1 | cat__payment_method_COD | 0.1788 |
| 2 | num__price_inr | 0.1323 |
| 3 | num__delivery_distance_km | 0.0957 |
| 4 | num__customer_tenure_days | 0.0900 |
| 5 | num__delivery_days | 0.0884 |
| 6 | num__discount_pct | 0.0859 |
| 7 | num__num_previous_orders | 0.0630 |
| 8 | num__num_previous_returns | 0.0460 |
| 9 | cat__payment_method_Prepaid_Card | 0.0413 |
| 10 | num__rating_given | 0.0396 |

### Top 5 (raw) interpretation

- **cat__payment_method_COD** (importance 0.1788)
- **num__price_inr** (importance 0.1323)
- **num__delivery_distance_km** (importance 0.0957)
- **num__customer_tenure_days** (importance 0.0900)
- **num__delivery_days** (importance 0.0884)

These plausibly drive return risk because: `payment_method`/COD orders skip upfront payment commitment, correlating with a higher propensity to reject/return on delivery; `price_inr` sets the absolute stake of the order (higher-value items are scrutinized more and returned more when unsatisfactory); `num_previous_returns`/`customer_tenure_days` capture a customer's historical return behaviour, a direct behavioural signal; and `discount_pct` correlates with impulse purchases that are more likely to be reconsidered and returned.

## 7b. Permutation importance (held-out test split, n_repeats=10, random_state=42, scoring='roc_auc')

Permutation importance runs on the **whole fitted Pipeline** against the raw (pre-transform) test columns, so shuffling happens at the original-column level — the interpretation below is directly comparable to the grouped impurity table, not the one-hot table.

### Side-by-side: grouped impurity vs permutation

| feature | impurity_rank | impurity_value | permutation_rank | perm_mean | perm_std |
| --- | --- | --- | --- | --- | --- |
| payment_method | 1 | 0.2692 | 1 | 0.0980 | 0.0098 |
| price_inr | 2 | 0.1323 | 2 | 0.0102 | 0.0042 |
| delivery_distance_km | 3 | 0.0957 | 7 | -0.0002 | 0.0016 |
| customer_tenure_days | 4 | 0.0900 | 11 | -0.0055 | 0.0018 |
| delivery_days | 5 | 0.0884 | 5 | 0.0026 | 0.0038 |
| discount_pct | 6 | 0.0859 | 8 | -0.0002 | 0.0026 |
| product_category | 7 | 0.0795 | 4 | 0.0060 | 0.0058 |
| num_previous_orders | 8 | 0.0630 | 10 | -0.0024 | 0.0015 |
| num_previous_returns | 9 | 0.0460 | 3 | 0.0085 | 0.0024 |
| rating_given | 10 | 0.0396 | 9 | -0.0019 | 0.0013 |
| is_weekend_order | 11 | 0.0103 | 6 | 0.0012 | 0.0007 |

### The planted decoy

`delivery_distance_km` ranks high on impurity importance but its permutation mean collapses toward ~0 on the held-out test split (and to a lesser degree `delivery_days`, `price_inr`, `num_previous_orders` show the same pattern of higher impurity rank than permutation rank). This is verified against the generator's own `z` formula:

```
z = -2.2 + 1.9*prev_return_ratio + 0.55*fit_risk_cat + 0.014*(discount_pct-20)/10
    + 0.9*(payment=="COD") + 0.10*(delivery_days-4.5)/2
    + 0.30*(price_inr/45000) + 0.05*is_weekend_order - 0.15*tanh(tenure/500)
```

`delivery_distance_km` **does not appear at all** in this formula — it is pure noise by construction, planted specifically to test whether impurity importance would be fooled by it (it is: continuous noise columns with many distinct values offer many candidate split points that reduce *training* impurity by chance).

**Why the gap exists (one sentence):** impurity-based importance counts how often a feature is chosen for a split and how much *training-set* impurity that split removes, so a noisy continuous column with thousands of distinct values offers enormous numbers of candidate split points and some of them will reduce training impurity purely by chance, inflating its score even though it carries no real signal; permutation importance instead measures the actual drop in **test-set** performance when the column is shuffled, so a true noise feature correctly scores ~0.

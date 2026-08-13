# 06 — Subgroup Analysis

All metrics below are computed on the **test split**, at the deployed operating point **t\*_rf = 0.5000**.

## Table A — by product_category

| product_category | n | support_returns | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| Apparel | 385 | 100 | 0.3171 | 0.5200 | 0.3939 |
| Beauty | 116 | 31 | 0.4750 | 0.6129 | 0.5352 |
| Electronics | 261 | 52 | 0.3286 | 0.4423 | 0.3770 |
| Footwear | 217 | 56 | 0.3626 | 0.5893 | 0.4490 |
| Home | 221 | 34 | 0.2347 | 0.6765 | 0.3485 |
| OVERALL | 1200 | 273 | 0.3240 | 0.5495 | 0.4076 |

## Table B — by payment_method

| payment_method | n | support_returns | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| COD | 503 | 155 | 0.3273 | 0.9355 | 0.4849 |
| Prepaid_Card | 283 | 49 | 0.2000 | 0.0204 | 0.0370 |
| Prepaid_UPI | 294 | 48 | 0.3333 | 0.0417 | 0.0741 |
| Wallet | 120 | 21 | 0.2222 | 0.0952 | 0.1333 |
| OVERALL | 1200 | 273 | 0.3240 | 0.5495 | 0.4076 |

## Diagnostic: mean predicted probability among actual (y=1) returns

If a group's actual returns cluster with predicted probabilities close to (just below) `t*_rf`, a fixed global threshold will under-catch that group even though the model ranks those orders as somewhat risky — this is the concrete mechanism behind a recall gap, not just a category label.

| product_category | mean_proba_among_actual_returns |
| --- | --- |
| Apparel | 0.5295 |
| Beauty | 0.4992 |
| Electronics | 0.4548 |
| Footwear | 0.5220 |
| Home | 0.5088 |

| payment_method | mean_proba_among_actual_returns |
| --- | --- |
| COD | 0.5780 |
| Prepaid_Card | 0.4185 |
| Prepaid_UPI | 0.4089 |
| Wallet | 0.4229 |

## Weak subgroup + specific fix

Within `product_category`, **Electronics** has the lowest recall (**0.4423** vs the overall **0.5495**, on 52 actual returns out of 261 orders). Among its actual returns, the mean predicted probability is only **0.4548** — sitting close to `t*_rf` = 0.5000, so a large share of this group's true positives fall just under the global cut point and are missed. This is consistent with the generator's structure: the risk score `z` only carries the `+0.55*fit_risk_cat` bonus for `Apparel`/`Footwear`, and `Electronics` does not receive it, so its positive class is driven almost entirely by the noisier `prev_return_ratio` and price terms, leaving less separation for the model to exploit.

Within `payment_method`, **Prepaid_Card** has the lowest recall (**0.0204** vs the overall **0.5495**), with a mean predicted probability among its actual returns of only **0.4185**. Non-COD payment methods lose the generator's `+0.9*(payment=="COD")` term entirely, pushing their true risk scores — and hence the model's predicted probabilities — well below the global threshold that is calibrated mostly on the COD-heavy majority of the training data.

**Specific fix:** add a `price_vs_category_median` feature — `price_inr` divided by the median `price_inr` **within that order's own `product_category`** — so, e.g., a relatively expensive `Electronics` order is flagged as a price outlier for its category even when its absolute price would look unremarkable in a category like Electronics; the raw `price_inr` column cannot express that relative signal on its own. Alternatively (or additionally), run `sweep_threshold()` independently on just the `Electronics` rows of the test split and adopt that subgroup's own F1-maximising cut point instead of inheriting the single global `t*_rf`, which is calibrated mostly on the higher-volume categories/payment methods. The same per-subgroup-threshold fix applies directly to `Prepaid_Card` on the payment-method side. Neither fix requires collecting more data — both are computable from the columns already in the dataset.

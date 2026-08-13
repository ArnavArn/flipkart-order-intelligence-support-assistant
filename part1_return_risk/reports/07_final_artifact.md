# 07 — Final Artifact

**Final model:** the tuned Random Forest — `grid.best_estimator_`, the full fitted sklearn Pipeline (preprocessing + `RandomForestClassifier`), **not** the Logistic Regression.

t\*_rf = **0.5000**, so `check_return_risk` buckets are: **Low** if p < 0.5000, **Medium** if 0.5000 <= p < 0.6500, **High** if p >= 0.6500.

- best params: `{'model__max_depth': 6, 'model__n_estimators': 200}`
- cv_roc_auc: **0.6193**
- test_roc_auc: **0.6203**
- t_star_logreg (F1-max, LogReg sweep, Task 5): **0.4400**

## Saved artifacts

- `models/return_risk_model.pkl` (joblib dump of the full RF Pipeline)
- `models/return_risk_meta.json`

```json
{
  "model": "RandomForestClassifier (GridSearchCV best) inside sklearn Pipeline",
  "best_params": {
    "model__max_depth": 6,
    "model__n_estimators": 200
  },
  "cv_roc_auc": 0.619257,
  "test_roc_auc": 0.620308,
  "t_star_rf": 0.5,
  "t_star_logreg": 0.44,
  "risk_buckets": {
    "low": "p < t_star_rf",
    "medium": "t_star_rf <= p < t_star_rf + 0.15",
    "high": "p >= t_star_rf + 0.15"
  },
  "bucket_cut_points": [
    0.5,
    0.65
  ],
  "numeric_features": [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
  ],
  "categorical_features": [
    "product_category",
    "payment_method"
  ],
  "sklearn_version": "1.9.0",
  "generated_by": "part1_return_risk/train.py"
}
```

## Reload verification

- **[PASS] joblib.load reproduces predict_proba to 1e-9** — in-memory = 0.537811244003, reloaded = 0.537811244003, |diff| = 0.00e+00

Sample row used for the spot-check (first row of the test split):

```
    price_inr  discount_pct  customer_tenure_days  num_previous_orders  num_previous_returns  delivery_distance_km  delivery_days  is_weekend_order  rating_given product_category payment_method
12     7181.0          36.1                    46                    0                     0                 104.3              5                 1           NaN             Home            COD
```

- In-memory model P(return=1): **0.537811244**
- Reloaded (`joblib.load`) model P(return=1): **0.537811244**
- Match to 1e-9: **True**

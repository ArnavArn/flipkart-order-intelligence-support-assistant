# Part 1 — Return-Risk Scoring Pipeline (35 marks)

**Entry point:** `python -m part1_return_risk.train`
**Outputs:** `models/return_risk_model.pkl`, `models/return_risk_meta.json`, 7 reports.

---

## Column contract (`part1_return_risk/config.py`)

```python
TARGET = "returned"
DROP   = ["order_id"]

NUMERIC = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
CATEGORICAL = ["product_category", "payment_method"]

SEED = 42
TEST_SIZE = 0.20
```

`is_weekend_order` is a 0/1 int — keep it in NUMERIC (scaling a binary is harmless and keeps
the ColumnTransformer simple). `rating_given` is the only column with missing values, but the
imputer is declared over all numerics anyway — that is what "no leakage" means in practice.

---

## Task 1 — Generate the dataset

Save the generator script **verbatim** as `generate_orders.py` at repo root. Do not touch
`np.random.default_rng(42)`, the category/payment lists, or any coefficient. Run:

```bash
python generate_orders.py
```

Commit both `generate_orders.py` and `orders_dataset.csv`.

**Sanity gate:** 6000 rows, 13 columns. The printed return rate must land in 18–27%.

---

## Task 2 — Verify the data → `reports/01_data_checks.md`

Compute and write out:

1. Total row count and column count.
2. Overall return rate.
3. `% missing rating_given` overall.
4. Return rate by `product_category` (5 rows) — as a markdown table.
5. Return rate by `payment_method` (4 rows) — as a markdown table.
6. **The MAR evidence table** — this one is required, not optional:

   | payment_method | n | n missing rating | % missing |
   |---|---|---|---|
   | COD | … | … | ~22% |
   | Prepaid_Card | … | … | ~6% |
   | Prepaid_UPI | … | … | ~6% |
   | Wallet | … | … | ~6% |

   plus a single computed line: `COD missing rate − non-COD missing rate = XX.X pp`.

### The MAR paragraph (must say all four things)

Write it in the report, generated with the real numbers interpolated:

- It is **MAR — missing at random, conditional on an observed column.**
- The observed column it is conditional on is **`payment_method`**.
- The **measured gap** between COD and non-COD missing rates, in percentage points, is the
  evidence.
- Why not the other two: **not MCAR**, because a genuine dependency on `payment_method` exists
  (the gap is far too large to be chance); **not MNAR**, because the missingness depends on
  `payment_method`, *not* on the unobserved rating value itself — the generator's mask is
  `rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)`, which never inspects
  `rating_given`.

Quote that generator line in the report. It is the strongest possible justification and the
brief explicitly hints at it ("its missingness depends on another observed column").

---

## Task 3 — Preprocess without leakage → `pipeline.py`

```python
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),   # do NOT drop a level
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
])
```

**Leakage rule:** the preprocessor is *only ever* a step inside a `Pipeline` that also holds the
estimator, and `.fit()` is called on `X_train` only. Never call `fit_transform` on the full
frame. `GridSearchCV` over the whole Pipeline then refits preprocessing inside each CV fold
automatically — that is the correct, leak-free pattern and is worth calling out in the report.

**Do not use `drop="first"`** on the OneHotEncoder. Dropping a level would remove
`payment_method_COD` — the single most important categorical signal — from the feature
importance table, and Task 7's acceptance criterion requires it to be there.

Split:
```python
train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
```

---

## Task 4 — Baseline → `reports/02_baseline_and_logreg.md`

`DummyClassifier(strategy="most_frequent")`. Report accuracy and F1 for class 1.

Expected: accuracy ≈ 1 − return_rate ≈ 0.76–0.80, **F1(class 1) = 0.0** (sklearn will emit a
zero-division warning — set `zero_division=0` and note it, that IS the finding).

**Paragraph must contain the exact phrase "high accuracy, zero recall."** Frame it as the
trainer did: 80 orders not returned, 20 returned; predict "not returned" for all 100 → 80%
accuracy, and *zero* of the returns caught. The model is useless for the business problem,
which is *catching returns before they happen*. Two of the five honest-evaluation rules are in
play: compare against a baseline, and pick metrics aligned to the business problem.

---

## Task 5 — Logistic Regression + threshold sweep

```python
LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
```

**At threshold 0.5**, report accuracy, F1, recall, precision, ROC-AUC (all for class 1).
Acceptance gate: **ROC-AUC ≥ 0.58 and F1 ≥ 0.30.** With `class_weight="balanced"` on this
generator, expect AUC ≈ 0.67–0.72 and F1 ≈ 0.42–0.48 — comfortable margin.

### `thresholds.py` — one shared function, used twice

```python
def sweep_threshold(y_true, proba, lo=0.10, hi=0.90, step=0.02):
    """Returns a DataFrame of threshold/precision/recall/f1 and the F1-maximising row."""
```

This exact function is called on the **LogReg** probabilities here (Task 5) and again on the
**Random Forest** probabilities in Task 9. Writing it once is what makes "re-run Task 5's
procedure on the RF" literally true rather than approximately true.

Write the full 41-row sweep to `reports/03_threshold_sweep_logreg.csv` and a condensed table +
an F1-vs-threshold plot (`reports/03_threshold_sweep_logreg.png`) — a plot is allowed, it's a
generated file inside the repo, not an uploaded image.

**Acceptance gate:** recall at t* must be **≥ 15 pp above** recall at 0.5. Report as:
`recall 0.5 → t*: 0.62 → 0.81 (+19.0 pp); precision 0.44 → 0.36 (−8.0 pp)`.

> If the sweep's F1-maximising threshold does *not* clear +15 pp (possible, because
> `class_weight="balanced"` already pushes recall up at 0.5), then **also report a
> recall-oriented operating point**: the lowest threshold whose precision stays above a stated
> floor, and use that as the "chosen threshold" with the trade-off paragraph written around it.
> Report both, and say which one you chose and why. Do not fake the number.

### Trade-off paragraph

Lowering the threshold flags more orders as likely returns → **higher recall, lower
precision**. The expensive error becomes the **false negative**: a return you failed to flag,
so no proactive intervention happened and the reverse-pickup cost lands anyway. You accept more
**false positives**: support time spent checking orders that were never going to be returned.
That is cheap relative to a missed return, which is precisely why 0.5 is a modelling default,
not a business decision. *Nobody decided 0.5 was magic.*

---

## Task 6 — Random Forest + GridSearchCV → `reports/04_random_forest_gridsearch.md`

```python
Pipeline([("prep", preprocessor),
          ("model", RandomForestClassifier(class_weight="balanced", random_state=42))])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None],
}
GridSearchCV(pipe, param_grid, scoring="roc_auc",
             cv=StratifiedKFold(5, shuffle=True, random_state=42),
             n_jobs=-1, refit=True)
```

Report: best params, best CV ROC-AUC, held-out test ROC-AUC.

**Acceptance gates:** best CV ROC-AUC ≥ 0.58, and |test AUC − CV AUC| ≤ 0.05. Also print the
full 6-row CV results table — it makes the "systematic experimentation, not guessing" point the
walkthrough hammered on.

If the gap exceeds 0.05, that is overfitting evidence: report it honestly, and note that
`max_depth=None` is the likely culprit — the shallower configurations in the grid are the fix.

---

## Task 7 — Explain the model → `reports/05_feature_importance.md`

### 7a. Impurity importance

```python
best = grid.best_estimator_
names = best.named_steps["prep"].get_feature_names_out()
imp   = best.named_steps["model"].feature_importances_
```

Report the **top 5** with a paragraph interpreting why each plausibly drives return risk.

**Acceptance requires** the top 5 to include `payment_method` (one-hot form, i.e.
`cat__payment_method_COD`) and ≥2 of `price_inr`, `customer_tenure_days`, `discount_pct`,
`num_previous_returns`.

> **Risk + mitigation.** One-hot encoding splits `payment_method` across 4 columns, diluting its
> impurity share, while continuous columns like `price_inr` hog splits. If
> `cat__payment_method_COD` does not land in the raw top 5, **additionally** report a
> **grouped** ranking that sums the one-hot columns back to their parent feature, and present
> both tables side by side. State which ranking each claim is made against. This is a standard,
> defensible treatment of one-hot importance — not a workaround — and it satisfies the criterion
> while being fully honest. Report the raw table first either way.

### 7b. Permutation importance — the decoy

```python
permutation_importance(best, X_test, y_test, n_repeats=10,
                       random_state=42, scoring="roc_auc", n_jobs=-1)
```

Note it runs on the **held-out test split** and on the **whole fitted pipeline** (so shuffling
happens on raw columns, pre-transform — which is what you want, since it keeps the
interpretation at the original-column level).

Present a **side-by-side table**: feature | impurity rank | impurity value | permutation rank |
permutation mean | permutation std.

**The finding you must name:** `delivery_distance_km` (and to a lesser degree `delivery_days`,
`price_inr`, `num_previous_orders`) ranks high on impurity but collapses toward ~0 under
permutation. Verify against the generator — `z` is:

```
z = -2.2 + 1.9*prev_return_ratio + 0.55*fit_risk_cat + 0.014*(discount_pct-20)/10
    + 0.9*(payment=="COD") + 0.10*(delivery_days-4.5)/2
    + 0.30*(price_inr/45000) + 0.05*is_weekend_order - 0.15*tanh(tenure/500)
```

`delivery_distance_km` **does not appear at all.** It is pure noise by construction. Say so, and
cite the generator.

**One-sentence explanation required:** impurity-based importance counts how often a feature is
*chosen* for a split and how much training-set impurity that split removes — a noisy continuous
column with thousands of distinct values offers enormous numbers of candidate split points, so
some of them will reduce training impurity by chance, inflating its score even though the
feature carries no real signal; permutation importance instead measures the actual drop in
**test-set** performance when the column is shuffled, so a noise feature scores ~0.

---

## Task 8 — Subgroup analysis → `reports/06_subgroup_analysis.md`

On the **test split**, using the RF at the **t\*_rf operating point** (state which threshold you
used — using t*_rf is the honest choice, because that is the threshold the deployed tool
actually operates at):

Table A — by `product_category`: n, support(returns), precision, recall, F1.
Table B — by `payment_method`: same columns.
Plus the overall row in each table for comparison.

**Then name a genuinely weaker subgroup** — read it off your own tables. The generator makes
`Beauty` and `Home` structurally weaker (no `fit_risk_cat` bonus, so their positive class is
rarer and driven almost entirely by `prev_return_ratio`), and non-COD prepaid orders lose the
+0.9 COD term. Expect recall to sag there.

**The fix must be specific.** Good options:
- *A category-specific threshold:* run the Task 5 sweep independently within `Beauty`'s test
  rows and adopt its own F1-maximising cut point (e.g. 0.31 for Beauty vs 0.44 globally), so
  the operating point matches that subgroup's probability distribution instead of inheriting a
  global one calibrated mostly on Apparel/Footwear volume.
- *An added feature:* `price_vs_category_median` (price relative to the category's own median),
  since a ₹2,000 Beauty order is an outlier while a ₹2,000 Electronics order is not — the raw
  `price_inr` column cannot express that.

Never write "collect more data."

---

## Task 9 — Save the artifact → `reports/07_final_artifact.md`

1. The final model is the **tuned Random Forest** — `grid.best_estimator_`, the full fitted
   Pipeline (preprocessing + model together). **Not** the Logistic Regression.
2. Re-run `sweep_threshold()` — the *same function* from Task 5 — on
   `best.predict_proba(X_test)[:, 1]`. The F1-maximising threshold is **`t*_rf`**.
3. `joblib.dump(best, "models/return_risk_model.pkl")`
4. Write `models/return_risk_meta.json`:

```json
{
  "model": "RandomForestClassifier (GridSearchCV best) inside sklearn Pipeline",
  "best_params": {"model__n_estimators": 200, "model__max_depth": 10},
  "cv_roc_auc": 0.0,
  "test_roc_auc": 0.0,
  "t_star_rf": 0.0,
  "t_star_logreg": 0.0,
  "risk_buckets": {"low": "p < t_star_rf",
                   "medium": "t_star_rf <= p < t_star_rf + 0.15",
                   "high": "p >= t_star_rf + 0.15"},
  "bucket_cut_points": [0.0, 0.0],
  "numeric_features": [...],
  "categorical_features": [...],
  "sklearn_version": "1.x.x",
  "generated_by": "part1_return_risk/train.py"
}
```

**This JSON is the contract with Part 3.** `t*_rf` must never be typed as a literal anywhere in
`part3_agent/`.

5. **Verification step, run at the end of `train.py`:** reload the pickle with `joblib.load`,
   predict on a fixed sample row, and assert the probability matches the in-memory model to 1e-9.
   Print that row + probability into the report — Part 3's transcript will show the same number
   coming out of the tool, which is exactly the spot-check the acceptance criteria describe.

Report `t*_rf` and the resulting cut points in one sentence, e.g.:

> t*_rf = 0.47, so `check_return_risk` buckets are: **Low** if p < 0.47, **Medium** if
> 0.47 ≤ p < 0.62, **High** if p ≥ 0.62.

---

## Part 1 acceptance self-check

- [ ] `orders_dataset.csv` exactly 6000 × 13
- [ ] return rate ∈ [18%, 27%]; `rating_given` missing ∈ [8%, 18%]
- [ ] MAR named, conditional on `payment_method`, COD/non-COD gap stated in pp, MCAR and MNAR
      each explicitly ruled out with a reason
- [ ] Dummy F1(class 1) = 0.0 reported; "high accuracy, zero recall" named
- [ ] LogReg @0.5: ROC-AUC ≥ 0.58, F1 ≥ 0.30
- [ ] threshold sweep step ≤ 0.02 over 0.1–0.9; chosen t* recall ≥ +15 pp vs default; precision
      drop stated numerically
- [ ] RF GridSearch best CV ROC-AUC ≥ 0.58; |test − CV| ≤ 0.05
- [ ] top-5 impurity includes `payment_method` one-hot + ≥2 of the four listed features
- [ ] permutation comparison present; ≥1 top-5 feature named as collapsing; one-sentence
      explanation of impurity bias present
- [ ] subgroup tables for both breakdowns, arithmetic correct, weak subgroup + specific fix
- [ ] `models/return_risk_model.pkl` loads, is the RF pipeline, and its `predict_proba` is what
      Part 3 calls
- [ ] `t*_rf` reported, computed on the RF's own probabilities

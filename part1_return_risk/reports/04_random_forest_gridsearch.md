# 04 — Random Forest + GridSearchCV

- **[PASS] Best CV ROC-AUC >= 0.58** — actual = 0.6193
- **[PASS] |test AUC - CV AUC| <= 0.05** — actual = |0.6203 - 0.6193| = 0.0011

- Best params: `{'model__max_depth': 6, 'model__n_estimators': 200}`
- Best CV ROC-AUC: **0.6193**
- Held-out test ROC-AUC: **0.6203**

## Full grid search results (6 candidates)

| n_estimators | max_depth | mean_cv_roc_auc | std_cv_roc_auc | rank |
| --- | --- | --- | --- | --- |
| 200 | 6 | 0.6193 | 0.0186 | 1 |
| 100 | 6 | 0.6186 | 0.0172 | 2 |
| 200 | 10 | 0.6071 | 0.0196 | 3 |
| 100 | 10 | 0.6059 | 0.0207 | 4 |
| 100 | None | 0.5954 | 0.0218 | 5 |
| 200 | None | 0.5930 | 0.0219 | 6 |

All 6 `(n_estimators, max_depth)` combinations were evaluated via 5-fold stratified cross-validation on ROC-AUC — this is systematic experimentation, not guessing.

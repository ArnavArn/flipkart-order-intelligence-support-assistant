# 02 — Baseline (Dummy) and Logistic Regression

## Task 4 — DummyClassifier(strategy='most_frequent')

- **[PASS] Dummy F1(class 1) == 0.0** — actual = 0.0000

- Accuracy: **0.7725**
- F1 (class 1, returned): **0.0000**  (sklearn emits a zero-division warning here — we set `zero_division=0` since the model never predicts class 1 at all; that absence of positive predictions *is* the finding, not a bug to silence away)

With 927 not-returned and 273 returned orders in the test split, predicting "not returned" for every single order yields **77.2% accuracy** but catches **zero** of the 273 actual returns. This is the textbook case of **high accuracy, zero recall** — the model is useless for the business problem, which is catching returns *before* they happen, not maximizing overall accuracy.

## Task 5 — LogisticRegression(class_weight='balanced') @ threshold 0.5

- **[PASS] ROC-AUC >= 0.58** — actual = 0.6253
- **[PASS] F1(class 1) >= 0.30** — actual = 0.3921

- Accuracy: **0.5917**
- Precision (class 1): **0.2964**
- Recall (class 1): **0.5788**
- F1 (class 1): **0.3921**
- ROC-AUC: **0.6253**

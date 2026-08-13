"""Shared threshold-sweep utility, called on both the LogReg and RF probabilities so both
models go through the exact same sweep procedure.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def sweep_threshold(y_true, proba, lo: float = 0.10, hi: float = 0.90, step: float = 0.02) -> pd.DataFrame:
    """Sweep decision thresholds over [lo, hi], scoring precision/recall/F1 for class 1."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 4)
    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1})

    return pd.DataFrame(rows)

"""Subgroup recall/precision/F1 by product_category and payment_method, at the deployed
operating point (t*_rf).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _row_metrics(y_true, y_pred) -> dict:
    n = len(y_true)
    support = int(np.sum(np.asarray(y_true) == 1))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"n": n, "support_returns": support, "precision": precision, "recall": recall, "f1": f1}


def subgroup_table(X_test: pd.DataFrame, y_test: pd.Series, proba: np.ndarray, threshold: float, group_col: str) -> pd.DataFrame:
    preds = (proba >= threshold).astype(int)
    df = X_test.copy()
    df["_y_true"] = np.asarray(y_test)
    df["_y_pred"] = preds

    rows = []
    for grp, sub in df.groupby(group_col):
        m = _row_metrics(sub["_y_true"], sub["_y_pred"])
        m[group_col] = grp
        rows.append(m)

    overall = _row_metrics(df["_y_true"], df["_y_pred"])
    overall[group_col] = "OVERALL"
    rows.append(overall)

    out = pd.DataFrame(rows)
    cols = [group_col, "n", "support_returns", "precision", "recall", "f1"]
    return out[cols].reset_index(drop=True)

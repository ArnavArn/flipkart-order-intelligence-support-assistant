"""Task 7 — impurity vs permutation importance.

Provides three views:
  1. raw_impurity_table   — one row per post-one-hot-encoding column (e.g. cat__payment_method_COD).
                            This is the table the acceptance gate checks (payment_method one-hot
                            form must appear in the top 5).
  2. grouped_impurity_table — one-hot columns summed back to their parent feature (payment_method,
                            product_category), so it lines up 1:1 with the permutation table, which
                            operates on raw (pre-transform) columns.
  3. permutation_table    — permutation_importance on the whole fitted Pipeline, evaluated on the
                            held-out test split, at the original-column level (n_repeats=10,
                            random_state=42, scoring="roc_auc").
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from part1_return_risk.config import CATEGORICAL, SEED


def raw_impurity_table(best) -> pd.DataFrame:
    names = best.named_steps["prep"].get_feature_names_out()
    imp = best.named_steps["model"].feature_importances_
    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _parent_of(feature_name: str) -> str:
    """Map a post-ColumnTransformer feature name back to its original raw column."""
    if feature_name.startswith("num__"):
        return feature_name[len("num__"):]
    if feature_name.startswith("cat__"):
        remainder = feature_name[len("cat__"):]
        for col in CATEGORICAL:
            if remainder.startswith(col + "_"):
                return col
        return remainder
    return feature_name


def grouped_impurity_table(best) -> pd.DataFrame:
    raw = raw_impurity_table(best)
    raw = raw.copy()
    raw["parent"] = raw["feature"].apply(_parent_of)
    grouped = raw.groupby("parent")["importance"].sum().reset_index()
    grouped = grouped.rename(columns={"parent": "feature"})
    grouped = grouped.sort_values("importance", ascending=False).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def permutation_table(best, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        best, X_test, y_test, n_repeats=10, random_state=SEED, scoring="roc_auc", n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_mean": result.importances_mean,
        "perm_std": result.importances_std,
    })
    df = df.sort_values("perm_mean", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def side_by_side_table(grouped_imp: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    merged = grouped_imp.rename(columns={"importance": "impurity_value", "rank": "impurity_rank"}).merge(
        perm.rename(columns={"rank": "permutation_rank"}),
        on="feature", how="outer",
    )
    merged = merged.sort_values("impurity_rank").reset_index(drop=True)
    return merged[["feature", "impurity_rank", "impurity_value", "permutation_rank", "perm_mean", "perm_std"]]

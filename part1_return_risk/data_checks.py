"""Task 2 — dataset verification: row/col counts, return rate, missingness, MAR evidence,
and breakdown tables by product_category and payment_method.

Every number returned here is computed from the actual loaded DataFrame — nothing is hand-typed.
"""
import pandas as pd

from part1_return_risk.config import DATA_PATH


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def basic_shape(df: pd.DataFrame) -> dict:
    return {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}


def overall_return_rate(df: pd.DataFrame) -> float:
    return float(df["returned"].mean())


def rating_missing_pct(df: pd.DataFrame) -> float:
    return float(df["rating_given"].isna().mean() * 100)


def return_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.groupby(col)["returned"].agg(n="count", return_rate="mean").reset_index()
    out["return_rate_pct"] = (out["return_rate"] * 100).round(2)
    return out.sort_values(col).reset_index(drop=True)


def mar_evidence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-rating rate broken down by payment_method — the MAR evidence table."""
    g = df.groupby("payment_method")["rating_given"].agg(
        n="count",  # non-null count via count(); we need total n and n missing separately below
    )
    total = df.groupby("payment_method").size().rename("n")
    n_missing = df.groupby("payment_method")["rating_given"].apply(lambda s: s.isna().sum()).rename("n_missing_rating")
    out = pd.concat([total, n_missing], axis=1).reset_index()
    out["pct_missing"] = (out["n_missing_rating"] / out["n"] * 100).round(2)
    return out.sort_values("payment_method").reset_index(drop=True)


def cod_vs_noncod_gap_pp(df: pd.DataFrame) -> float:
    """COD missing rate - non-COD missing rate, in percentage points."""
    cod_rate = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean() * 100
    noncod_rate = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean() * 100
    return float(cod_rate - noncod_rate)

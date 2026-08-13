"""check_return_risk(order_features: dict) -> dict

Loads Part 1's real saved artifacts:
    models/return_risk_model.pkl   -- joblib-dumped sklearn Pipeline {"prep", "model"}
    models/return_risk_meta.json   -- t_star_rf and bucket_cut_points, computed by Part 1

t_star_rf is read from the meta JSON at runtime, on every module import -- it is NEVER a
literal in this file. A fixed 0.3/0.6 pair of cut points would not be self-calibrating: a
differently-tuned-but-equally-valid RF could dump every real order into one bucket.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from part3_agent import config

_MODEL = None  # module-level lazy cache, loaded once per process

with open(config.RETURN_RISK_META_PATH, "r", encoding="utf-8") as _f:
    _META = json.load(_f)

T_STAR = _META["t_star_rf"]  # <- read from Part 1's artifact at runtime, never hardcoded

NUMERIC_FEATURES = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(config.RETURN_RISK_MODEL_PATH)
    return _MODEL


def _normalise(order_features: dict) -> dict:
    """Fill any of the 11 training feature columns the caller omitted with np.nan -- the
    pipeline's own median/mode imputer (fit inside the saved Pipeline) handles the rest.
    """
    row = {}
    for col in ALL_FEATURES:
        row[col] = order_features.get(col, np.nan)
    return row


def check_return_risk(order_features: dict) -> dict:
    """Score one order's return probability with Part 1's tuned RandomForest pipeline and map
    it to a Low/Medium/High bucket anchored to t_star_rf, read live from return_risk_meta.json.

    Bucket rule (verbatim from the brief): Low if p < t_star_rf; High if p >= t_star_rf + 0.15;
    Medium otherwise.
    """
    model = _load_model()
    row = pd.DataFrame([_normalise(order_features)])
    p = float(model.predict_proba(row)[0, 1])

    high_min = round(T_STAR + 0.15, 4)
    if p >= high_min:
        bucket = "High"
    elif p >= T_STAR:
        bucket = "Medium"
    else:
        bucket = "Low"

    return {
        "return_probability": round(p, 4),
        "risk_bucket": bucket,
        "t_star_rf": T_STAR,
        "cut_points": {"low_max": T_STAR, "high_min": high_min},
        "model": "RandomForest (Part 1, GridSearchCV-tuned pipeline)",
        "features_used": row.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    demo = {
        "price_inr": 1899, "discount_pct": 10, "customer_tenure_days": 12,
        "num_previous_orders": 3, "num_previous_returns": 1,
        "delivery_distance_km": 340, "delivery_days": 6, "is_weekend_order": 0,
        "rating_given": np.nan, "product_category": "Apparel", "payment_method": "COD",
    }
    print(check_return_risk(demo))

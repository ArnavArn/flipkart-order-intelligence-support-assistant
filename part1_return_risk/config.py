"""Column contract, paths, and split parameters for the Part 1 return-risk pipeline.

All paths are resolved relative to the repo root via Path(__file__).resolve().parents[1] —
no absolute paths are hardcoded, so this works regardless of where the repo is checked out.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = REPO_ROOT / "orders_dataset.csv"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "part1_return_risk" / "reports"

MODEL_PATH = MODELS_DIR / "return_risk_model.pkl"
META_PATH = MODELS_DIR / "return_risk_meta.json"

TARGET = "returned"
DROP = ["order_id"]

NUMERIC = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
CATEGORICAL = ["product_category", "payment_method"]

SEED = 42
TEST_SIZE = 0.20

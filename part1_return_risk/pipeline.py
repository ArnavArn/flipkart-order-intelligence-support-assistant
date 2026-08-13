"""Task 3 — build the leak-free preprocessing ColumnTransformer.

The preprocessor is only ever used as a step inside a Pipeline that also holds the estimator.
`.fit()` (via GridSearchCV or plain Pipeline.fit) must be called on X_train only — never call
`fit_transform` on the full frame before splitting. GridSearchCV then refits the preprocessing
inside each CV fold automatically, which is the correct, leak-free pattern.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from part1_return_risk.config import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),  # do NOT drop a level
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])
    return preprocessor

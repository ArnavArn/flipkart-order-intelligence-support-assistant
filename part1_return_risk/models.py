"""Tasks 4/5/6 — dummy baseline, logistic regression, and the Random Forest GridSearchCV.

Every estimator that touches raw columns is wrapped in a sklearn Pipeline together with the
preprocessor from pipeline.py, so `.fit()` on X_train never leaks information from X_test.
"""
from typing import Tuple

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from part1_return_risk.config import CATEGORICAL, DROP, NUMERIC, SEED, TARGET, TEST_SIZE
from part1_return_risk.pipeline import build_preprocessor


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = NUMERIC + CATEGORICAL
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def fit_dummy(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """DummyClassifier wrapped in the same Pipeline shape for API consistency."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", DummyClassifier(strategy="most_frequent", random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_logreg(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_rf_gridsearch(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=SEED)),
    ])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 10, None],
    }
    grid = GridSearchCV(
        pipe, param_grid, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train)
    return grid

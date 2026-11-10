"""Preprocessing pipeline (replaces the manual get_dummies / replace steps in the notebook).

Fitting the encoder/scaler inside a Pipeline avoids train/test leakage and guarantees
the exact same transformation at prediction time.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

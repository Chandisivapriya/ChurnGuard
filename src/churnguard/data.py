"""Loading, validating and cleaning the Telco churn dataset."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [c for c in REQUIRED_COLUMNS if c not in NUMERIC_FEATURES]


class SchemaError(ValueError):
    """Raised when input data does not match the expected schema."""


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature columns (used for both training and prediction)."""
    validate_schema(df)
    out = df[REQUIRED_COLUMNS].copy()
    # TotalCharges arrives as text with blank strings for brand-new customers (tenure == 0).
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out.loc[out["tenure"] == 0, "TotalCharges"] = out.loc[out["tenure"] == 0, "TotalCharges"].fillna(0.0)
    # Collapse "No internet service" / "No phone service" into "No" (same meaning as the notebook).
    out = out.replace({"No internet service": "No", "No phone service": "No"})
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(str)
    return out


def prepare_training_data(df: pd.DataFrame, target: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with y encoded as 1 = churned."""
    if target not in df.columns:
        raise SchemaError(f"Target column '{target}' not found")
    X = clean_features(df)
    y = df[target].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise SchemaError("Target contains values other than Yes/No")
    keep = X["TotalCharges"].notna()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True).astype(int)

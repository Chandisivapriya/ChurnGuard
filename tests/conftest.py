import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset with the Telco schema (the real CSV is not committed)."""
    rng = np.random.default_rng(0)
    n = 400
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.2, 0.2])
    churn_p = 0.1 + 0.4 * (contract == "Month-to-month") - 0.003 * tenure
    churn = np.where(rng.random(n) < np.clip(churn_p, 0.02, 0.9), "Yes", "No")
    yn = lambda: rng.choice(["Yes", "No"], n)  # noqa: E731
    total = (tenure * monthly).round(2).astype(str)
    total = np.where(tenure == 0, " ", total)  # the real data has blank TotalCharges
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yn(), "Dependents": yn(), "tenure": tenure, "PhoneService": yn(),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": yn(), "DeviceProtection": yn(), "TechSupport": yn(),
        "StreamingTV": yn(), "StreamingMovies": yn(),
        "Contract": contract, "PaperlessBilling": yn(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
    })

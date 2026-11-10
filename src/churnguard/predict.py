"""Load the saved artefact and score customers."""
from __future__ import annotations

import joblib
import pandas as pd

from .data import clean_features


class ChurnPredictor:
    def __init__(self, model_path: str):
        artefact = joblib.load(model_path)
        self.pipeline = artefact["pipeline"]
        self.threshold = artefact["threshold"]
        self.model_name = artefact["model_name"]

    def predict(self, records: list[dict]) -> list[dict]:
        X = clean_features(pd.DataFrame(records))
        proba = self.pipeline.predict_proba(X)[:, 1]
        return [
            {"churn_probability": round(float(p), 4), "will_churn": bool(p >= self.threshold)}
            for p in proba
        ]

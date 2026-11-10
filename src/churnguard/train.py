"""Training CLI: compare candidate models, tune the threshold, save the best pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import load_config
from .data import load_raw, prepare_training_data
from .evaluate import best_threshold, compute_metrics
from .features import build_preprocessor
from .models import get_model


def train(cfg: dict, data_path: str | None = None) -> dict:
    df = load_raw(data_path or cfg["data"]["raw_path"])
    X, y = prepare_training_data(df, cfg["data"]["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"], stratify=y,
    )

    results, fitted = {}, {}
    for name in cfg["model"]["candidates"]:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", get_model(name, cfg["split"]["random_state"]))])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, proba, 0.5)
        fitted[name] = (pipe, proba)

    metric = cfg["model"]["selection_metric"]
    best = max(results, key=lambda n: results[n][metric])
    pipe, proba = fitted[best]
    cost = cfg["cost"]
    threshold = best_threshold(y_test, proba, cost["false_negative"], cost["false_positive"])
    tuned = compute_metrics(y_test, proba, threshold)

    report = {"best_model": best, "threshold": threshold, "candidates_at_0.5": results, "best_tuned": tuned}
    out = cfg["output"]
    Path(out["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold, "model_name": best}, out["model_path"])
    Path(out["metrics_path"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="Override path to the raw CSV")
    args = parser.parse_args()
    report = train(load_config(args.config), args.data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

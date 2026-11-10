import json

from churnguard.config import load_config
from churnguard.predict import ChurnPredictor
from churnguard.train import train


def test_end_to_end_training_and_prediction(synthetic_df, tmp_path):
    csv = tmp_path / "telco.csv"
    synthetic_df.to_csv(csv, index=False)
    cfg = load_config()
    cfg["output"] = {
        "model_path": str(tmp_path / "model.joblib"),
        "metrics_path": str(tmp_path / "metrics.json"),
    }
    report = train(cfg, str(csv))
    assert report["best_model"] in cfg["model"]["candidates"]
    assert json.loads((tmp_path / "metrics.json").read_text())["threshold"] == report["threshold"]

    predictor = ChurnPredictor(cfg["output"]["model_path"])
    rows = synthetic_df.drop(columns=["customerID", "Churn"]).head(3).to_dict("records")
    preds = predictor.predict(rows)
    assert len(preds) == 3
    assert all(0.0 <= p["churn_probability"] <= 1.0 for p in preds)

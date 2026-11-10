# ChurnGuard

A customer churn prediction service built on the Telco Customer Churn dataset. It started as a
notebook experiment with a Keras ANN and is being turned into a tested, deployable application:
a training pipeline, cost-aware evaluation, and a REST API.

## Why not just accuracy?
Missing a customer who is about to leave costs far more than sending an unnecessary retention
offer. ChurnGuard compares models on PR-AUC and then picks the decision threshold that minimises
a configurable business cost (`configs/config.yaml`), instead of using a fixed 0.5.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[api,dev]"

# 1. Download Telco-Customer-Churn.csv from Kaggle into data/raw/
# 2. Train and compare models
churnguard-train
# 3. Serve predictions
uvicorn churnguard.api:app --reload
# 4. Run tests
pytest
```

## API
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Active model and decision threshold |
| POST | `/predict` | Score one customer |
| POST | `/predict/batch` | Score a list of customers |

## Project layout
```
src/churnguard/   data cleaning, feature pipeline, models, evaluation, training CLI, API
tests/            pytest suite (uses synthetic data, no download needed)
configs/          model, split and cost settings
```

## Roadmap
- [ ] Port the original Keras ANN as the `keras_ann` candidate (validation split, class weights)
- [ ] SHAP explanations: why is this customer at risk?
- [ ] Streamlit dashboard for the retention team
- [ ] Store predictions in a SQL database and query the top at-risk customers
- [ ] Data drift check (PSI) on incoming batches
- [ ] Deploy the Docker image to Azure Container Apps

## Original notebook
The initial exploration is kept in `notebooks/` for reference.

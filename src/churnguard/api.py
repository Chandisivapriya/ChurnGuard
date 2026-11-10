"""REST API. Run with: uvicorn churnguard.api:app --reload"""
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import SchemaError
from .predict import ChurnPredictor

MODEL_PATH = os.getenv("CHURNGUARD_MODEL", "models/churn_model.joblib")
YesNo = Literal["Yes", "No"]

app = FastAPI(title="ChurnGuard", version="0.1.0")
_predictor: ChurnPredictor | None = None


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: str
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = ChurnPredictor(MODEL_PATH)
        except FileNotFoundError as exc:
            raise HTTPException(503, "Model not trained yet. Run churnguard-train first.") from exc
    return _predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    p = get_predictor()
    return {"model": p.model_name, "threshold": p.threshold}


@app.post("/predict")
def predict(customer: Customer) -> dict:
    try:
        return get_predictor().predict([customer.model_dump()])[0]
    except SchemaError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]) -> list[dict]:
    if not customers:
        raise HTTPException(422, "Empty batch")
    return get_predictor().predict([c.model_dump() for c in customers])

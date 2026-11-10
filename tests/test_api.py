import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from churnguard.api import app  # noqa: E402


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_predict_rejects_invalid_payload():
    assert TestClient(app).post("/predict", json={"tenure": -5}).status_code == 422

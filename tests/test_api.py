import pytest

testclient = pytest.importorskip("fastapi.testclient")


def _client():
    from churnguard.api import app

    return testclient.TestClient(app)


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_predict_rejects_invalid_payload():
    assert _client().post("/predict", json={"tenure": -5}).status_code == 422

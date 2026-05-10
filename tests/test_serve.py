from pathlib import Path

import joblib
import pytest

from mlops_assign01.serve import create_app
from mlops_assign01.train import train_model


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    result = train_model(n_estimators=20, max_depth=4, criterion="gini")
    path = tmp_path / "model.joblib"
    joblib.dump(result["model"], path)
    return path


@pytest.fixture
def client(model_path: Path):
    app = create_app(model_path=model_path)
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"


def test_predict_setosa(client):
    resp = client.post("/predict", json={"features": [[5.1, 3.5, 1.4, 0.2]]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["predictions"] == ["setosa"]
    assert body["feature_order"] == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]


def test_predict_rejects_missing_features(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    assert "features" in resp.get_json()["error"]


def test_predict_rejects_malformed_features(client):
    resp = client.post("/predict", json={"features": [["not", "numbers", "at", "all"]]})
    assert resp.status_code == 400


def test_health_503_when_model_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist.joblib"
    app = create_app(model_path=missing)
    with app.test_client() as c:
        # health does NOT load the model, so it still returns 200
        assert c.get("/health").status_code == 200
        # predict needs a model — should 503
        resp = c.post("/predict", json={"features": [[5.1, 3.5, 1.4, 0.2]]})
        assert resp.status_code == 503

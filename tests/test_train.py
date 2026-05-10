from mlops_assign01.train import train_model


def test_train_model_returns_metrics_and_fitted_model():
    result = train_model(n_estimators=20, max_depth=4, criterion="gini")

    assert "model" in result
    assert "metrics" in result

    metrics = result["metrics"]
    for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    # Iris is easy — even a small forest should comfortably clear 0.8 accuracy
    assert metrics["accuracy"] > 0.8


def test_train_model_is_deterministic_given_random_state():
    a = train_model(n_estimators=20, max_depth=4, criterion="gini", random_state=7)
    b = train_model(n_estimators=20, max_depth=4, criterion="gini", random_state=7)
    assert a["metrics"] == b["metrics"]

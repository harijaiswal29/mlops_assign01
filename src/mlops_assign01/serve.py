"""Flask service exposing /health and /predict for the trained Iris model."""

from __future__ import annotations

import os
from pathlib import Path

import click
import joblib
import pandas as pd
from flask import Flask, jsonify, request

from . import __version__
from .data import DEFAULT_MODEL_PATH, FEATURE_COLUMNS

MODEL_PATH_ENV = "MODEL_PATH"


def _resolve_model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH)))


def create_app(model_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MODEL_PATH"] = model_path or _resolve_model_path()
    app.config["MODEL"] = None

    def _load_model():
        if app.config["MODEL"] is None:
            path = app.config["MODEL_PATH"]
            if not path.exists():
                raise FileNotFoundError(
                    f"Model not found at {path}. Train first: python -m mlops_assign01.train"
                )
            app.config["MODEL"] = joblib.load(path)
        return app.config["MODEL"]

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            version=__version__,
            model_path=str(app.config["MODEL_PATH"]),
            model_loaded=app.config["MODEL"] is not None,
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        features = payload.get("features")
        if features is None:
            return jsonify(error="missing 'features' in request body"), 400
        try:
            model = _load_model()
        except FileNotFoundError as exc:
            return jsonify(error=str(exc)), 503

        try:
            frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
            preds = model.predict(frame).tolist()
        except (ValueError, TypeError) as exc:
            return jsonify(error=f"prediction failed: {exc}"), 400

        return jsonify(
            predictions=preds,
            model_version=__version__,
            feature_order=FEATURE_COLUMNS,
        )

    return app


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--model-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override model path; defaults to $MODEL_PATH or models/best_model.joblib.",
)
def main(host: str, port: int, model_path: Path | None) -> None:
    """Run the Flask dev server (production uses gunicorn — see Dockerfile)."""
    app = create_app(model_path=model_path)
    app.run(host=host, port=port)


# Module-level WSGI handle for gunicorn: ``gunicorn mlops_assign01.serve:app``
app = create_app()


if __name__ == "__main__":
    main()

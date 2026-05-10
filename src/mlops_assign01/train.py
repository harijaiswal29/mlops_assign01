"""MLflow-instrumented training entrypoint."""

from __future__ import annotations

from pathlib import Path

import click
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .data import (
    DEFAULT_MODEL_PATH,
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_iris_df,
    read_csv,
)


def _load_data(csv_path: Path | None) -> pd.DataFrame:
    if csv_path is not None and csv_path.exists():
        return read_csv(csv_path)
    return load_iris_df()


def train_model(
    *,
    n_estimators: int,
    max_depth: int | None,
    criterion: str,
    random_state: int = 42,
    test_size: float = 0.2,
    csv_path: Path | None = None,
) -> dict:
    """Train a RandomForest on Iris and return the fitted model plus metrics."""
    df = _load_data(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
    }
    return {"model": model, "metrics": metrics}


@click.command()
@click.option("--n-estimators", type=int, default=100, show_default=True)
@click.option("--max-depth", type=int, default=None, show_default=True)
@click.option(
    "--criterion",
    type=click.Choice(["gini", "entropy", "log_loss"]),
    default="gini",
    show_default=True,
)
@click.option("--random-state", type=int, default=42, show_default=True)
@click.option(
    "--csv-path",
    type=click.Path(path_type=Path),
    default=Path("data/iris.csv"),
    show_default=True,
    help="CSV path; falls back to sklearn loader if missing.",
)
@click.option(
    "--model-out",
    type=click.Path(path_type=Path),
    default=DEFAULT_MODEL_PATH,
    show_default=True,
)
@click.option(
    "--tracking-uri",
    default="file:./mlruns",
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--run-name",
    default=None,
    help="Friendly MLflow run name.",
)
def main(
    n_estimators: int,
    max_depth: int | None,
    criterion: str,
    random_state: int,
    csv_path: Path,
    model_out: Path,
    tracking_uri: str,
    run_name: str | None,
) -> None:
    """Train one model run, log to MLflow, and save the artifact."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "criterion": criterion,
                "random_state": random_state,
            }
        )
        result = train_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
            csv_path=csv_path,
        )
        mlflow.log_metrics(result["metrics"])
        mlflow.sklearn.log_model(result["model"], artifact_path="model")

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], model_out)
        mlflow.log_artifact(str(model_out), artifact_path="serialised")

        click.echo(f"Run {run.info.run_id} — metrics: {result['metrics']}")
        click.echo(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()

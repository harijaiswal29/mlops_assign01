"""Optuna hyperparameter tuning with MLflow tracking."""

from __future__ import annotations

from pathlib import Path

import click
import joblib
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .data import (
    DEFAULT_MODEL_PATH,
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_iris_df,
    read_csv,
)


def _objective_factory(X, y, random_state: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
        model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
        return float(scores.mean())

    return objective


@click.command()
@click.option("--n-trials", type=int, default=30, show_default=True)
@click.option("--random-state", type=int, default=42, show_default=True)
@click.option(
    "--csv-path",
    type=click.Path(path_type=Path),
    default=Path("data/iris.csv"),
    show_default=True,
)
@click.option(
    "--model-out",
    type=click.Path(path_type=Path),
    default=DEFAULT_MODEL_PATH,
    show_default=True,
)
@click.option("--tracking-uri", default="file:./mlruns", show_default=True)
def main(
    n_trials: int,
    random_state: int,
    csv_path: Path,
    model_out: Path,
    tracking_uri: str,
) -> None:
    """Run an Optuna study; log every trial to MLflow; persist the best model."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = read_csv(csv_path) if csv_path.exists() else load_iris_df()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name="f1_macro")
    study = optuna.create_study(direction="maximize", study_name="iris-rf-tuning")
    study.optimize(
        _objective_factory(X, y, random_state),
        n_trials=n_trials,
        callbacks=[mlflc],
        show_progress_bar=False,
    )

    best = study.best_params
    best_score = study.best_value
    click.echo(f"Best f1_macro={best_score:.4f} with params={best}")

    final_model = RandomForestClassifier(**best, random_state=random_state, n_jobs=-1)
    final_model.fit(X, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_out)
    click.echo(f"Saved best model to {model_out}")

    with mlflow.start_run(run_name="optuna-best") as run:
        mlflow.log_params(best)
        mlflow.log_metric("best_f1_macro_cv", best_score)
        mlflow.sklearn.log_model(final_model, artifact_path="model")
        mlflow.log_artifact(str(model_out), artifact_path="serialised")
        click.echo(f"Best model run id: {run.info.run_id}")


if __name__ == "__main__":
    main()

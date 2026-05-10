"""Iris dataset loader.

Exposes :func:`load_iris_df` for in-process use and a CLI that materialises
the dataset to ``data/iris.csv`` so it can be version-controlled with DVC.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from sklearn.datasets import load_iris

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"

EXPERIMENT_NAME = "iris-classifier"
DEFAULT_MODEL_PATH = Path("models/best_model.joblib")


def load_iris_df() -> pd.DataFrame:
    """Return the Iris dataset as a pandas DataFrame with named columns."""
    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(dict(enumerate(bunch.target_names)))
    return df


def read_csv(path: str | Path) -> pd.DataFrame:
    """Load the Iris CSV produced by :func:`load_iris_df`."""
    return pd.read_csv(path)


@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/iris.csv"),
    show_default=True,
    help="Where to write the CSV.",
)
@click.option(
    "--drop-rows",
    type=int,
    default=0,
    show_default=True,
    help="Drop the last N rows — useful for the DVC revert demo.",
)
def main(out: Path, drop_rows: int) -> None:
    """Materialise the Iris dataset to a CSV file."""
    df = load_iris_df()
    if drop_rows > 0:
        df = df.iloc[:-drop_rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    click.echo(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

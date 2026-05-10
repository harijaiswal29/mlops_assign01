from pathlib import Path

from mlops_assign01.data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_df, read_csv


def test_load_iris_df_shape_and_columns():
    df = load_iris_df()
    assert len(df) == 150
    assert list(df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert set(df[TARGET_COLUMN].unique()) == {"setosa", "versicolor", "virginica"}


def test_round_trip_csv(tmp_path: Path):
    df = load_iris_df()
    csv_path = tmp_path / "iris.csv"
    df.to_csv(csv_path, index=False)

    loaded = read_csv(csv_path)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)

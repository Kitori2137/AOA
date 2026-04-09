import numpy as np
import pandas as pd
import pytest

from AOA.core.evaluation import (
    append_metrics_row,
    calculate_classification_metrics,
    calculate_regression_metrics,
    fill_missing_values,
    transform_numeric_columns,
)


def test_fill_missing_values_fills_numeric_with_mean_and_text_with_ffill_or_unknown():
    df = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0],
            "txt": [None, "A", None],
            "txt2": [None, None, "B"],
        }
    )

    out = fill_missing_values(df)

    assert out["num"].iloc[1] == pytest.approx(2.0)
    assert out["txt"].tolist() == ["Unknown", "A", "A"]
    assert out["txt2"].tolist() == ["Unknown", "Unknown", "B"]


@pytest.mark.parametrize(
    ("transformation", "check"),
    [
        (
            "MinMax Normalizacja",
            lambda out: out["a"].min() >= 0 and out["a"].max() <= 1 and out["b"].min() >= 0 and out["b"].max() <= 1,
        ),
        (
            "Standaryzacja",
            lambda out: abs(out["a"].mean()) < 1e-6 and abs(out["b"].mean()) < 1e-6,
        ),
        (
            "Logarytm",
            lambda out: np.allclose(out["a"].to_numpy(), np.log1p([1.0, 2.0, 4.0])),
        ),
        (
            "Skalowanie 0-1",
            lambda out: out["a"].min() >= 0 and out["a"].max() <= 1 and out["b"].min() >= 0 and out["b"].max() <= 1,
        ),
    ],
)
def test_transform_numeric_columns_supported_modes(transformation, check):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 4.0],
            "b": [10.0, 20.0, 30.0],
            "txt": ["x", "y", "z"],
        }
    )

    out = transform_numeric_columns(df, transformation)

    assert check(out)
    assert out["txt"].tolist() == ["x", "y", "z"]


def test_transform_numeric_columns_returns_same_df_when_no_numeric_columns():
    df = pd.DataFrame({"txt": ["a", "b"]})

    out = transform_numeric_columns(df, "Standaryzacja")

    pd.testing.assert_frame_equal(out, df)


def test_calculate_regression_metrics_returns_expected_keys():
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [2, 4, 6, 8, 10, 12],
            "target": [3, 6, 9, 12, 15, 18],
        }
    )

    metrics = calculate_regression_metrics(df, "target")

    assert set(metrics) == {"R2", "MAE", "MSE", "RMSE", "MAPE_%"}
    assert all(isinstance(v, (int, float, np.floating)) for v in metrics.values())


def test_calculate_regression_metrics_requires_numeric_target():
    df = pd.DataFrame({"f1": [1, 2, 3], "target": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="Target musi być kolumną liczbową do regresji"):
        calculate_regression_metrics(df, "target")


def test_calculate_regression_metrics_requires_feature_columns():
    df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Brak cech liczbowych do regresji"):
        calculate_regression_metrics(df, "target")


def test_calculate_classification_metrics_returns_expected_keys():
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [10, 11, 12, 13, 14, 15],
            "target": ["A", "A", "B", "B", "C", "C"],
        }
    )

    metrics = calculate_classification_metrics(df, "target")

    assert set(metrics) == {"Accuracy", "F1_score", "Precision", "Recall"}
    assert all(isinstance(v, (int, float, np.floating)) for v in metrics.values())


def test_calculate_classification_metrics_requires_categorical_target():
    df = pd.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})

    with pytest.raises(ValueError, match="Target musi być kolumną kategorialną do klasyfikacji"):
        calculate_classification_metrics(df, "target")


def test_calculate_classification_metrics_requires_numeric_features():
    df = pd.DataFrame({"target": ["A", "B", "A"]})

    with pytest.raises(ValueError, match="Brak cech liczbowych do klasyfikacji"):
        calculate_classification_metrics(df, "target")


def test_append_metrics_row_adds_last_row_with_metrics():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    metrics = {"A": "R2", "B": 0.99}

    out = append_metrics_row(df, metrics)

    assert len(out) == 3
    assert out.iloc[-1]["A"] == "R2"
    assert out.iloc[-1]["B"] == 0.99

import pandas as pd
import pytest

from AOA.core.services import (
    build_dataframe_preview_text,
    build_main_page_status,
    build_main_page_summary,
    prepare_results_analysis,
)


def test_build_main_page_summary_contains_selected_options():
    config = {
        "selected_models": ["Quality", "Schedule"],
        "selected_ksztalty": ["kwadrat", "trapez"],
        "selected_materialy": ["bawelna", "wiskoza"],
        "n": "5000",
        "n_machines": "2",
        "test_size": "0.2",
        "seed": "42",
        "prod_min": "1",
        "prod_max": "48",
        "deadline_min": "1",
        "deadline_max": "72",
    }

    summary = build_main_page_summary(config)

    assert "Quality, Schedule" in summary
    assert "kwadrat, trapez" in summary
    assert "bawelna, wiskoza" in summary
    assert "Liczba rekordów: 5000" in summary


def test_build_main_page_status_without_data():
    status = build_main_page_status(None, None)
    assert status == "Brak danych treningowych"


def test_build_main_page_status_with_data():
    train_df = pd.DataFrame({"a": range(8)})
    test_df = pd.DataFrame({"a": range(2)})

    status = build_main_page_status(train_df, test_df)

    assert "Train: 8 rekordów" in status
    assert "Test: 2 rekordów" in status


def test_build_dataframe_preview_text_contains_title_and_shape():
    df = pd.DataFrame({
        "cena": [100, 200],
        "odpad": [0.1, 0.2],
    })

    text = build_dataframe_preview_text(df, title="Podgląd testowy", max_rows=5)

    assert "Podgląd testowy" in text
    assert "Liczba rekordów: 2" in text
    assert "Liczba kolumn: 2" in text


def test_prepare_results_analysis_regression_returns_dataframe_and_text():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 3, 4, 5, 6],
        "target": [10, 20, 30, 40, 50],
    })

    result = prepare_results_analysis(
        df=df,
        selected_cols=["feature1", "feature2", "target"],
        transformation="Surowe",
        target="target",
        mode="regresja",
    )

    assert "df" in result
    assert "text" in result
    assert "R2" in result["text"] or "MAE" in result["text"]


def test_prepare_results_analysis_invalid_mode_raises():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "target": [10, 20, 30],
    })

    with pytest.raises(ValueError):
        prepare_results_analysis(
            df=df,
            selected_cols=["feature1", "target"],
            transformation="Surowe",
            target="target",
            mode="nieznany_tryb",
        )

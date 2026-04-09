from matplotlib.figure import Figure
import pandas as pd
import pytest

from AOA.core.visualization_service import build_figure_from_request


def _df():
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 15, 30],
            "z": [5, 6, 7, 8],
            "t_start": [0, 1, 3, 6],
            "t_end": [1, 3, 6, 10],
            "czas_produkcji_h": [1, 2, 3, 4],
            "termin_h": [2, 4, 7, 12],
            "ksztalt": ["kwadrat", "trojkat", "trapez", "romb"],
        }
    )


@pytest.mark.parametrize(
    "chart_type,kwargs",
    [
        ("Scatter", {"x_col": "x", "y_col": "y"}),
        ("Line", {"x_col": "x", "y_col": "y"}),
        ("Histogram", {"x_col": "x"}),
        ("Boxplot", {"y_col": "y"}),
        ("Gantt", {}),
        ("CorrelationMatrix", {}),
        ("SimilarityMatrix", {}),
        ("DecisionTree", {}),
    ],
)
def test_build_figure_from_request_supported_chart_types(chart_type, kwargs):
    fig = build_figure_from_request(_df(), chart_type, **kwargs)

    assert isinstance(fig, Figure)


def test_build_figure_from_request_rejects_empty_df():
    with pytest.raises(ValueError, match="Brak danych do wizualizacji"):
        build_figure_from_request(pd.DataFrame(), "Scatter", x_col="x", y_col="y")


def test_build_figure_from_request_rejects_bad_scatter_columns():
    with pytest.raises(ValueError, match="Nieprawidłowe kolumny dla wykresu Scatter"):
        build_figure_from_request(_df(), "Scatter", x_col="bad", y_col="y")


def test_build_figure_from_request_rejects_bad_histogram_column():
    with pytest.raises(ValueError, match="Nieprawidłowa kolumna dla histogramu"):
        build_figure_from_request(_df(), "Histogram", x_col="bad")


def test_build_figure_from_request_rejects_bad_boxplot_column():
    with pytest.raises(ValueError, match="Nieprawidłowa kolumna dla boxplot"):
        build_figure_from_request(_df(), "Boxplot", y_col="bad")


def test_build_figure_from_request_rejects_unsupported_chart_type():
    with pytest.raises(ValueError, match="Nieobsługiwany typ wykresu"):
        build_figure_from_request(_df(), "Pie")

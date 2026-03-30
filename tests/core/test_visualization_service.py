from matplotlib.figure import Figure
import pandas as pd

from AOA.core.visualization_service import build_figure_from_request


def test_build_figure_from_request_scatter_returns_figure():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
    })

    fig = build_figure_from_request(df, chart_type="Scatter", x_col="x", y_col="y")

    assert isinstance(fig, Figure)


def test_build_figure_from_request_line_returns_figure():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
    })

    fig = build_figure_from_request(df, chart_type="Line", x_col="x", y_col="y")

    assert isinstance(fig, Figure)

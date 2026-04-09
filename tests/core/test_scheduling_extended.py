import pandas as pd

from AOA.core.scheduling import (
    extract_schedule_features,
    generate_schedule_label,
    optimize_schedule,
    simulate_schedule,
)


def _df():
    return pd.DataFrame(
        {
            "czas_produkcji_h": [5.0, 1.0, 2.0],
            "termin_h": [100.0, 2.0, 4.0],
            "cena": [10.0, 20.0, 30.0],
        }
    )


def test_optimize_schedule_returns_best_strategy_with_required_columns():
    result = optimize_schedule(_df())

    assert {"t_start", "t_end", "lateness_h", "strategy", "total_time", "total_lateness"}.issubset(result.columns)
    assert result["strategy"].iloc[0] in {"EDF", "SPT", "LPT", "Slack"}


def test_simulate_schedule_preserves_input_order_and_adds_time_columns():
    df = _df()
    result = simulate_schedule(df)

    pd.testing.assert_series_equal(result["czas_produkcji_h"], df["czas_produkcji_h"], check_names=False)
    assert result["t_start"].tolist() == [0.0, 5.0, 6.0]
    assert result["t_end"].tolist() == [5.0, 6.0, 8.0]
    assert result["lateness_h"].tolist() == [0.0, 4.0, 4.0]


def test_extract_schedule_features_returns_expected_keys():
    features = extract_schedule_features(_df())

    assert set(features) == {"n_jobs", "mean_time", "std_time", "max_time", "mean_deadline", "load_ratio"}
    assert features["n_jobs"] == 3


def test_generate_schedule_label_returns_known_strategy_name():
    label = generate_schedule_label(_df())

    assert label in {"EDF", "SPT", "LPT", "Slack"}

import numpy as np
import pandas as pd
import pytest

from AOA.core import services
from AOA.core.models import save_model_pack


class DummyModel:
    def __init__(self, values):
        self.values = values

    def predict(self, X):
        if callable(self.values):
            return np.asarray(self.values(X), dtype=float)
        return np.asarray(self.values, dtype=float)


@pytest.fixture
def valid_generation_config():
    return {
        "n": "100",
        "n_machines": "2",
        "test_size": "0.2",
        "seed": "42",
        "prod_min": "1",
        "prod_max": "48",
        "deadline_min": "2",
        "deadline_max": "72",
        "selected_ksztalty": ["kwadrat", "trapez"],
        "selected_materialy": ["bawelna", "wiskoza"],
    }


@pytest.fixture
def solution_input_df():
    return pd.DataFrame(
        {
            "cena": [100.0, 150.0, 120.0],
            "odpad": [0.10, 0.20, 0.05],
            "termin_h": [12.0, 10.0, 24.0],
            "czas_produkcji_h": [4.0, 8.0, 6.0],
            "x": [10.0, 12.0, 14.0],
            "y": [5.0, 6.0, 7.0],
            "z": [1.0, 1.0, 1.0],
            "ksztalt": ["kwadrat", "trapez", "trojkat"],
            "material": ["bawelna", "wiskoza", "mikrofibra"],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", "0"),
        ("n", "-5"),
        ("n_machines", "0"),
        ("seed", "0"),
        ("prod_min", "0"),
        ("deadline_min", "-1"),
    ],
)
def test_parse_generation_config_rejects_non_positive_values(
    valid_generation_config, field, value
):
    config = valid_generation_config.copy()
    config[field] = value

    with pytest.raises(ValueError, match="Głuptasie"):
        services.parse_generation_config(config)


@pytest.mark.parametrize("test_size", ["0", "-0.1"])
def test_parse_generation_config_rejects_non_positive_test_size(
    valid_generation_config, test_size
):
    config = valid_generation_config.copy()
    config["test_size"] = test_size

    with pytest.raises(ValueError, match="Głuptasie"):
        services.parse_generation_config(config)


@pytest.mark.parametrize("test_size", ["1", "1.5"])
def test_parse_generation_config_rejects_test_size_greater_or_equal_one(
    valid_generation_config, test_size
):
    config = valid_generation_config.copy()
    config["test_size"] = test_size

    with pytest.raises(ValueError, match="Test size musi być mniejsze od 1"):
        services.parse_generation_config(config)


def test_parse_generation_config_rejects_prod_min_greater_than_prod_max(
    valid_generation_config,
):
    config = valid_generation_config.copy()
    config["prod_min"] = "50"
    config["prod_max"] = "10"

    with pytest.raises(
        ValueError,
        match="Minimalny czas produkcji nie może być większy niż maksymalny",
    ):
        services.parse_generation_config(config)


def test_parse_generation_config_rejects_deadline_min_greater_than_deadline_max(
    valid_generation_config,
):
    config = valid_generation_config.copy()
    config["deadline_min"] = "80"
    config["deadline_max"] = "10"

    with pytest.raises(
        ValueError,
        match="Minimalny bufor terminu nie może być większy niż maksymalny",
    ):
        services.parse_generation_config(config)


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("selected_ksztalty", "Wybierz przynajmniej jeden kształt"),
        ("selected_materialy", "Wybierz przynajmniej jeden materiał"),
    ],
)
def test_parse_generation_config_requires_shape_and_material_selection(
    valid_generation_config, field, expected_message
):
    config = valid_generation_config.copy()
    config[field] = []

    with pytest.raises(ValueError, match=expected_message):
        services.parse_generation_config(config)


def test_parse_generation_config_returns_parsed_values(valid_generation_config):
    parsed = services.parse_generation_config(valid_generation_config)

    assert parsed["n"] == 100
    assert parsed["n_machines"] == 2
    assert parsed["test_size"] == 0.2
    assert parsed["seed"] == 42
    assert parsed["prod_min"] == 1.0
    assert parsed["prod_max"] == 48.0
    assert parsed["deadline_min"] == 2.0
    assert parsed["deadline_max"] == 72.0
    assert parsed["selected_ksztalty"] == ["kwadrat", "trapez"]
    assert parsed["selected_materialy"] == ["bawelna", "wiskoza"]


def test_solve_models_flow_predicts_quality_delay_saves_csv_and_sorts_priority(
    tmp_path, monkeypatch, solution_input_df
):
    data_path = tmp_path / "input.csv"
    solution_input_df.to_csv(data_path, index=False)

    pack_path = tmp_path / "pack.pkl"
    pack = {
        "pack_kind": "ml",
        "quality": DummyModel([0.2, 0.9, 0.5]),
        "delay": DummyModel([5.0, 1.0, 2.0]),
        "schedule": None,
        "scaler": None,
        "selected_models": ["Quality", "Delay"],
    }
    save_model_pack(pack, pack_path)

    result_file = tmp_path / "wynik.csv"

    monkeypatch.setattr(
        services,
        "build_result_filename",
        lambda model_name, source_name, suffix=".csv": result_file,
    )

    result = services.solve_models_flow(pack_path, data_path)

    assert result_file.exists()
    assert list(result["df"]["pred_quality"]) == [0.9, 0.5, 0.2]
    assert list(result["df"]["pred_delay"]) == [1.0, 2.0, 5.0]
    assert "priority" in result["df"].columns
    assert result["df"]["priority"].is_monotonic_decreasing
    saved_df = pd.read_csv(result_file)
    assert "pred_quality" in saved_df.columns
    assert "pred_delay" in saved_df.columns
    assert any("Rozwiązanie gotowe" in message for message in result["messages"])


def test_solve_models_flow_handles_only_quality_model(
    tmp_path, monkeypatch, solution_input_df
):
    data_path = tmp_path / "input.csv"
    solution_input_df.to_csv(data_path, index=False)

    pack_path = tmp_path / "pack.pkl"
    save_model_pack(
        {
            "pack_kind": "ml",
            "quality": DummyModel([0.3, 0.4, 0.5]),
            "delay": None,
            "schedule": None,
            "scaler": None,
            "selected_models": ["Quality"],
        },
        pack_path,
    )

    result_file = tmp_path / "wynik.csv"

    monkeypatch.setattr(
        services,
        "build_result_filename",
        lambda model_name, source_name, suffix=".csv": result_file,
    )

    result = services.solve_models_flow(pack_path, data_path)

    assert "pred_quality" in result["df"].columns
    assert "pred_delay" not in result["df"].columns
    assert "priority" not in result["df"].columns
    assert result_file.exists()


@pytest.mark.parametrize(
    ("model_path", "data_path", "message"),
    [
        (None, "data.csv", "Nie wybrano pliku modelu"),
        ("pack.pkl", None, "Nie wybrano pliku danych"),
    ],
)
def test_solve_models_flow_requires_paths(model_path, data_path, message):
    with pytest.raises(ValueError, match=message):
        services.solve_models_flow(model_path, data_path)


def test_solve_models_flow_rejects_csv_with_headers_but_no_rows(tmp_path):
    data_path = tmp_path / "empty_rows.csv"
    pack_path = tmp_path / "pack.pkl"

    pd.DataFrame(
        columns=[
            "cena",
            "odpad",
            "termin_h",
            "czas_produkcji_h",
            "x",
            "y",
            "z",
            "ksztalt",
            "material",
        ]
    ).to_csv(data_path, index=False)

    save_model_pack(
        {
            "pack_kind": "ml",
            "quality": None,
            "delay": None,
            "schedule": None,
            "scaler": None,
            "selected_models": [],
        },
        pack_path,
    )

    with pytest.raises(ValueError, match="Plik danych jest pusty"):
        services.solve_models_flow(pack_path, data_path)

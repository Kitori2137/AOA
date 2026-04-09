from pathlib import Path

import pandas as pd
import pytest

from AOA.core import services


def _raw_config(**overrides):
    cfg = {
        "n": "100",
        "n_machines": "2",
        "test_size": "0.2",
        "seed": "42",
        "prod_min": "1.0",
        "prod_max": "10.0",
        "deadline_min": "2.0",
        "deadline_max": "20.0",
        "selected_ksztalty": ["kwadrat", "trapez"],
        "selected_materialy": ["bawelna", "poliester"],
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("field", ["n", "n_machines", "seed"])
def test_parse_generation_config_rejects_zero_or_negative_ints(field):
    bad = _raw_config(**{field: "0"})
    with pytest.raises(ValueError, match="ujemne rzeczy|dodatnie"):
        services.parse_generation_config(bad)

    bad = _raw_config(**{field: "-5"})
    with pytest.raises(ValueError, match="ujemne rzeczy|dodatnie"):
        services.parse_generation_config(bad)


@pytest.mark.parametrize("field", ["prod_min", "prod_max", "deadline_min", "deadline_max"])
def test_parse_generation_config_rejects_zero_or_negative_floats(field):
    bad = _raw_config(**{field: "0"})
    with pytest.raises(ValueError, match="ujemne rzeczy|dodatnie"):
        services.parse_generation_config(bad)

    bad = _raw_config(**{field: "-1.5"})
    with pytest.raises(ValueError, match="ujemne rzeczy|dodatnie"):
        services.parse_generation_config(bad)


@pytest.mark.parametrize("value", ["1", "1.0", "2"])
def test_parse_generation_config_rejects_test_size_greater_or_equal_one(value):
    with pytest.raises(ValueError, match="Test size musi być mniejsze od 1"):
        services.parse_generation_config(_raw_config(test_size=value))


@pytest.mark.parametrize("value", ["0", "-0.1"])
def test_parse_generation_config_rejects_non_positive_test_size(value):
    with pytest.raises(ValueError, match="ujemne rzeczy|dodatnie"):
        services.parse_generation_config(_raw_config(test_size=value))


def test_parse_generation_config_rejects_prod_min_greater_than_prod_max():
    with pytest.raises(ValueError, match="Minimalny czas produkcji nie może być większy"):
        services.parse_generation_config(_raw_config(prod_min="12", prod_max="10"))


def test_parse_generation_config_rejects_deadline_min_greater_than_deadline_max():
    with pytest.raises(ValueError, match="Minimalny bufor terminu nie może być większy"):
        services.parse_generation_config(_raw_config(deadline_min="30", deadline_max="20"))


def test_parse_generation_config_requires_selected_shapes_and_materials():
    with pytest.raises(ValueError, match="Wybierz przynajmniej jeden kształt"):
        services.parse_generation_config(_raw_config(selected_ksztalty=[]))

    with pytest.raises(ValueError, match="Wybierz przynajmniej jeden materiał"):
        services.parse_generation_config(_raw_config(selected_materialy=[]))


def test_generate_and_store_datasets_saves_three_csv_files(monkeypatch, tmp_path):
    df_full = pd.DataFrame({"a": [1, 2, 3]})
    df_train = pd.DataFrame({"a": [1, 2]})
    df_test = pd.DataFrame({"a": [3]})

    monkeypatch.setattr(services, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        services,
        "generate_production_data",
        lambda **kwargs: (df_full, df_train, df_test),
    )

    saved = []

    def fake_save_csv(df, path):
        saved.append((df.copy(), Path(path)))
        Path(path).write_text(df.to_csv(index=False), encoding="utf-8")

    monkeypatch.setattr(services, "save_csv", fake_save_csv)

    result = services.generate_and_store_datasets()

    assert result["full_path"].exists()
    assert result["train_path"].exists()
    assert result["test_path"].exists()
    assert len(saved) == 3
    assert result["full_df"].equals(df_full)
    assert result["train_df"].equals(df_train)
    assert result["test_df"].equals(df_test)


def test_load_and_prepare_visual_file_returns_defaults(monkeypatch):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    monkeypatch.setattr(services, "load_csv", lambda path: df)

    result = services.load_and_prepare_visual_file("dummy.csv")

    assert result["columns"] == ["x", "y", "z"]
    assert result["x_default"] == "x"
    assert result["y_default"] == "y"
    pd.testing.assert_frame_equal(result["df"], df)


def test_load_and_prepare_visual_file_raises_for_empty_columns(monkeypatch):
    monkeypatch.setattr(services, "load_csv", lambda path: pd.DataFrame())

    with pytest.raises(ValueError, match="Plik CSV nie zawiera kolumn"):
        services.load_and_prepare_visual_file("dummy.csv")


def test_load_training_data_returns_full_train_test(monkeypatch):
    df_full = pd.DataFrame({"a": range(10)})
    df_train = pd.DataFrame({"a": range(8)})
    df_test = pd.DataFrame({"a": range(8, 10)})

    monkeypatch.setattr(services, "load_csv", lambda path: df_full)
    monkeypatch.setattr(services, "split_train_test", lambda df, train_ratio=0.8: (df_train, df_test))

    result = services.load_training_data("dummy.csv", train_ratio=0.8)

    assert result["full_df"].equals(df_full)
    assert result["train_df"].equals(df_train)
    assert result["test_df"].equals(df_test)
    assert any("Wczytano dane" in msg for msg in result["messages"])


def test_train_models_flow_rejects_empty_df():
    with pytest.raises(ValueError, match="Brak danych treningowych"):
        services.train_models_flow(pd.DataFrame(), ["Quality"])


def test_train_models_flow_rejects_missing_selected_models():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Nie wybrano żadnego modelu do trenowania"):
        services.train_models_flow(df, [])


def test_train_models_flow_saves_model_pack(monkeypatch, tmp_path):
    df_train = pd.DataFrame({"a": [1, 2, 3]})
    fake_pack = {"quality": object(), "delay": None, "schedule": None, "scaler": object(), "selected_models": ["Quality"]}
    target_path = tmp_path / "pack.pkl"

    monkeypatch.setattr(services, "train_selected_models", lambda **kwargs: fake_pack)
    monkeypatch.setattr(services, "build_model_filename", lambda selected_models, metadata: target_path)

    saved = {}

    def fake_save_model_pack(pack, path):
        saved["pack"] = pack
        saved["path"] = path

    monkeypatch.setattr(services, "save_model_pack", fake_save_model_pack)

    result = services.train_models_flow(df_train, ["Quality"], metadata={"n": 3})

    assert result["model_pack"] is fake_pack
    assert result["model_path"] == target_path
    assert saved["pack"] is fake_pack
    assert saved["path"] == target_path


def test_solve_models_flow_rejects_missing_paths():
    with pytest.raises(ValueError, match="Nie wybrano pliku modelu"):
        services.solve_models_flow("", "data.csv")

    with pytest.raises(ValueError, match="Nie wybrano pliku danych"):
        services.solve_models_flow("model.pkl", "")


def test_solve_models_flow_rejects_empty_data(monkeypatch):
    monkeypatch.setattr(services, "load_model_pack", lambda path: {})
    monkeypatch.setattr(services, "load_csv", lambda path: pd.DataFrame())

    with pytest.raises(ValueError, match="Plik danych jest pusty"):
        services.solve_models_flow("model.pkl", "data.csv")


def test_build_model_filename_contains_sorted_model_names_and_metadata(monkeypatch, tmp_path):
    class FixedNow:
        @staticmethod
        def now():
            class X:
                @staticmethod
                def strftime(fmt):
                    return "20260409_140000"
            return X()

    monkeypatch.setattr(services, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(services, "datetime", FixedNow)

    path = services.build_model_filename(
        ["Delay", "Quality"],
        {
            "n": 500,
            "n_machines": 3,
            "ksztalty": ["kwadrat", "trapez"],
            "materialy": ["bawelna"],
        },
    )

    assert path.parent == tmp_path
    assert "delay-quality" in path.name
    assert "500r" in path.name
    assert "3m" in path.name
    assert "kwadrat-trapez" in path.name
    assert "bawelna" in path.name
    assert path.suffix == ".pkl"


def test_sanitize_filename_replaces_invalid_chars():
    name = 'model a<b>:c"/d\\e|f?g*h.pkl'
    cleaned = services.sanitize_filename(name)

    for bad in ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ']:
        assert bad not in cleaned
    assert cleaned.endswith(".pkl")

from pathlib import Path

import pandas as pd
import pytest

from AOA.core import services
from AOA.core.models import load_model_pack, save_model_pack


class PickleableDummyModel:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def training_df():
    return pd.DataFrame(
        {
            "cena": [100, 120, 130, 150],
            "odpad": [0.1, 0.2, 0.15, 0.05],
            "termin_h": [10, 12, 14, 16],
            "czas_produkcji_h": [3, 5, 4, 6],
            "x": [10, 12, 8, 15],
            "y": [5, 6, 4, 7],
            "z": [1, 1, 1, 1],
            "ksztalt": ["kwadrat", "trapez", "trojkat", "kwadrat"],
            "material": ["bawelna", "wiskoza", "mikrofibra", "poliester"],
        }
    )


def test_save_and_load_model_pack_roundtrip_preserves_structure(tmp_path):
    pack = {
        "quality": PickleableDummyModel("quality"),
        "delay": PickleableDummyModel("delay"),
        "schedule": None,
        "scaler": {"type": "mock-scaler"},
        "selected_models": ["Quality", "Delay"],
        "backend": "classic",
    }

    path = tmp_path / "model_pack.pkl"
    save_model_pack(pack, path)
    loaded = load_model_pack(path)

    assert path.exists()
    assert set(loaded.keys()) == {
        "quality",
        "delay",
        "schedule",
        "scaler",
        "selected_models",
        "backend",
    }
    assert loaded["quality"].name == "quality"
    assert loaded["delay"].name == "delay"
    assert loaded["schedule"] is None
    assert loaded["scaler"] == {"type": "mock-scaler"}
    assert loaded["selected_models"] == ["Quality", "Delay"]
    assert loaded["backend"] == "classic"


def test_train_models_flow_saves_pack_with_generated_filename(
    tmp_path,
    monkeypatch,
    training_df,
):
    expected_path = tmp_path / "trained_pack.pkl"
    fake_pack = {
        "quality": PickleableDummyModel("quality"),
        "delay": None,
        "schedule": None,
        "scaler": {"type": "mock-scaler"},
        "selected_models": ["Quality"],
        "backend": "tabpfn",
    }

    monkeypatch.setattr(
        services,
        "build_model_filename",
        lambda *_args, **_kwargs: expected_path,
    )
    monkeypatch.setattr(services, "train_selected_models", lambda **_kwargs: fake_pack)

    result = services.train_models_flow(
        training_df,
        selected_models=["Quality"],
        metadata={"n": 4, "n_machines": 1},
        backend="tabpfn",
    )

    assert result["model_path"] == expected_path
    assert expected_path.exists()

    loaded = load_model_pack(expected_path)
    assert loaded["quality"].name == "quality"
    assert loaded["selected_models"] == ["Quality"]
    assert loaded["backend"] == "tabpfn"
    assert any("Trening zakończony" in message for message in result["messages"])
    assert any("backend: TabPFN" in message for message in result["messages"])


@pytest.mark.parametrize(
    ("df_train", "selected_models", "message"),
    [
        (None, ["Quality"], "Brak danych treningowych"),
        (pd.DataFrame(), ["Quality"], "Brak danych treningowych"),
        (pd.DataFrame({"a": [1]}), [], "Nie wybrano żadnego modelu do trenowania"),
    ],
)
def test_train_models_flow_validates_inputs(df_train, selected_models, message):
    with pytest.raises(ValueError, match=message):
        services.train_models_flow(df_train, selected_models=selected_models)

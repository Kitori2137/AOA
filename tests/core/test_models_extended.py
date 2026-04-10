import numpy as np
import pandas as pd
import pytest

from AOA.core import models


def _sample_training_df(n=40):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "cena": rng.uniform(50, 200, n),
            "odpad": rng.uniform(0.01, 0.3, n),
            "termin_h": rng.uniform(10, 80, n),
            "czas_produkcji_h": rng.uniform(1, 20, n),
            "x": rng.uniform(1, 30, n),
            "y": rng.uniform(1, 30, n),
            "z": rng.uniform(0.1, 5, n),
            "ksztalt": rng.choice(["kwadrat", "trojkat", "trapez"], n),
            "material": rng.choice(
                ["bawelna", "mikrofibra", "poliester", "wiskoza"],
                n,
            ),
            "lateness_h_sim": rng.uniform(0, 15, n),
        }
    )


def test_train_selected_models_returns_only_requested_quality():
    df = _sample_training_df()
    pack = models.train_selected_models(df, ["Quality"])

    assert pack["quality"] is not None
    assert pack["delay"] is None
    assert pack["schedule"] is None
    assert pack["scaler"] is not None
    assert pack["selected_models"] == ["Quality"]
    assert pack["backend"] == "classic"


def test_train_selected_models_returns_all_requested_models():
    df = _sample_training_df()
    pack = models.train_selected_models(df, ["Quality", "Delay", "Schedule"])

    assert pack["quality"] is not None
    assert pack["delay"] is not None
    assert pack["schedule"] is not None
    assert pack["scaler"] is not None
    assert pack["selected_models"] == ["Quality", "Delay", "Schedule"]
    assert pack["backend"] == "classic"


def test_train_schedule_model_calls_progress_callback():
    df = _sample_training_df()
    progress_calls = []

    def progress_callback(*args):
        progress_calls.append(args)

    model = models.train_schedule_model(df, n_samples=20, progress_callback=progress_callback)

    assert model is not None
    assert progress_calls, "Callback postępu nie został wywołany"
    flattened = " | ".join(" ".join(map(str, call)) for call in progress_calls)
    assert "Schedule" in flattened
    assert any(
        "100" in " ".join(map(str, call)) or "Zakończono" in " ".join(map(str, call))
        for call in progress_calls
    )


def test_train_selected_models_rejects_unknown_backend():
    df = _sample_training_df()

    with pytest.raises(ValueError, match="Nieznany backend modeli"):
        models.train_selected_models(df, ["Quality"], backend="cosmos")


def test_train_selected_models_uses_tabpfn_backend_when_requested(monkeypatch):
    df = _sample_training_df()

    monkeypatch.setattr(models, "train_tabpfn_regressor", lambda X, y: {"kind": "tabpfn_reg"})
    monkeypatch.setattr(models, "train_tabpfn_classifier", lambda X, y: {"kind": "tabpfn_clf"})

    pack = models.train_selected_models(
        df,
        ["Quality", "Delay", "Schedule"],
        backend="tabpfn",
    )

    assert pack["quality"] == {"kind": "tabpfn_reg"}
    assert pack["delay"] == {"kind": "tabpfn_reg"}
    assert pack["schedule"] == {"kind": "tabpfn_clf"}
    assert pack["backend"] == "tabpfn"

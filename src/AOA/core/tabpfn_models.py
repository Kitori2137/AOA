from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    TABPFN_AVAILABLE = True
    TABPFN_IMPORT_ERROR = None
except Exception as e:
    TabPFNClassifier = None
    TabPFNRegressor = None
    TABPFN_AVAILABLE = False
    TABPFN_IMPORT_ERROR = e


class TabPFNNotAvailableError(RuntimeError):
    """Raised when TabPFN backend was requested but dependency is unavailable."""


def ensure_tabpfn_available() -> None:
    if not TABPFN_AVAILABLE:
        raise TabPFNNotAvailableError(
            "TabPFN nie jest zainstalowany. "
            f"Błąd importu: {TABPFN_IMPORT_ERROR}"
        )


def _prepare_env_for_cpu_large_dataset() -> None:
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"


def _build_tabpfn_regressor():
    ensure_tabpfn_available()
    _prepare_env_for_cpu_large_dataset()

    errors = []

    for kwargs in (
        {"ignore_pretraining_limits": True},
        {"device": "cpu", "ignore_pretraining_limits": True},
        {},
        {"device": "cpu"},
    ):
        try:
            return TabPFNRegressor(**kwargs)
        except TypeError as e:
            errors.append(f"{kwargs} -> TypeError: {e}")
        except Exception as e:
            errors.append(f"{kwargs} -> {type(e).__name__}: {e}")

    raise RuntimeError(
        "Nie udało się utworzyć TabPFNRegressor.\n"
        + "\n".join(errors)
    )


def _build_tabpfn_classifier():
    ensure_tabpfn_available()
    _prepare_env_for_cpu_large_dataset()

    errors = []

    for kwargs in (
        {"ignore_pretraining_limits": True},
        {"device": "cpu", "ignore_pretraining_limits": True},
        {},
        {"device": "cpu"},
    ):
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError as e:
            errors.append(f"{kwargs} -> TypeError: {e}")
        except Exception as e:
            errors.append(f"{kwargs} -> {type(e).__name__}: {e}")

    raise RuntimeError(
        "Nie udało się utworzyć TabPFNClassifier.\n"
        + "\n".join(errors)
    )


def train_tabpfn_regressor(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Any:
    model = _build_tabpfn_regressor()

    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise RuntimeError(
            "TabPFNRegressor.fit() zakończył się błędem.\n"
            f"Typ błędu: {type(e).__name__}\n"
            f"Szczegóły: {e}"
        ) from e


def train_tabpfn_classifier(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Any:
    model = _build_tabpfn_classifier()

    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise RuntimeError(
            "TabPFNClassifier.fit() zakończył się błędem.\n"
            f"Typ błędu: {type(e).__name__}\n"
            f"Szczegóły: {e}"
        ) from e

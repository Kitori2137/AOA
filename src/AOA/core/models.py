import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from AOA.core.features import prepare_features
from AOA.core.scheduling import extract_schedule_features, generate_schedule_label


def _emit_progress(progress_callback, model_name: str, percent: float, detail: str = ""):
    if progress_callback is None:
        return

    try:
        progress_callback(model_name, percent, detail)
        return
    except TypeError:
        pass

    try:
        progress_callback(model_name, percent)
        return
    except TypeError:
        pass

    try:
        progress_callback(percent)
    except TypeError:
        return


def _train_quality_with_progress(X_train, yq, progress_callback=None):
    total_estimators = 200
    model = RandomForestRegressor(
        n_estimators=1,
        warm_start=True,
        random_state=42,
    )

    _emit_progress(progress_callback, "Quality", 0.0, "Start")
    _emit_progress(progress_callback, "Quality", 0.5, "Przygotowanie danych")

    for i in range(1, total_estimators + 1):
        model.n_estimators = i
        model.fit(X_train, yq)

        percent = min(100.0, 0.5 + (i / total_estimators) * 99.0)
        _emit_progress(
            progress_callback,
            "Quality",
            round(percent, 1),
            f"Drzewo {i}/{total_estimators}",
        )

    _emit_progress(progress_callback, "Quality", 100.0, "Zakończono")
    return model


def _train_delay_with_progress(X_train, yd, progress_callback=None):
    total_estimators = 200
    model = GradientBoostingRegressor(
        n_estimators=1,
        warm_start=True,
        random_state=42,
    )

    _emit_progress(progress_callback, "Delay", 0.0, "Start")
    _emit_progress(progress_callback, "Delay", 0.5, "Przygotowanie danych")

    for i in range(1, total_estimators + 1):
        model.n_estimators = i
        model.fit(X_train, yd)

        percent = min(100.0, 0.5 + (i / total_estimators) * 99.0)
        _emit_progress(
            progress_callback,
            "Delay",
            round(percent, 1),
            f"Estimator {i}/{total_estimators}",
        )

    _emit_progress(progress_callback, "Delay", 100.0, "Zakończono")
    return model


def train_schedule_model(df, n_samples=200, progress_callback=None):
    _emit_progress(progress_callback, "Schedule", 0.0, "Start")

    X = []
    y = []

    for i in range(n_samples):
        batch_size = np.random.randint(5, len(df))
        batch = df.sample(n=batch_size, replace=False)
        X.append(extract_schedule_features(batch))
        y.append(generate_schedule_label(batch))

        percent = ((i + 1) / n_samples) * 100.0
        _emit_progress(
            progress_callback,
            "Schedule",
            round(percent, 1),
            f"Próbka {i + 1}/{n_samples}",
        )

    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    _emit_progress(progress_callback, "Schedule", 100.0, "Zakończono")
    return model


def train_selected_models(df_train, selected_models, progress_callback=None):
    if not selected_models:
        raise ValueError("Nie wybrano żadnego modelu do trenowania")

    X_train, yq, yd, scaler = prepare_features(df_train)

    quality_model = None
    delay_model = None
    schedule_model = None

    if "Quality" in selected_models:
        quality_model = _train_quality_with_progress(
            X_train,
            yq,
            progress_callback=progress_callback,
        )

    if "Delay" in selected_models:
        delay_model = _train_delay_with_progress(
            X_train,
            yd,
            progress_callback=progress_callback,
        )

    if "Schedule" in selected_models:
        schedule_model = train_schedule_model(
            df_train,
            progress_callback=progress_callback,
        )

    return {
        "quality": quality_model,
        "delay": delay_model,
        "schedule": schedule_model,
        "scaler": scaler,
        "selected_models": selected_models,
    }


def save_model_pack(model_pack, path):
    with open(path, "wb") as f:
        pickle.dump(model_pack, f)


def load_model_pack(path):
    with open(path, "rb") as f:
        return pickle.load(f)

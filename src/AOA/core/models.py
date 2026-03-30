import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor

from AOA.core.features import prepare_features
from AOA.core.scheduling import extract_schedule_features, generate_schedule_label


def train_schedule_model(df, n_samples=200, progress_callback=None):
    X = []
    y = []

    for i in range(n_samples):
        batch = df.sample(n=np.random.randint(5, len(df)), replace=False)
        X.append(extract_schedule_features(batch))
        y.append(generate_schedule_label(batch))

        if progress_callback:
            progress_callback((i + 1) / n_samples * 100)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


def train_selected_models(df_train, selected_models, progress_callback=None):
    if not selected_models:
        raise ValueError("Nie wybrano żadnego modelu do trenowania")

    X_train, yq, yd, scaler = prepare_features(df_train)

    quality_model = None
    delay_model = None
    schedule_model = None

    if "Quality" in selected_models:
        quality_model = RandomForestRegressor(n_estimators=300, random_state=42)
        quality_model.fit(X_train, yq)

    if "Delay" in selected_models:
        delay_model = GradientBoostingRegressor(n_estimators=300, random_state=42)
        delay_model.fit(X_train, yd)

    if "Schedule" in selected_models:
        schedule_model = train_schedule_model(df_train, progress_callback=progress_callback)

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

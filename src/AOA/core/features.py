import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from AOA.core.constants import FEATURES, KSZTALT_MAP, MATERIAL_MAP


def prepare_features(df: pd.DataFrame, scaler_obj=None):
    df = df.copy()

    df["ksztalt_q"] = df["ksztalt"].map(KSZTALT_MAP).fillna(0.8)
    df["material_q"] = df["material"].map(MATERIAL_MAP).fillna(0.8)

    df["odpad_norm"] = df["odpad"] / (df["odpad"].max() + 1e-6)
    df["czas_na_deadline"] = df["czas_produkcji_h"] / (df["termin_h"] + 1e-6)
    df["koszt_czasu"] = df["cena"] * df["czas_produkcji_h"]
    df["presja"] = 1 / (df["termin_h"] + 1e-6)

    df["pole"] = 0.0
    df.loc[df["ksztalt"] == "kwadrat", "pole"] = df["x"] * df["y"]
    df.loc[df["ksztalt"] == "trojkat", "pole"] = 0.5 * df["x"] * df["y"]
    df.loc[df["ksztalt"] == "trapez", "pole"] = 0.5 * (df["x"] + df["y"]) * df["y"]

    X = df[
        FEATURES + [
            "x", "y", "z", "pole", "ksztalt_q", "material_q",
            "odpad_norm", "czas_na_deadline", "koszt_czasu", "presja"
        ]
    ]

    y_quality = ((1 - df["odpad_norm"]) * df["material_q"] * df["ksztalt_q"]).clip(0, 1)
    y_delay = df["czas_na_deadline"] * (2 - df["ksztalt_q"])

    if scaler_obj is None:
        scaler_obj = MinMaxScaler()
        X_scaled = scaler_obj.fit_transform(X)
    else:
        X_scaled = scaler_obj.transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y_quality, y_delay, scaler_obj

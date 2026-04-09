import pandas as pd
import pytest

from AOA.core.features import prepare_features


def _sample_df():
    return pd.DataFrame(
        {
            "cena": [100.0, 150.0, 200.0],
            "odpad": [0.10, 0.20, 0.05],
            "termin_h": [10.0, 12.0, 8.0],
            "czas_produkcji_h": [5.0, 4.0, 3.0],
            "ksztalt": ["kwadrat", "trojkat", "trapez"],
            "material": ["bawelna", "mikrofibra", "poliester"],
            "x": [2.0, 4.0, 6.0],
            "y": [3.0, 5.0, 2.0],
            "z": [1.0, 1.5, 2.0],
        }
    )


def test_prepare_features_returns_scaled_features_and_targets():
    X, y_quality, y_delay, scaler = prepare_features(_sample_df())

    assert len(X) == 3
    assert len(y_quality) == 3
    assert len(y_delay) == 3
    assert scaler is not None
    assert X.min().min() >= 0
    assert X.max().max() <= 1


def test_prepare_features_computes_shape_area_for_supported_shapes():
    X, _, _, _ = prepare_features(_sample_df())

    assert "pole" in X.columns

    pole_idx = list(X.columns).index("pole")
    # Po skalowaniu kolejność nadal odpowiada wejściu, więc porównujemy relacje
    assert X.iloc[0, pole_idx] != X.iloc[1, pole_idx]
    assert X.iloc[1, pole_idx] != X.iloc[2, pole_idx]


def test_prepare_features_maps_unknown_shape_and_material_to_fallback_value():
    df = _sample_df()
    df.loc[0, "ksztalt"] = "hex"
    df.loc[0, "material"] = "kevlar"

    X, _, _, _ = prepare_features(df)

    assert "ksztalt_q" in X.columns
    assert "material_q" in X.columns
    assert X.notna().all().all()


def test_prepare_features_reuses_existing_scaler():
    df = _sample_df()
    X1, _, _, scaler = prepare_features(df)
    X2, _, _, same_scaler = prepare_features(df, scaler_obj=scaler)

    assert same_scaler is scaler
    pd.testing.assert_frame_equal(X1, X2)

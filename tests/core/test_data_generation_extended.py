import pandas as pd
import pytest

from AOA.core.data_generation import generate_production_data


def test_generate_production_data_is_repeatable_for_same_seed():
    df1, train1, test1 = generate_production_data(n=40, seed=123)
    df2, train2, test2 = generate_production_data(n=40, seed=123)

    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(train1.sort_index(axis=1), train2.sort_index(axis=1))
    pd.testing.assert_frame_equal(test1.sort_index(axis=1), test2.sort_index(axis=1))


def test_generate_production_data_uses_requested_ranges():
    df, _, _ = generate_production_data(
        n=100,
        seed=42,
        production_time_range=(2.0, 3.0),
        deadline_buffer_range=(5.0, 6.0),
    )

    assert df["czas_produkcji_h"].between(2.0, 3.0).all()
    assert ((df["termin_h"] - df["czas_produkcji_h"]).between(5.0, 6.0)).all()


def test_generate_production_data_deadline_is_always_greater_than_processing_time():
    df, _, _ = generate_production_data(n=80, seed=42)

    assert (df["termin_h"] > df["czas_produkcji_h"]).all()


def test_generate_production_data_supports_multiple_machines():
    df, _, _ = generate_production_data(n=60, n_machines=3, seed=42)

    assert "lateness_h_sim" in df.columns
    assert (df["lateness_h_sim"] >= 0).all()


def test_generate_production_data_rejects_invalid_shape():
    with pytest.raises(ValueError, match="Nieprawidłowe wartości w liście ksztalty"):
        generate_production_data(n=10, ksztalty=["nie_ma_takiego"], seed=42)


def test_generate_production_data_rejects_invalid_material():
    with pytest.raises(ValueError, match="Nieprawidłowe wartości w liście materialy"):
        generate_production_data(n=10, materialy=["stal"], seed=42)


def test_generate_production_data_empty_shape_or_material_list_falls_back_to_defaults():
    df_shapes, _, _ = generate_production_data(n=20, ksztalty=[], seed=42)
    df_materials, _, _ = generate_production_data(n=20, materialy=[], seed=42)

    assert not df_shapes.empty
    assert not df_materials.empty
    assert "ksztalt" in df_shapes.columns
    assert "material" in df_materials.columns
    assert df_shapes["ksztalt"].nunique() >= 1
    assert df_materials["material"].nunique() >= 1

from AOA.core.data_generation import generate_production_data
from AOA.core.models import train_selected_models


def test_train_selected_models_returns_model_pack():
    _, train_df, _ = generate_production_data(n=100)
    pack = train_selected_models(
        train_df,
        selected_models=["Quality", "Delay"],
    )

    assert isinstance(pack, dict)
    assert "quality" in pack
    assert "delay" in pack
    assert "schedule" in pack
    assert "scaler" in pack
    assert "selected_models" in pack
    assert "backend" in pack
    assert pack["quality"] is not None
    assert pack["delay"] is not None
    assert pack["schedule"] is None
    assert pack["scaler"] is not None
    assert pack["selected_models"] == ["Quality", "Delay"]
    assert pack["backend"] == "classic"

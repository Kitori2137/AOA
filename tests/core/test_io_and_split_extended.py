from pathlib import Path

import pandas as pd
import pytest

from AOA.core.data_io import load_csv, save_csv
from AOA.core.dataset_ops import split_train_test


def test_load_csv_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(Exception):
        load_csv(missing)


def test_save_csv_and_load_csv_roundtrip(tmp_path):
    path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    save_csv(df, path)
    loaded = load_csv(path)

    pd.testing.assert_frame_equal(loaded, df)


def test_split_train_test_handles_small_dataset():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    train_df, test_df = split_train_test(df, train_ratio=0.8)

    assert len(train_df) + len(test_df) == len(df)
    assert 0 < len(train_df) < len(df)
    assert 0 < len(test_df) < len(df)

import pandas as pd
import pytest

from AOA.core.dataset_ops import split_train_test


def test_split_train_test_returns_two_non_empty_dataframes():
    df = pd.DataFrame({
        "a": range(10),
        "b": range(10, 20),
    })

    train_df, test_df = split_train_test(df, train_ratio=0.8)

    assert len(train_df) == 8
    assert len(test_df) == 2
    assert not train_df.empty
    assert not test_df.empty


def test_split_train_test_invalid_ratio_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError):
        split_train_test(df, train_ratio=1.0)


def test_split_train_test_empty_dataframe_raises():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        split_train_test(df, train_ratio=0.8)

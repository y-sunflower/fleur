import pytest
import pandas as pd
import polars as pl
import os

from fleur.datasets import _load_data, load_iris

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = "iris.csv"
DATASET_PATH = os.path.join(PACKAGE_DIR, DATASET_FILE)


def test_invalid_dataset_name():
    with pytest.raises(ValueError, match=r"^dataset_name must be one of:"):
        _load_data("invalid_dataset", backend="pandas")


def test_invalid_backend():
    with pytest.raises(ValueError, match=r"^backend must be one of:"):
        _load_data("iris", backend="invalid_format")


def test_load_data_pandas():
    df = _load_data("iris", backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_polars():
    df = _load_data("iris", backend="polars")
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()


def test_load_iris():
    df = load_iris()
    assert len(df) == 150
    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    assert df.columns.to_list() == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

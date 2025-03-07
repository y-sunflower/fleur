import pytest
import pandas as pd
import polars as pl
import os

from inferplot.datasets import load_data

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = "iris.csv"
DATASET_PATH = os.path.join(PACKAGE_DIR, DATASET_FILE)


def test_invalid_dataset_name():
    with pytest.raises(ValueError):
        load_data("invalid_dataset")


def test_invalid_return_as():
    with pytest.raises(ValueError):
        load_data("iris", return_as="invalid_format")


def test_load_data_pandas():
    df = load_data("iris", return_as="pandas")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_polars():
    df = load_data("iris", return_as="polars")
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()

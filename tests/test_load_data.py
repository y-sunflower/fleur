import pytest
import pandas as pd
import polars as pl
import os

from fleur.datasets import _load_data, load_iris, load_mtcars

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = "iris.csv"
DATASET_PATH = os.path.join(PACKAGE_DIR, DATASET_FILE)


def test_invalid_dataset_name():
    with pytest.raises(ValueError, match=r"^dataset_name must be one of:"):
        _load_data("invalid_dataset", backend="pandas")


@pytest.mark.parametrize("dataset", ["iris", "mtcars"])
def test_invalid_backend(dataset):
    with pytest.raises(ValueError, match=r"^backend must be one of:"):
        _load_data(dataset, backend="invalid_format")


@pytest.mark.parametrize("dataset", ["iris", "mtcars"])
def test_load_data_pandas(dataset):
    df = _load_data(dataset, backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


@pytest.mark.parametrize("dataset", ["iris", "mtcars"])
def test_load_data_polars(dataset):
    df = _load_data(dataset, backend="polars")
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


def test_load_mtcars():
    df = load_mtcars()
    assert len(df) == 32
    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    assert df.columns.to_list() == [
        "model",
        "mpg",
        "cyl",
        "disp",
        "hp",
        "drat",
        "wt",
        "qsec",
        "vs",
        "am",
        "gear",
        "carb",
    ]

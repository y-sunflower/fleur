import narwhals as nw
import pandas as pd
import polars as pl
import numpy as np

from typing import Dict

from fleur import InputDataHandler

import pytest


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_output_scheme(backend):
    df = nw.from_dict(
        {"height": [150, 160, 170], "weight": [50, 60, 70]}, backend=backend
    )
    info_df = InputDataHandler("height", "weight", data=df).get_info()

    x_series = nw.new_series("height", [150, 160, 170], backend=backend)
    y_series = nw.new_series("weight", [50, 60, 70], backend=backend)
    info_series = InputDataHandler(x_series, y_series).get_info()

    expected_keys = [
        "x",
        "y",
        "x_name",
        "y_name",
        "dataframe",
        "source",
    ]

    for info in [info_df, info_series]:
        assert isinstance(info, Dict)
        assert list(info) == expected_keys

    assert info_df["source"] == "dataframe"
    assert info_series["source"] == "series"


@pytest.mark.parametrize(
    ["x", "y"],
    [
        ([1, 2, 3], [1, 2, 3]),
        (pd.Series([1, 2, 3]), pd.Series([1, 2, 3])),
        (pl.Series([1, 2, 3]), pl.Series([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
    ],
)
def test_different_x_and_y_inputs(x, y):
    info_df = InputDataHandler(x, y).get_info()

    assert info_df["x"] == nw.new_series("x", x, backend="pandas")
    assert info_df["y"] == nw.new_series("y", y, backend="pandas")

    assert isinstance(info_df["x"], nw.Series)
    assert isinstance(info_df["y"], nw.Series)

    assert info_df["source"] in ["series", "array"]


def test_raise_type_error():
    with pytest.raises(
        TypeError, match=r"^`x` and `y` must be both strings \(column names\),"
    ):
        InputDataHandler("a", pd.Series([1, 2, 3]))


def test_raise_value_error():
    with pytest.raises(
        ValueError, match="If x and y are strings, `data` argument must be passed."
    ):
        InputDataHandler("a", "b")

    with pytest.raises(ValueError, match="`x` and/or `y` not found in `data` columns."):
        InputDataHandler("a", "b", pd.DataFrame({"a": [1, 2], "c": [1, 2]}))

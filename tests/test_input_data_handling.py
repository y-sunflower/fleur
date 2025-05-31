import narwhals as nw

from typing import Dict

from inferplot.data_input import InputDataHandler

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
        "x_name",
        "y_name",
        "source",
        "x",
        "y",
    ]

    for info in [info_df, info_series]:
        assert isinstance(info, Dict)
        assert list(info) == expected_keys

    for key in expected_keys:
        if key != "source":
            assert info_df[key] == info_series[key]

    assert info_df["source"] == "dataframe"
    assert info_series["source"] == "series"


@pytest.mark.parametrize("x", [[1, 2, 3, 4, 5], "height"])
@pytest.mark.parametrize("y", [[1, 2, 3, 4, 5], "weight"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_different_x_and_y_inputs(x, y, backend):
    df = nw.from_dict(
        {"height": [1, 2, 3, 4, 5], "weight": [1, 2, 3, 4, 5]}, backend=backend
    )
    info_df = InputDataHandler("height", "weight", data=df).get_info()

    assert info_df["x"] == nw.new_series(x, backend=backend)
    assert info_df["y"] == nw.new_series(y, backend=backend)

import narwhals as nw

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
        "x_dtype",
        "y_dtype",
        "length",
    ]

    for info in [info_df, info_series]:
        assert isinstance(info, dict)
        assert list(info) == expected_keys

    for key in expected_keys:
        if key != "source":
            assert info_df[key] == info_series[key]

    assert info_df["source"] == "dataframe"
    assert info_series["source"] == "series"

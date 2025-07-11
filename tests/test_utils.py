import fleur
from fleur._utils import _count_n_decimals, _infer_types

import pytest

import matplotlib.pyplot as plt
import narwhals as nw
import pandas as pd


def test_version():
    assert fleur.__version__ == "0.0.4"


def test_count_n_decimals():
    assert _count_n_decimals(12.3456) == 4
    assert _count_n_decimals(0.123) == 3
    assert _count_n_decimals(0.0001) == 4
    assert _count_n_decimals(123.456789) == 6
    assert _count_n_decimals(2.0) == 0
    assert _count_n_decimals(100.000) == 0
    assert _count_n_decimals(500) == 0
    assert _count_n_decimals(0.0) == 0
    assert _count_n_decimals(1.0) == 0


def test_count_n_decimals_error():
    with pytest.raises(TypeError):
        _count_n_decimals("1.0")

    with pytest.raises(TypeError):
        _count_n_decimals([1, 2])


def test_infer_types():
    data1 = nw.from_native(pd.DataFrame({"y": ["a", "b"], "x": [1, 2]}))
    data2 = nw.from_native(pd.DataFrame({"x": ["a", "b"], "y": [1, 2]}))
    data3 = nw.from_native(pd.DataFrame({"x": ["c", "d"], "y": ["a", "b"]}))
    data4 = nw.from_native(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))

    assert _infer_types("x", "y", data1) == ("y", "x")
    assert _infer_types("x", "y", data2) == ("x", "y")

    with pytest.raises(KeyError):
        _infer_types("x", "y", data3)

    with pytest.raises(KeyError):
        _infer_types("x", "y", data4)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)

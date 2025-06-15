import fleur
from fleur._utils import _count_n_decimals, _infer_types, _themify

import pytest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import narwhals as nw
import pandas as pd


def test_version():
    assert fleur.__version__ == "0.0.3"


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


def test_themify_grid(ax: Axes):
    _themify(ax)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        assert line.get_color() == "#525252"
        assert line.get_alpha() == 0.2
        assert line.get_zorder() == -5


def test_themify_spines_hidden(ax: Axes):
    _themify(ax)
    for position in ["top", "right", "left", "bottom"]:
        assert not ax.spines[position].get_visible()


def test_themify_tick_params(ax: Axes):
    _themify(ax)
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        assert tick.tick1line.get_markersize() == 0
        assert tick.label1.get_fontsize() == 8

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fleur import scatterstats


@pytest.fixture
def sample_data_pandas():
    np.random.seed(42)
    x = np.random.normal(loc=5, scale=10, size=200)
    y = x * 0.06 + np.random.normal(loc=0, scale=5, size=200)
    return pd.DataFrame({"x": x, "y": y})


def test_default(sample_data_pandas):
    fig, stats = scatterstats("x", "y", sample_data_pandas)
    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure object"
    assert isinstance(stats, dict), "Expected a dictionnary"
    assert len(stats.keys()) == 9
    for key in stats.keys():
        assert key in [
            "pvalue",
            "t_critical",
            "correlation",
            "intercept",
            "slope",
            "stderr_slope",
            "ci_lower",
            "ci_upper",
            "dof",
        ]
    plt.close(fig)


def test_no_marginal(sample_data_pandas):
    ax, stats = scatterstats("x", "y", sample_data_pandas, marginal=False)
    assert isinstance(ax, plt.Axes), "Expected a matplotlib Axes object"
    assert isinstance(stats, dict), "Expected a dictionnary"
    assert len(stats.keys()) == 9
    for key in stats.keys():
        assert key in [
            "pvalue",
            "t_critical",
            "correlation",
            "intercept",
            "slope",
            "stderr_slope",
            "ci_lower",
            "ci_upper",
            "dof",
        ]
    plt.close(ax.figure)


def test_custom_ax(sample_data_pandas):
    fig, ax = plt.subplots()
    returned_ax, stats = scatterstats(
        "x", "y", sample_data_pandas, ax=ax, marginal=False
    )
    assert returned_ax == ax, (
        "Expected the returned Axes to be the same as the input Axes"
    )
    plt.close(fig)


def test_invalid_columns(sample_data_pandas):
    with pytest.raises(KeyError):
        scatterstats("invalid_x", "y", sample_data_pandas)
    with pytest.raises(KeyError):
        scatterstats("x", "invalid_y", sample_data_pandas)


def test_style_params(sample_data_pandas):
    fig, stats = scatterstats(
        "x",
        "y",
        sample_data_pandas,
        bins=[10, 20],
        line_kws={"lw": 3, "color": "#1c3475"},
    )
    assert fig.get_figheight() == 6.0
    assert fig.get_figwidth() == 8.0

    axes = fig.get_axes()
    assert isinstance(axes, list)
    assert len(axes) == 3

    axesA = axes[1]
    axesB = axes[0]
    axesC = axes[2]

    line = axesA.get_lines()[0]
    assert line.get_color() == "#1c3475"
    assert line.get_linewidth() == 3.0

    assert sum(isinstance(item, Rectangle) for item in axesB.get_children()) >= 10
    assert sum(isinstance(item, Rectangle) for item in axesC.get_children()) >= 20


def test_raise_warning(sample_data_pandas):
    with pytest.warns(UserWarning):
        scatterstats("x", "y", sample_data_pandas, bins=20, marginal=False)

    with pytest.warns(UserWarning):
        scatterstats(
            "x", "y", sample_data_pandas, hist_kws={"color": "red"}, marginal=False
        )


def test_error_alternative(sample_data_pandas):
    with pytest.raises(ValueError):
        scatterstats("x", "y", sample_data_pandas, alternative="invalid")


def test_correlation_measure_invalid(sample_data_pandas):
    with pytest.raises(ValueError):
        scatterstats("x", "y", sample_data_pandas, correlation_measure="invalid")


if __name__ == "__main__":
    pytest.main()

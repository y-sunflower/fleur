import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fleur import ScatterStats
from fleur import datasets


@pytest.fixture
def sample_data():
    df = datasets.load_iris()
    df = df.rename(columns={"sepal_width": "x", "sepal_length": "y"})
    return df


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("correlation_measure", ["pearson", "kendall", "spearman"])
@pytest.mark.parametrize("bins", [None, 10, [10, 10], [10, 20]])
def test_default(sample_data, alternative, correlation_measure, bins):
    ss = ScatterStats(
        sample_data["x"],
        sample_data["y"],
        alternative=alternative,
        correlation_measure=correlation_measure,
    )

    fig = ss.plot(bins=bins)

    assert isinstance(fig, plt.Figure), "Expected a matplotlib Figure object"

    assert hasattr(ss, "pvalue")
    assert hasattr(ss, "intercept")
    assert hasattr(ss, "slope")
    assert hasattr(ss, "stderr_slope")
    assert hasattr(ss, "ci_lower")
    assert hasattr(ss, "ci_upper")
    assert hasattr(ss, "n_obs")
    assert hasattr(ss, "dof")
    assert hasattr(ss, "alpha")
    assert hasattr(ss, "fig")
    assert hasattr(ss, "ax")

    plt.close(fig)


def test_style_params(sample_data):
    fig = ScatterStats(sample_data["x"], sample_data["y"]).plot()
    assert fig.get_figheight() == 6.0
    assert fig.get_figwidth() == 8.0

    axes = fig.get_axes()
    assert isinstance(axes, list)
    assert len(axes) == 3

    axesA = axes[1]
    axesB = axes[0]
    axesC = axes[2]

    line = axesA.get_lines()[0]
    assert line.get_color() == "#1f77b4"

    n_rectangle_axesC = sum(
        isinstance(item, Rectangle) for item in axesC.get_children()
    )
    n_rectangle_axesB = sum(
        isinstance(item, Rectangle) for item in axesB.get_children()
    )
    assert n_rectangle_axesC >= 10, f"{axesC.get_children()}"
    assert n_rectangle_axesB >= 10, f"{axesB.get_children()}"


def test_raise_warning(sample_data):
    with pytest.warns(
        UserWarning, match="bins/hist_kws arguments are ignored when marginal=False."
    ):
        ScatterStats(sample_data["x"], sample_data["y"]).plot(bins=20, marginal=False)

    with pytest.warns(
        UserWarning, match="bins/hist_kws arguments are ignored when marginal=False."
    ):
        ScatterStats(sample_data["x"], sample_data["y"]).plot(
            hist_kws={"color": "red"},
            marginal=False,
        )


def test_error_alternative(sample_data):
    with pytest.raises(
        ValueError,
        match="alternative argument must be one of: 'two-sided', 'less', 'greater'.",
    ):
        ScatterStats(sample_data["x"], sample_data["y"], alternative="invalid")


def test_correlation_measure_invalid(sample_data):
    with pytest.raises(
        ValueError,
        match="correlation_measure argument must be one of: 'pearson', 'kendall', 'spearman'.",
    ):
        ScatterStats(sample_data["x"], sample_data["y"], correlation_measure="invalid")


def test_summary_prints(capsys, sample_data):
    ss = ScatterStats(sample_data["x"], sample_data["y"])
    ss.summary()
    captured = capsys.readouterr()
    assert captured.out.startswith("Correlation stats")

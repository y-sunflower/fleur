import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import polars as pl

from fleur.barstats import BarStats
import fleur.data as data


@pytest.fixture
def sample_data():
    df = data.load_mtcars()
    return df


@pytest.fixture
def sample_data2():
    df = data.load_iris("polars").with_columns(
        pl.when(pl.col("petal_length") < 2)
        .then(pl.lit("short (< 2)"))
        .when(pl.col("petal_length") < 5)
        .then(pl.lit("medium (> 2 & < 5)"))
        .otherwise(pl.lit("long (> 5)"))
        .alias("petal_size")
    )
    return df


@pytest.fixture
def sample_2x2_data():
    df = data.load_mtcars()
    # Filter to get 2x2 contingency table
    df_2x2 = df[df["cyl"].isin([4, 6])]
    return df_2x2


@pytest.mark.parametrize("ax", [None, "new"])
def test_default(sample_data, ax):
    if ax is not None:
        fig, ax = plt.subplots()

    bs = BarStats(x="cyl", y="vs", data=sample_data)
    assert bs.test_name == "Fisher's exact"
    assert hasattr(bs, "statistic")
    assert hasattr(bs, "pvalue")
    assert hasattr(bs, "contingency_table")
    assert bs.contingency_table.shape == (3, 2)
    assert bs.n_obs == 32
    assert bs.n_cat == 3
    assert bs.n_levels == 2

    fig_output = bs.plot(ax=ax)

    if ax is not None:
        assert fig_output == fig


def test_fisher_exact_2x2(sample_2x2_data):
    bs = BarStats(x="cyl", y="vs", data=sample_2x2_data)
    assert bs.test_name == "Fisher's exact"
    assert hasattr(bs, "pvalue")
    assert hasattr(bs, "odds_ratio")
    assert bs.contingency_table.shape == (2, 2)


def test_auto_approach_selects_fisher(sample_2x2_data):
    bs = BarStats(x="cyl", y="vs", data=sample_2x2_data)
    assert bs.test_name in ["Chi-square", "Fisher's exact"]


def test_plot_stacked(sample_data):
    fig, ax = plt.subplots()
    bs = BarStats(x="cyl", y="vs", data=sample_data)
    fig_out = bs.plot(ax=ax, plot_type="stacked")

    assert isinstance(fig_out, Figure)
    assert bs.ax == ax
    plt.close(fig)


def test_plot_grouped(sample_data):
    fig, ax = plt.subplots()
    bs = BarStats(x="cyl", y="vs", data=sample_data)
    fig_out = bs.plot(ax=ax, plot_type="grouped")

    assert isinstance(fig_out, Figure)
    assert bs.ax == ax
    plt.close(fig)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("show_stats", [True, False])
@pytest.mark.parametrize("show_counts", [True, False])
@pytest.mark.parametrize("plot_type", ["stacked", "grouped"])
def test_plot_options(sample_data, orientation, show_stats, show_counts, plot_type):
    fig, ax = plt.subplots()
    bs = BarStats(x="cyl", y="vs", data=sample_data)
    fig_out = bs.plot(
        ax=ax,
        orientation=orientation,
        show_stats=show_stats,
        show_counts=show_counts,
        plot_type=plot_type,
    )

    assert isinstance(fig_out, Figure)
    assert bs.ax == ax
    plt.close(fig)


def test_attributes_exist(sample_data):
    bs = BarStats(x="cyl", y="vs", data=sample_data)

    # Check required attributes
    assert hasattr(bs, "statistic")
    assert hasattr(bs, "pvalue")
    assert hasattr(bs, "test_name")
    assert hasattr(bs, "contingency_table")
    assert hasattr(bs, "n_obs")
    assert hasattr(bs, "n_cat")
    assert hasattr(bs, "n_levels")
    assert hasattr(bs, "expression")
    assert hasattr(bs, "cramers_v")

    # Check data types
    assert isinstance(bs.pvalue, (float, np.floating))
    assert isinstance(bs.test_name, str)
    assert isinstance(bs.contingency_table, np.ndarray)
    assert isinstance(bs.n_obs, (int, np.integer))
    assert isinstance(bs.n_cat, (int, np.integer))
    assert isinstance(bs.n_levels, (int, np.integer))
    assert isinstance(bs.expression, str)


def test_error_invalid_approach(sample_data):
    with pytest.raises(ValueError, match="`approach` must be one of"):
        BarStats(x="cyl", y="vs", data=sample_data, approach="invalid")


def test_error_invalid_orientation(sample_data):
    with pytest.raises(
        ValueError, match="`orientation` must be one of: 'vertical', 'horizontal'"
    ):
        BarStats(x="cyl", y="vs", data=sample_data).plot(orientation="invalid")


def test_error_invalid_plot_type(sample_data):
    bs = BarStats(x="cyl", y="vs", data=sample_data)
    with pytest.raises(
        ValueError, match="`plot_type` must be one of: 'stacked', 'grouped'"
    ):
        bs.plot(plot_type="invalid")


def test_contingency_table_correct(sample_data):
    bs = BarStats(x="cyl", y="vs", data=sample_data)

    # Check that contingency table sums to total observations
    assert bs.contingency_table.sum() == bs.n_obs

    # Check that contingency table has correct shape
    assert bs.contingency_table.shape == (bs.n_cat, bs.n_levels)

    # Check that all values are non-negative
    assert np.all(bs.contingency_table >= 0)


def test_expression_format(sample_data):
    bs = BarStats(x="cyl", y="vs", data=sample_data)

    # Check that expression is properly formatted
    assert bs.expression.startswith("$")
    assert bs.expression.endswith("$")
    assert "p =" in bs.expression
    assert "n_{obs}" in bs.expression


def test_cramers_v_range(sample_data2):
    bs = BarStats("species", "petal_size", data=sample_data2)

    # Cramer's V should be between 0 and 1
    assert 0 <= bs.cramers_v <= 1, f"Cramer's V has unexpected value: {bs.cramers_v}"


def test_colors_parameter(sample_data):
    fig, ax = plt.subplots()
    bs = BarStats(x="cyl", y="vs", data=sample_data)

    custom_colors = ["red", "blue"]
    fig_out = bs.plot(ax=ax, colors=custom_colors)

    assert isinstance(fig_out, Figure)
    plt.close(fig)


def test_not_implemented_error(sample_data):
    with pytest.raises(
        NotImplementedError,
        match="Paired group comparison has not been implemented yet.",
    ):
        BarStats(x="cyl", y="vs", data=sample_data, paired=True)

    with pytest.raises(
        NotImplementedError,
        match='Only `approach="freq"` has been implemented.',
    ):
        BarStats(x="cyl", y="vs", data=sample_data, approach="bayes")

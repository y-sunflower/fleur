import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl

from fleur import BetweenStats
import fleur.datasets as datasets


@pytest.fixture
def sample_data():
    df = datasets.load_iris()
    df = df.rename(columns={"species": "x", "sepal_length": "y"})
    return df


def test_default(sample_data):
    fig = BetweenStats(sample_data["x"], sample_data["y"]).plot()
    assert isinstance(fig, mpl.figure.Figure), (
        f"Expected a matplotlib Figure, not: {type(fig)}"
    )
    plt.close(fig)


def test_custom_ax(sample_data):
    fig, ax = plt.subplots()
    bs = BetweenStats(sample_data["x"], sample_data["y"])
    bs.plot(ax=ax)
    assert bs.ax == ax, "Expected the returned Axes to be the same as the input Axes"
    plt.close(fig)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("show_stats", [True, False])
def test_expected_attributes(sample_data, orientation, show_stats):
    fig, ax = plt.subplots()
    bs = BetweenStats(sample_data["x"], sample_data["y"])
    bs.plot(orientation=orientation, show_stats=show_stats, ax=ax)

    assert hasattr(bs, "ax")
    assert hasattr(bs, "statistic")
    assert hasattr(bs, "pvalue")
    assert hasattr(bs, "main_stat")
    assert hasattr(bs, "n_cat")
    assert hasattr(bs, "n_obs")
    assert hasattr(bs, "name")
    assert hasattr(bs, "dof_between")
    assert hasattr(bs, "dof_within")

    assert bs.name == "One-way ANOVA"

    plt.close(fig)


def test_summary_prints(capsys, sample_data):
    bs = BetweenStats(sample_data["x"], sample_data["y"])
    bs.summary()
    captured = capsys.readouterr()
    assert captured.out.startswith("Between stats comparison")


@pytest.mark.parametrize("paired", [True, False])
def test_not_two_categories(sample_data, paired):
    sample_data = sample_data[sample_data["x"] != "setosa"]

    bs = BetweenStats(sample_data["x"], sample_data["y"], paired=paired)

    assert hasattr(bs, "dof")

    if paired:
        assert bs.name == "Paired t-test"
    else:
        assert bs.name == "T-test"


def test_not_enough_categories(sample_data):
    sample_data = sample_data[sample_data["x"] == "setosa"]

    with pytest.raises(
        ValueError,
        match="You must have at least 2 distinct categories in your category column",
    ):
        BetweenStats(sample_data["x"], sample_data["y"])


def test_error_paired_anova(sample_data):
    with pytest.raises(
        NotImplementedError,
        match="Repeated measures ANOVA has not been implemented yet.",
    ):
        BetweenStats(sample_data["x"], sample_data["y"], paired=True)


def test_error_invalid_orientation(sample_data):
    with pytest.raises(
        ValueError,
        match="orientation argument must be one of: 'vertical', 'horizontal'.",
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).plot(orientation="invalid")


def test_error_invalid_colors(sample_data):
    with pytest.raises(
        ValueError,
        match=r"^`colors` argument must have at least",
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).plot(colors=["#fff"])

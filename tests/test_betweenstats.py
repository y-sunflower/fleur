import pytest
import re
import matplotlib.pyplot as plt

from fleur import BetweenStats
import fleur.datasets as datasets


@pytest.fixture
def sample_data():
    df = datasets.load_iris()
    df = df.rename(columns={"species": "x", "sepal_length": "y"})
    return df


def test_summary_error(sample_data):
    with pytest.raises(
        RuntimeError, match=re.escape("Must call 'plot()' before calling 'summary()'.")
    ):
        BetweenStats(sample_data["x"], sample_data["y"]).summary()


def test_default(sample_data):
    bs = BetweenStats(sample_data["x"], sample_data["y"]).plot()
    assert isinstance(bs.ax, plt.Axes), (
        f"Expected a matplotlib Axes object, not: {type(bs.ax)}"
    )
    plt.close(bs.ax.figure)


def test_custom_ax(sample_data):
    fig, ax = plt.subplots()
    bs = BetweenStats(sample_data["x"], sample_data["y"]).plot(ax=ax)
    assert bs.ax == ax, "Expected the returned Axes to be the same as the input Axes"
    plt.close(fig)


def test_expected_attributes(sample_data):
    fig, ax = plt.subplots()
    bs = BetweenStats(sample_data["x"], sample_data["y"]).plot(ax=ax)

    assert hasattr(bs, "ax")
    assert hasattr(bs, "statistic")
    assert hasattr(bs, "pvalue")
    assert hasattr(bs, "main_stat")
    assert hasattr(bs, "expression")
    assert hasattr(bs, "n_cat")


if __name__ == "__main__":
    pytest.main()

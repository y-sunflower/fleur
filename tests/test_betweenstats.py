import pytest
import matplotlib.pyplot as plt

from inferplot import betweenstats
import inferplot.datasets as datasets


@pytest.fixture
def sample_data_pandas():
    df = datasets.load_iris()
    df = df.rename(columns={"species": "x", "sepal_length": "y"})
    return df


def test_default(sample_data_pandas):
    ax = betweenstats("x", "y", sample_data_pandas)
    assert isinstance(ax, plt.Axes), "Expected a matplotlib Axes object"
    plt.close(ax.figure)


def test_custom_ax(sample_data_pandas):
    fig, ax = plt.subplots()
    returned_ax = betweenstats("x", "y", sample_data_pandas, ax=ax)
    assert returned_ax == ax, (
        "Expected the returned Axes to be the same as the input Axes"
    )
    plt.close(fig)


def test_invalid_columns(sample_data_pandas):
    with pytest.raises(KeyError):
        betweenstats("invalid_x", "y", sample_data_pandas)
    with pytest.raises(KeyError):
        betweenstats("x", "invalid_y", sample_data_pandas)


if __name__ == "__main__":
    pytest.main()

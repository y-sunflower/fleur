import matplotlib.pyplot as plt
import narwhals as nw
import seaborn as sns
from narwhals.typing import IntoDataFrame


def betweenstats(
    x: str,
    y: str,
    data: IntoDataFrame,
    ax=None,
):
    data = nw.from_native(data).to_pandas()

    if ax is None:
        ax = plt.gca()

    sns.violinplot(x=x, y=y, data=data, ax=ax)
    sns.boxplot(
        x=x,
        y=y,
        data=data,
        width=0.3,
        color="black",
        linewidth=1,
        fill=False,
        ax=ax,
    )

    return ax


if __name__ == "__main__":
    from inferplot import datasets

    data = datasets.load_data("iris")
    fig, ax = plt.subplots()
    betweenstats(data=data, x="species", y="sepal_length", ax=ax)
    plt.savefig("cache.png", dpi=300)

import matplotlib.pyplot as plt
import narwhals as nw
from narwhals.typing import IntoDataFrame
from typing import Union

from inferplot._utils import _infer_types


def betweenstats(
    x: str,
    y: str,
    data: IntoDataFrame,
    orientation: str = "vertical",
    violin_kws: Union[dict, None] = None,
    box_kws: Union[dict, None] = None,
    ax=None,
):
    df = nw.from_native(data)
    cat_col, num_col = _infer_types(x, y, df)
    n_cat = df[cat_col].n_unique()
    print(n_cat)
    result = [sub_df[num_col].to_list() for _key, sub_df in df.group_by(cat_col)]

    if ax is None:
        ax = plt.gca()

    if violin_kws is None:
        violin_kws = {}
    if box_kws is None:
        box_kws = {}

    ax.violinplot(result, orientation=orientation, **violin_kws)
    ax.boxplot(result, orientation=orientation, **box_kws)

    return ax


if __name__ == "__main__":
    from inferplot import datasets

    data = datasets.load_data("iris", return_as="polars")

    fig, ax = plt.subplots()
    betweenstats(
        data=data,
        x="species",
        y="sepal_length",
        orientation="vertical",
        ax=ax,
    )
    plt.savefig("cache.png", dpi=300)

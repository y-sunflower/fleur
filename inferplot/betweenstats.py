import matplotlib.pyplot as plt
import matplotlib
import narwhals as nw
from narwhals.typing import IntoDataFrame
from typing import Union
import scipy.stats as st
import numpy as np

from inferplot._utils import _infer_types
from inferplot.utils import themify

np.random.seed(0)


def betweenstats(
    x: str,
    y: str,
    data: IntoDataFrame,
    orientation: str = "vertical",
    paired: bool = False,
    colors: list = None,
    plot_violin: bool = True,
    plot_box: bool = True,
    plot_scatter: bool = True,
    violin_kws: Union[dict, None] = None,
    box_kws: Union[dict, None] = None,
    scatter_kws: Union[dict, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
    **kwargs,
):
    r"""
    Plot a boxplot and violinplot and annotate it with main statistical results.

    :param x: The column name for the x-axis variable.
    :param y: The column name for the y-axis variable.
    :param data: The DataFrame containing the data to be plotted. Can be any dataframe format supported by `narwhals <https://narwhals-dev.github.io/narwhals/>`_ (pandas, Polars, PyArrow, cuDF, Modin).
    :param orientation: Defines the orientation of the violins/boxs. Default is 'vertical'. Must be one of 'vertical' and 'horizontal'.
    :param paired: Whether or not the samples are related or not (e.g independent).
    :param plot_violin: Whether or not to plot a violin plot. Default to ``True``.
    :param plot_box: Whether or not to plot a box plot. Default to ``True``.
    :param plot_scatter: Whether or not to plot a scatter plot. Default to ``True``.
    :param violin_kws: Additional parameters which will be passed to the ``violinplot()`` function in matplotlib.
    :param box_kws: Additional parameters which will be passed to the ``boxplot()`` function in matplotlib.
    :param scatter_kws: Additional parameters which will be passed to the ``scatter()`` function in matplotlib.
    :param ax: The Axes to plot on. If ``None``, use the current Axes using ``plt.gca()``. Default is ``None``.
    :return: A matplotlib Axes.

    Examples
    --------

    .. plot::

        import inferplot
        from inferplot import datasets

        iris = datasets.load_iris()
        ax = inferplot.betweenstats(
            data=iris,
            x="species",
            y="sepal_length",
        )
    """
    if orientation not in ["vertical", "horizontal"]:
        raise ValueError(
            "orientation argument must be one of: 'vertical', 'horizontal'."
        )

    df = nw.from_native(data)
    cat_col, num_col = _infer_types(x, y, df)
    result = [sub_df[num_col].to_list() for _, sub_df in df.group_by(cat_col)]
    sample_sizes = [len(sub_df) for _, sub_df in df.group_by(cat_col)]
    cat_labels = df[cat_col].unique().to_list()
    n_cat = df[cat_col].n_unique()
    n = len(data)

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_cat]
    if ax is None:
        ax = plt.gca()
    if violin_kws is None:
        violin_kws = {}
    if box_kws is None:
        box_kws = {}
    if scatter_kws is None:
        scatter_kws = {}
    violin_default_kws = {"orientation": orientation, "showextrema": False}
    violin_default_kws.update(violin_kws)
    box_default_kws = {"orientation": orientation}
    box_default_kws.update(box_kws)
    scatter_default_kws = {"alpha": 0.5}
    scatter_default_kws.update(box_kws)

    if plot_violin:
        violin_artists = ax.violinplot(result, **violin_default_kws)

    if plot_box:
        box_style = {"color": "#3b3b3b"}
        ax.boxplot(
            result,
            boxprops=box_style,
            medianprops=box_style,
            capprops=box_style,
            whiskerprops=box_style,
            **box_default_kws,
        )

    if plot_scatter:
        for i, (values, label, color) in enumerate(zip(result, cat_labels, colors)):
            jitter = np.random.uniform(low=-0.1, high=0.1, size=len(values))
            x_coords = np.full(len(values), i) + jitter + 1
            if orientation == "vertical":
                ax.scatter(x_coords, values, color=color, **scatter_default_kws)
            else:  # "horizontal":
                ax.scatter(values, x_coords, color=color, **scatter_default_kws)

    for patch, color in zip(violin_artists["bodies"], colors):
        patch.set(color=color)

    if n_cat < 2:
        raise ValueError(
            "You must have at least 2 distinct categories in your category column"
        )
    elif n_cat == 2:
        if paired:
            ttest = st.ttest_rel(result[0], result[1])
        else:
            ttest = st.ttest_ind(result[0], result[1])
        statistic = ttest.statistic
        pvalue = ttest.pvalue
        dof = int(ttest.df)
        main_stat = f"t_{{Student}}({dof}) = {statistic:.2f}"
    else:  # n >= 3
        if paired:
            raise NotImplementedError(
                "Repeated measures ANOVA has not been implemented yet."
            )
        else:
            anova = st.f_oneway(*result)
        statistic = anova.statistic
        pvalue = anova.pvalue
        dof_between = n_cat - 1
        dof_within = n - n_cat
        main_stat = f"F({dof_between}, {dof_within}) = {statistic:.2f}"

    expr_list = [
        "$",
        f"{main_stat}, ",
        f"p = {pvalue:.4f}, ",
        f"n_{{obs}} = {n}",
        "$",
    ]

    all_expr = "".join(expr_list)

    annotation_params = dict(transform=ax.transAxes, va="top")
    ax.text(x=0.05, y=1.09, s=all_expr, size=9, **annotation_params)

    ax = themify(ax)

    ticks = [i + 1 for i in range(len(sample_sizes))]
    labels = [f"{label}\nn = {n}" for n, label in zip(sample_sizes, cat_labels)]
    if orientation == "vertical":
        ax.set_xticks(ticks, labels=labels)
    elif orientation == "horizontal":
        ax.set_yticks(ticks, labels=labels)

    return ax


if __name__ == "__main__":
    from inferplot import datasets

    data = datasets.load_iris()

    fig, ax = plt.subplots()
    betweenstats(
        data=data,
        x="species",
        y="sepal_length",
        orientation="vertical",
        ax=ax,
    )
    plt.savefig("cache.png", dpi=300)

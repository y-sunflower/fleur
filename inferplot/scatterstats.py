import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import seaborn as sns
from typing import Union, List, Tuple
import narwhals as nw
from narwhals.typing import IntoDataFrame
import warnings
from numbers import Number

from inferplot._utils import _count_n_decimals


def scatterstats(
    x: str,
    y: str,
    data: IntoDataFrame,
    marginal: bool = True,
    ci: Number = 95,
    alternative: str = "two-sided",
    correlation_measure: str = "pearson",
    bins: Union[int, List[int]] = None,
    color: Union[str, tuple, None] = None,
    figsize: Tuple[float, float] = (8, 6),
    scatter_kws: Union[dict, None] = None,
    line_kws: Union[dict, None] = None,
    hist_kws: Union[dict, None] = None,
    subplot_mosaic_kwargs: Union[dict, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
    **kwargs,
) -> Union[matplotlib.axes.Axes, matplotlib.figure.Figure]:
    r"""
    Plot a scatter plot of two variables, with a linear regression line and annotate it with main statistical results.

    :param x: The column name for the x-axis variable.
    :param y: The column name for the y-axis variable.
    :param data: The DataFrame containing the data to be plotted. Can be any dataframe format supported by `narwhals <https://narwhals-dev.github.io/narwhals/>`_ (pandas, Polars, PyArrow, cuDF, Modin).
    :param ci: Confidence level for the top label and the regression plot. The default value is 95 (for a 95% confidence level).
    :param alternative: Defines the alternative hypothesis. Default is 'two-sided'. Must be one of 'two-sided', 'less' and 'greater'.
    :param correlation_measure: The correlation measure to use. Default is 'pearson'. Must be one of 'pearson', 'kendall', 'spearman'.
    :param bins: Number of bins for the marginal distributions. This can be an integer or a list of two integers (the first for the top distribution and the second for the other).
    :param marginal: Whether to include marginal histograms. Default is ``True``.
    :param color: Any valid matplotlib color that will be used to color all the elements of the plot.
    :param figsize: Dimensions of the matplotlib figure created. The default value is ``(8, 6)``.
    :param line_kws: Additional parameters which will be passed to the ``plot()`` function in matplotlib.
    :param scatter_kws: Additional parameters which will be passed to the ``scatter()`` function in matplotlib.
    :param hist_kws: Additional parameters which will be passed to the ``hist()`` function in matplotlib.
    :param subplot_mosaic_kwargs: Additional keyword arguments to pass to ``plt.subplot_mosaic()``. Default is ``None``.
    :param ax: The Axes to plot on. If ``None``, use the current Axes using ``plt.gca()``. Default is ``None``.
    :param kwargs: Additional keyword arguments to pass to `seaborn.regplot <https://seaborn.pydata.org/generated/seaborn.regplot.html>`_.
    :return: A Tuple containing either an Axes (if ``marginal=False``) or a Figure (if ``marginal=True``) for the first element, and a statistics dictionary for the second element.

    Examples
    --------

    .. plot::

        >>> import numpy as np
        >>> import pandas as pd
        >>> import inferplot
        >>> np.random.seed(42)

        >>> x = np.random.normal(loc=5, scale=10, size=200)
        >>> y = x * 0.06 + np.random.normal(loc=0, scale=5, size=200)
        >>> data = pd.DataFrame({"x": x, "y": y})

        >>> fig, stats = inferplot.scatterstats("x", "y", data, bins=20, ci=90)
    """

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(
            "alternative argument must be one of: 'two-sided', 'less', 'greater'."
        )

    if not marginal and any([bins is not None, hist_kws is not None]):
        warnings.warn(
            "bins/hist_kws arguments are ignored when marginal=False.",
            category=UserWarning,
        )

    if correlation_measure not in ["pearson", "kendall", "spearman"]:
        raise ValueError(
            "correlation_measure argument must be one of: 'pearson', 'kendall', 'spearman'."
        )

    default_subplot_mosaic_kwargs = dict(
        width_ratios=(5, 1), height_ratios=(1, 5), figsize=figsize
    )
    if subplot_mosaic_kwargs is None:
        subplot_mosaic_kwargs = {}
    subplot_mosaic_kwargs.update(default_subplot_mosaic_kwargs)

    if marginal:
        scheme = """
B.
AC
"""
        fig, axs = plt.subplot_mosaic(scheme, **subplot_mosaic_kwargs)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax = axs["A"]  # main Axes of the Figure
    else:
        if ax is None:
            ax = plt.gca()
        fig = plt.gcf()

    data = nw.from_native(data).to_pandas()

    regression = st.linregress(data[x], data[y], alternative=alternative)

    n = len(data)
    alpha = 1 - ci / 100
    dof = n - 2

    if correlation_measure == "pearson":
        correlation = st.pearsonr(data[x], data[y]).statistic
        symbol_correl = "\\rho"
    elif correlation_measure == "kendall":
        correlation = st.kendalltau(data[x], data[y]).statistic
        symbol_correl = "\\tau"
    elif correlation_measure == "spearman":
        correlation = st.spearmanr(data[x], data[y]).statistic
        symbol_correl = "\\rho"

    p_value = regression.pvalue
    intercept = regression.intercept
    slope = regression.slope
    stderr_slope = regression.stderr
    t_critical = st.t.ppf(1 - alpha / 2, dof)
    ci_lower = slope - t_critical * stderr_slope
    ci_upper = slope + t_critical * stderr_slope
    statistics = {
        "p_value": p_value,
        "t_critical": t_critical,
        "correlation": correlation,
        "intercept": intercept,
        "slope": slope,
        "stderr_slope": stderr_slope,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "dof": dof,
    }

    ci_decimal = _count_n_decimals(ci)

    expr_list = [
        "$",
        f"t_{{Student}}({dof}) = {slope:.2f}, ",
        f"CI_{{{ci:.{ci_decimal}f}\\%}} = [{ci_lower:.2f}, {ci_upper:.2f}], ",
        f"p = {p_value:.4f}, ",
        f"{symbol_correl}_{{{correlation_measure.title()}}} = {correlation:.2f}, ",
        f"n = {n}",
        "$",
    ]
    all_expr = "".join(expr_list)

    if scatter_kws is None:
        scatter_kws = {}
    if line_kws is None:
        line_kws = {}
    sns.regplot(
        x=x,
        y=y,
        data=data,
        ci=ci,
        color=color,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        ax=ax,
        **kwargs,
    )

    ax.set(xlabel="", ylabel="")
    ax.grid(color="#525252", alpha=0.2)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(size=0, labelsize=8)

    if marginal:
        if bins is None:
            bins = 12

        if hist_kws is None:
            hist_kws = {}

        if isinstance(bins, List):
            binsB = bins[0]
            binsC = bins[1]
        else:
            binsB = binsC = bins

        axs["B"].hist(data[x], bins=binsB, color=color, **hist_kws)
        axs["C"].hist(
            data[y], orientation="horizontal", bins=binsC, color=color, **hist_kws
        )

        axs["B"].axis("off")
        axs["C"].axis("off")

    annotation_params = dict(transform=fig.transFigure, va="top")
    fig.text(x=0.1, y=0.95, s=all_expr, size=9, **annotation_params)
    fig.text(
        x=0.75,
        y=0.05,
        s=f"$\\hat{{y}}_i = {intercept:.2f} + {slope:.2f}x_i$",
        ha="right",
        weight="bold",
        style="normal",
        size=12,
        **annotation_params,
    )

    return (fig, statistics) if marginal else (ax, statistics)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    x = np.random.normal(loc=5, scale=10, size=200)
    y = x * 0.06 + np.random.normal(loc=0, scale=5, size=200)
    data = pd.DataFrame({"x": x, "y": y})

    fig, stats = scatterstats(
        "x", "y", data, bins=20, ci=90, correlation_measure="pearson"
    )
    fig.savefig("cache.png", dpi=300)

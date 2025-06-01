import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import numpy as np
import narwhals as nw
from narwhals.typing import IntoDataFrame

import warnings
from numbers import Number
from typing import Union, List, Tuple

from fleur._utils import _count_n_decimals
from fleur.utils import themify


def scatterstats(
    x: str,
    y: str,
    data: IntoDataFrame,
    marginal: bool = True,
    ci: Number = 95,
    alternative: str = "two-sided",
    correlation_measure: str = "pearson",
    bins: Union[int, List[int]] = None,
    figsize: Tuple[float, float] = (8, 6),
    scatter_kws: Union[dict, None] = None,
    line_kws: Union[dict, None] = None,
    area_kws: Union[dict, None] = None,
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
    :param figsize: Dimensions of the matplotlib figure created. The default value is ``(8, 6)``.
    :param line_kws: Additional parameters which will be passed to the ``plot()`` function in matplotlib.
    :param scatter_kws: Additional parameters which will be passed to the ``scatter()`` function in matplotlib.
    :param area_kws: Additional parameters which will be passed to the ``fill_between()`` function in matplotlib.
    :param hist_kws: Additional parameters which will be passed to the ``hist()`` function in matplotlib.
    :param subplot_mosaic_kwargs: Additional keyword arguments to pass to ``plt.subplot_mosaic()``. Default is ``None``.
    :param ax: The Axes to plot on. If ``None``, use the current Axes using ``plt.gca()``. Default is ``None``.
    :return: A Tuple containing either an Axes (if ``marginal=False``) or a Figure (if ``marginal=True``) for the first element, and a statistics dictionary for the second element.

    Examples
    --------

    .. plot::

        import fleur
        from fleur import datasets

        data = datasets.load_iris()
        fig, stats = fleur.scatterstats(
            x="sepal_length",
            y="sepal_width",
            data=data,
            correlation_measure="spearman",
        )
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
    default_subplot_mosaic_kwargs.update(subplot_mosaic_kwargs)

    if marginal:
        scheme = """
B.
AC
"""
        fig, axs = plt.subplot_mosaic(scheme, **default_subplot_mosaic_kwargs)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax = axs["A"]  # main Axes of the Figure
    else:
        if ax is None:
            ax = plt.gca()
        fig = plt.gcf()

    data = nw.from_native(data)
    x_np = data[x].to_numpy()
    y_np = data[y].to_numpy()

    regression = st.linregress(x_np, y_np, alternative=alternative)

    n = len(data)
    alpha = 1 - ci / 100
    dof = n - 2

    if correlation_measure == "pearson":
        correlation = st.pearsonr(x_np, y_np).statistic
        symbol_correl = "\\rho"
    elif correlation_measure == "kendall":
        correlation = st.kendalltau(x_np, y_np).statistic
        symbol_correl = "\\tau"
    elif correlation_measure == "spearman":
        correlation = st.spearmanr(x_np, y_np).statistic
        symbol_correl = "\\rho"

    pvalue = regression.pvalue
    intercept = regression.intercept
    slope = regression.slope
    stderr_slope = regression.stderr
    t_critical = st.t.ppf(1 - alpha / 2, dof)
    ci_lower = slope - t_critical * stderr_slope
    ci_upper = slope + t_critical * stderr_slope
    statistics = {
        "pvalue": pvalue,
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
        f"p = {pvalue:.4f}, ",
        f"{symbol_correl}_{{{correlation_measure.title()}}} = {correlation:.2f}, ",
        f"n_{{obs}} = {n}",
        "$",
    ]
    all_expr = "".join(expr_list)

    if scatter_kws is None:
        scatter_kws = {}
    if line_kws is None:
        line_kws = {}
    if area_kws is None:
        area_kws = {}
    area_default_kws = {
        "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        "alpha": 0.2,
    }
    area_default_kws.update(area_kws)

    x_values = np.linspace(np.min(x_np), np.max(x_np), 100)
    y_values = slope * x_values + intercept
    residuals = y_np - (slope * x_np + intercept)
    rse = np.sqrt(np.sum(residuals**2) / dof)
    x_mean = np.mean(x_np)
    x_var = np.sum((x_np - x_mean) ** 2)
    y_err = t_critical * rse * np.sqrt(1 / n + (x_values - x_mean) ** 2 / x_var)

    ax.fill_between(
        x_values,
        y_values - y_err,
        y_values + y_err,
        **area_default_kws,
    )
    ax.scatter(x_np, y_np, **scatter_kws)
    ax.plot(x_values, y_values, **line_kws)

    ax = themify(ax)

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

        axs["B"].hist(x_np, bins=binsB, **hist_kws)
        axs["C"].hist(y_np, orientation="horizontal", bins=binsC, **hist_kws)

        axs["B"].axis("off")
        axs["C"].axis("off")

    equation_sign = "+" if slope >= 0 else "-"

    annotation_params = dict(transform=fig.transFigure, va="top")
    fig.text(x=0.1, y=0.95, s=all_expr, size=9, **annotation_params)
    fig.text(
        x=0.75,
        y=0.05,
        s=f"$\\hat{{y}}_i = {intercept:.2f} {equation_sign} {abs(slope):.2f}x_i$",
        ha="right",
        weight="bold",
        style="normal",
        size=12,
        **annotation_params,
    )

    return (fig, statistics) if marginal else (ax, statistics)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fleur import datasets

    data = datasets.load_iris()

    fig, stats = scatterstats(
        x="sepal_length",
        y="sepal_width",
        data=data,
        bins=20,
        ci=95,
        correlation_measure="pearson",
    )
    fig.savefig("cache.png", dpi=300)

import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns
from typing import Union
import narwhals as nw
from narwhals.typing import IntoDataFrame


def scatterstats(
    x: str,
    y: str,
    data: IntoDataFrame,
    marginal: bool = True,
    ax: Union[matplotlib.axes.Axes, None] = None,
    subplot_mosaic_kwargs: Union[dict, None] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot a scatter plot of two variables, with a linear regression line and annotate it with the p-value, rho (correlation coefficient), and R-squared.

    :param x: The column name for the x-axis variable.
    :param y: The column name for the y-axis variable.
    :param data: The DataFrame containing the data to be plotted. Can be any dataframe format supported by `narwhals <https://narwhals-dev.github.io/narwhals/>`_ (pandas, Polars, PyArrow, cuDF, Modin).
    :param marginal: Whether to include marginal histograms. Default is ``True``.
    :param ax: The Axes to plot on. If ``None``, use the current Axes using ``plt.gca()``. Default is ``None``.
    :param subplot_mosaic_kwargs: Additional keyword arguments to pass to ``plt.subplot_mosaic``. Default is ``None``.
    :param **kwargs: Additional keyword arguments to pass to `seaborn.regplot <https://seaborn.pydata.org/generated/seaborn.regplot.html>`_.
    :return: The axes with the scatter plot and annotations.

    Examples
    --------

    .. plot::

       >>> import matplotlib.pyplot as plt
       >>> import numpy as np
       >>> import pandas as pd
       >>> import inferplot

       >>> x = np.random.normal(loc=5, scale=10, size=200)
       >>> y = x + np.random.normal(loc=0, scale=5, size=200)
       >>> data = pd.DataFrame({"x": x, "y": y})

       >>> inferplot.scatterstats("x", "y", data)

    Notes
    -----
    The function uses `seaborn.regplot` to create the scatter plot and linear regression line. Marginal histograms are added if ``marginal=True``.

    The annotations include:

    - p-value
    - ρ (rho): correlation coefficient
    - R-squared: coefficient of determination
    - The linear regression equation: :math:`\\hat{y}_i = \\text{intercept} + \\text{slope} \\cdot x_i`
    """
    default_subplot_mosaic_kwargs = dict(
        width_ratios=(5, 1), height_ratios=(1, 5), figsize=(8, 6)
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

    linear_regression = stats.linregress(data[x], data[y])

    p_value = linear_regression.pvalue
    r_value = linear_regression.rvalue
    rsqr_value = r_value**2

    sns.regplot(x=x, y=y, data=data, ax=ax, **kwargs)
    ax.set(xlabel="", ylabel="")

    if marginal:
        axs["B"].hist(data[x])
        axs["C"].hist(data[y], orientation="horizontal")

        axs["B"].axis("off")
        axs["C"].axis("off")

    annotation_params = dict(transform=fig.transFigure, va="top")
    fig.text(x=0.1, y=0.95, s=f"pvalue: {p_value:.4f}", **annotation_params)
    fig.text(x=0.1, y=0.9, s=f"ρ (rho): {r_value:.2f}", **annotation_params)
    fig.text(x=0.1, y=0.85, s=f"R squared: {rsqr_value:.2f}", **annotation_params)
    fig.text(
        x=0.75,
        y=0.05,
        s=f"$\\hat{{y}}_i = {linear_regression.intercept:.2f} + {linear_regression.slope:.2f}x_i$",
        ha="right",
        weight="bold",
        style="normal",
        **annotation_params,
    )

    return fig if marginal else ax

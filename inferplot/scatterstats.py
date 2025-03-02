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
    ax: Union[matplotlib.axes.Axes, None] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot a scatter plot of two variables, with a linear regression line
    and annotate it with the p-value, rho (correlation coefficient), and R-squared.

    Parameters
    ----------
    `x` : The column name for the x-axis variable

    `y` : The column name for the y-axis variable

    `data` : The DataFrame containing the data to be plotted.
    Can be any dataframe format supported by
    [narwhals](https://narwhals-dev.github.io/narwhals/)
    (pandas, Polars, PyArrow, cuDF, Modin).

    `ax` : The Axes to plot on. If None, use the current Axes using `plt.gca()`. Default is `None`.

    `**kwargs` : Additional keyword arguments to pass to [`sns.regplot()`](https://seaborn.pydata.org/generated/seaborn.regplot.html)

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        The axes with the scatter plot and annotations

    Examples
    --------

        import matplotlib.pyplot as plt
        import pandas as pd
        import inferplot

        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 3, 6, 9, 10])
        data = pd.DataFrame({"x": x, "y": y})

        fig, ax = plt.subplots(figsize=(8, 6))
        inferplot.scatterstats("x", "y", data, ax=ax)
        plt.show()

    ![img](https://raw.githubusercontent.com/JosephBARBIERDARNAL/inferplot/main/docs/img/scatterstats.png)
    """
    if ax is None:
        ax = plt.gca()

    data = nw.from_native(data).to_pandas()

    linear_regression = stats.linregress(data[x], data[y])

    p_value = linear_regression.pvalue
    r_value = linear_regression.rvalue
    rsqr_value = r_value**2

    sns.regplot(x=x, y=y, data=data, ax=ax, **kwargs)

    annotation_params = dict(transform=ax.transAxes, va="top")
    ax.text(x=0.1, y=1, s=f"pvalue: {p_value:.4f}", **annotation_params)
    ax.text(x=0.1, y=0.95, s=f"ρ (rho): {r_value:.2f}", **annotation_params)
    ax.text(x=0.1, y=0.9, s=f"R squared: {rsqr_value:.2f}", **annotation_params)

    ax.spines[["top", "right"]].set_visible(False)

    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import inferplot

    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 3, 6, 9, 10])
    data = pd.DataFrame({"x": x, "y": y})

    fig, ax = plt.subplots(figsize=(8, 6))
    inferplot.scatterstats("x", "y", data, ax=ax)
    plt.savefig("docs/img/scatterstats.png", dpi=300, bbox_inches="tight")
    plt.close()

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import narwhals as nw

from typing import Iterable
from narwhals.typing import SeriesT, Frame

from fleur._utils import _themify, _InputDataHandler, _get_first_n_colors


class BarStats:
    """
    Bar charts for categorical data with statistical details.

    Attributes:
        n_obs (int): Total number of observations.
    """

    def __init__(
        self,
        x: str | SeriesT | Iterable,
        y: str | SeriesT | Iterable,
        data: Frame | None = None,
    ):
        """
        Initialize a `BarStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like.
            y: Colname of `data` or a Series or array-like.
            data: An optional dataframe.
        """
        self._data_info = _InputDataHandler(x=x, y=y, data=data).get_info()

        x_name: str = self._data_info["x_name"]
        y_name: str = self._data_info["y_name"]
        df: Frame = self._data_info["dataframe"]

        df.with_columns(nw.col("bar").cast(nw.Enum))

        self.df_proportion = (
            df.group_by([x_name, y_name], observed=True)
            .size()
            .unstack(fill_value=0)
            .apply(lambda x: x / x.sum(), axis=1)
        )

        self.n_obs = len(df)
        self.n_cat = df[x_name].n_unique()

    def _fit(self): ...

    def plot(
        self,
        *,
        orientation: str = "vertical",
        colors: list | None = None,
        ax: Axes | None = None,
        bar_kws: dict | None = None,
    ) -> Figure:
        r"""
        Plot a barplot

        Args:
            ax (matplotlib.axes.Axes, ): Existing Axes to plot on. If None, uses
                current Axes.
            orientation: 'vertical' or 'horizontal' orientation of plots.
            colors: List of colors for each group.
        """
        if ax is None:
            ax: Axes = plt.gca()
        ax: Axes = _themify(ax)

        bottom = None

        colors = _get_first_n_colors(colors, self.n_cat)

        for idx, vs_value in enumerate(self.df_proportion.columns):
            if orientation == "horizontal":
                ax.barh(
                    self.df_proportion.index,
                    self.df_proportion[vs_value],
                    left=bottom,
                    color=colors[idx],
                )
            else:  # orientation == "vertical"
                ax.bar(
                    self.df_proportion.index,
                    self.df_proportion[vs_value],
                    bottom=bottom,
                    color=colors[idx],
                )

            if bottom is None:
                bottom = self.df_proportion[vs_value]
            else:
                bottom += self.df_proportion[vs_value]

        return self.fig


if __name__ == "__main__":
    from fleur import data

    df = data.load_mtcars("polars")
    fig, ax = plt.subplots()

    BarStats(x="vs", y="cyl", data=df).plot(ax=ax)

    fig.savefig("cache.png", dpi=300)

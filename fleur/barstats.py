import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import narwhals as nw
import numpy as np
import scipy.stats as st

from typing import Iterable, Any
from narwhals.typing import SeriesT, Frame

from fleur._utils import _themify, _InputDataHandler, _get_first_n_colors

import warnings


class BarStats:
    """
    Statistical comparison and plotting class for categorical data analysis.

    This class provides functionality to visualize and statistically compare
    categorical data across groups. It supports chi-square tests for independence,
    Fisher's exact test for small samples, and creates nice-looking stacked or
    grouped bar charts.

    Attributes:
        statistic (float): The computed test statistic (chi-square or Fisher's exact).
        pvalue (float): The p-value of the statistical test.
        main_stat (str): The formatted test statistic string for display.
        expression (str): Full LaTeX-style annotation string.
        test_name (str): Name of the statistical test used.
        n_obs (int): Total number of observations.
        n_cat (int): Number of unique categories in the first variable.
        n_levels (int): Number of unique levels in the second variable.
        contingency_table (np.ndarray): The contingency table.
        expected_frequencies (np.ndarray): Expected frequencies for chi-square test.
        cramers_v (float): Cramer's V effect size measure.
        ax (matplotlib.axes.Axes): The matplotlib axes used for plotting.
    """

    def __init__(
        self,
        x: str | SeriesT | Iterable,
        y: str | SeriesT | Iterable,
        data: Frame | None = None,
        approach: str = "parametric",
        paired: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize a `BarStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like (categorical).
            y: Colname of `data` or a Series or array-like (categorical).
            data: An optional dataframe.
            approach: A character specifying the type of statistical approach:
                "parametric" (default), "nonparametric", "robust", "bayes".
            paired: Whether comparing the same observations or not.
            kwargs: Additional arguments passed to the scipy test function.
        """
        valid_approachs: list[str] = ["parametric", "nonparametric", "robust", "bayes"]
        if approach not in valid_approachs:
            raise ValueError(
                f"`approach` must be one of {valid_approachs}, not {approach}"
            )

        self._data_info = _InputDataHandler(x=x, y=y, data=data).get_info()

        x_name: str = self._data_info["x_name"]
        y_name: str = self._data_info["y_name"]
        df: Frame = self._data_info["dataframe"].with_columns(
            nw.col(x_name).cast(nw.String).cast(nw.Categorical),
            nw.col(y_name).cast(nw.String).cast(nw.Categorical),
        )

        self.is_paired = paired
        self.n_obs = len(df)
        self.n_cat = df[x_name].n_unique()
        self.n_levels = df[y_name].n_unique()
        self._x_name = x_name
        self._y_name = y_name
        self._df = df

        self._x_levels: list = df[x_name].unique().to_list()
        self._y_levels: list = df[y_name].unique().to_list()

        self.contingency_df = (
            df.group_by(x_name, y_name)
            .agg(nw.len())
            .pivot(values="len", index=x_name, on=y_name)
            .with_columns(nw.selectors.numeric().fill_null(0))
        )
        self.contingency_table = self.contingency_df.select(
            nw.selectors.numeric()
        ).to_numpy()
        self._df_proportion = self.contingency_df.with_columns(
            nw.col(*self._y_levels) / nw.sum_horizontal(nw.col(*self._y_levels))
        )

        self._fit(approach=approach, **kwargs)

    def _fit(self, approach: str, **kwargs: Any):
        """
        Internal method to compute all the statistics and store
        them as attributes.

        Args:
            approach: A character specifying the type of statistical approach:
                "parametric" (default), "nonparametric", "robust", "bayes".
            kwargs: Additional arguments passed to the scipy test function.
        """
        if approach in ["parametric", "nonparametric"]:
            if self.is_paired:
                # if binary (2 levels) variable, use McNemar's test
                # if 3 levels or more, use Stuart-Maxwell test
                raise NotImplementedError(
                    "Paired group comparison has not been implemented yet."
                )
            else:  # not paired
                test = "Fisher"
                if test == "Fisher":
                    if self.contingency_table.shape != (2, 2):
                        raise ValueError(
                            "Fisher's exact test can only be used with 2x2 contingency tables"
                        )
                    self.test_output = st.fisher_exact(self.contingency_table, **kwargs)
                    self.statistic = None
                    self.pvalue = self.test_output[1]
                    self.odds_ratio = self.test_output[0]
                    self.test_name = "Fisher's exact"
                    self._letter = "p"
                    self.main_stat = f"OR = {self.odds_ratio:.3f}"
                    self.expected_frequencies = None
                    self.cramers_v = None

                    expr_list: list[str] = [
                        "$",
                        "Fisher's exact test, ",
                        f"p = {self.pvalue:.4f}, ",
                        f"OR = {self.odds_ratio:.3f}, ",
                        f"n_{{obs}} = {self.n_obs}",
                        "$",
                    ]
                else:  # test is chi2
                    self.test_output = st.chi2_contingency(
                        self.contingency_table, **kwargs
                    )
                    self.statistic = self.test_output[0]
                    self.pvalue = self.test_output[1]
                    self.dof = self.test_output[2]
                    self.expected_frequencies = self.test_output[3]
                    self.test_name = "Chi-square"
                    self._letter = "\\chi^2"
                    self.main_stat = f"\\chi^2({self.dof}) = {self.statistic:.3f}"

                    if np.any(self.expected_frequencies < 5):
                        warnings.warn(
                            "Some expected frequencies are less than 5. Consider using Fisher's exact test."
                        )

                    min_dim = min(self.contingency_table.shape) - 1
                    self.cramers_v = np.sqrt(self.statistic / (self.n_obs * min_dim))

                    expr_list: list[str] = [
                        "$",
                        f"{self.main_stat}, ",
                        f"p = {self.pvalue:.4f}, ",
                        f"Cramer's V = {self.cramers_v:.3f}, ",
                        f"n_{{obs}} = {self.n_obs}",
                        "$",
                    ]

                self.expression = "".join(expr_list)
        else:
            raise NotImplementedError(
                (
                    'Only `approach="parametric"` and `approach="nonparametric"` '
                    "have been implemented."
                )
            )

    def plot(
        self,
        *,
        orientation: str = "vertical",
        colors: list | None = None,
        show_stats: bool = True,
        show_counts: bool = True,
        plot_type: str = "stacked",
        ax: Axes | None = None,
        bar_kws: dict | None = None,
    ) -> Figure:
        """
        Plot a statistical comparison bar chart for categorical data.

        Args:
            orientation: 'vertical' or 'horizontal' orientation of plots.
            colors: List of colors for each group.
            show_stats: If True, adds statistics on the plot.
            show_counts: If True, shows sample counts in axis labels.
            plot_type: Type of bar chart ('stacked' or 'grouped').
            ax: Existing Axes to plot on. If None, uses current Axes.
            bar_kws: Keyword args for bar plot customization.

        Returns:
            A matplotlib Figure.
        """
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("`orientation` must be one of: 'vertical', 'horizontal'.")

        if plot_type not in ["stacked", "grouped"]:
            raise ValueError("`plot_type` must be one of: 'stacked', 'grouped'.")

        if ax is None:
            ax: Axes = plt.gca()

        if bar_kws is None:
            bar_kws: dict = {}

        ax: Axes = _themify(ax)

        colors: list[str] = _get_first_n_colors(colors, self.n_levels)

        if plot_type == "stacked":
            self._plot_stacked(ax, orientation, colors, bar_kws)
        else:  # grouped
            self._plot_grouped(ax, orientation, colors, bar_kws, self.contingency_df)

        if show_stats:
            ax.text(
                x=0.05,
                y=1.09,
                s=self.expression,
                size=9,
                transform=ax.transAxes,
                va="top",
            )

        if show_counts:
            totals = self.contingency_table.sum(axis=1)
            labels = [
                f"{label}\nn = {total}" for label, total in zip(self._x_levels, totals)
            ]
        else:
            labels = self._x_levels

        if orientation == "vertical":
            ax.set_xticks(range(self.n_cat), labels=labels)
            ax.set_xlabel(self._x_name)
            ax.set_ylabel("Proportion" if hasattr(self, "_df_proportion") else "Count")
        else:  # horizontal
            ax.set_yticks(range(self.n_cat), labels=labels)
            ax.set_ylabel(self._x_name)
            ax.set_xlabel("Proportion" if hasattr(self, "_df_proportion") else "Count")

        ax.legend(self._y_levels, title=self._y_name, loc="upper right")

        self.ax = ax
        return plt.gcf()

    def _plot_stacked(
        self, ax: Axes, orientation: str, colors: list[str], bar_kws: dict
    ):
        """Plot stacked bar chart."""
        bottom = None

        for i, y_level in enumerate(self._y_levels):
            values = self._df_proportion[y_level].to_numpy()

            if orientation == "horizontal":
                ax.barh(
                    list(range(self.n_cat)),
                    values,
                    left=bottom,
                    color=colors[i],
                    label=y_level,
                    **bar_kws,
                )
            else:  # vertical
                ax.bar(
                    list(range(self.n_cat)),
                    values,
                    bottom=bottom,
                    color=colors[i],
                    label=y_level,
                    **bar_kws,
                )

            if bottom is None:
                bottom = values
            else:
                bottom = bottom + values

    def _plot_grouped(
        self,
        ax: Axes,
        orientation: str,
        colors: list[str],
        bar_kws: dict,
        counts_df: Frame,
    ):
        """Plot grouped bar chart."""
        n_groups = self.n_cat
        n_bars = self.n_levels
        bar_width = 0.8 / n_bars

        for i, y_level in enumerate(self._y_levels):
            values = counts_df[y_level].to_numpy()
            positions = (
                np.arange(n_groups) + i * bar_width - (n_bars - 1) * bar_width / 2
            )

            if orientation == "horizontal":
                ax.barh(
                    positions,
                    values,
                    height=bar_width,
                    color=colors[i],
                    label=y_level,
                    **bar_kws,
                )
            else:  # vertical
                ax.bar(
                    positions,
                    values,
                    width=bar_width,
                    color=colors[i],
                    label=y_level,
                    **bar_kws,
                )


if __name__ == "__main__":
    from fleur import data

    df = data.load_mtcars("pandas")
    fig, ax = plt.subplots()

    bs = BarStats(x="cyl", y="vs", data=df)
    bs.plot(ax=ax, show_stats=True, plot_type="grouped")

    fig.savefig("cache.png", dpi=300)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy.stats as st
import numpy as np

from typing import Iterable, Literal
from narwhals.typing import SeriesT, Frame

from fleur._utils import _count_n_decimals, _InputDataHandler

import warnings


class ScatterStats:
    """
    Statistical correlation and plotting class for numerical variables.

    Attributes:
        n_obs (int): Total number of observations.
        correlation (float): Value of the correlation (Pearson, etc).
        alpha (float): Probability of rejecting a true null hypothesis.
        dof (int): Degrees of freedom for t-test.
        pvalue (float): P-value of the t-test.
        intercept (float): The intercept (estimation of beta2) in the model.
        slope (float): The slope (estimation of beta1) in the model.
        stderr_slope (float): Standard error of the slope.
        ci_lower (float): Lower bound of the confidence interval.
        ci_upper (float): Upper bound of the confidence interval.
        ax (matplotlib.axes.Axes): The main matplotlib axes.
        fig (matplotlib.figure.Figure): The matplotlib figure.
    """

    def __init__(
        self,
        x: str | SeriesT | Iterable,
        y: str | SeriesT | Iterable,
        data: Frame | None = None,
        alternative: str = "two-sided",
        effect_size: str = "pearson",
        ci: int | float = 95,
    ):
        """
        Initialize a `ScatterStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like.
            y: Colname of `data` or a Series or array-like.
            data: An optional dataframe.
            alternative: Defines the alternative hypothesis. Default
                is 'two-sided'. Must be one of 'two-sided', 'less' and 'greater'.
            effect_size: The correlation measure to use.
                Default is 'pearson'. Must be one of 'pearson',
                'kendall', 'spearman'.
            ci: Confidence level for the label and the regression
                plot. The default value is 95 (for a 95% confidence
                level).
        """
        if effect_size not in ["pearson", "kendall", "spearman"]:
            raise ValueError(
                "effect_size argument must be one of: 'pearson', 'kendall', 'spearman'."
            )

        self._data_info = _InputDataHandler(x=x, y=y, data=data).get_info()

        x_name: str = self._data_info["x_name"]
        y_name: str = self._data_info["y_name"]
        df = self._data_info["dataframe"]

        self._x_np = df[x_name].to_numpy()
        self._y_np = df[y_name].to_numpy()
        self.n_obs = len(df)

        self._fit(alternative=alternative, effect_size=effect_size, ci=ci)

    def _fit(self, alternative: str, effect_size: str, ci: int | float):
        regression = st.linregress(self._x_np, self._y_np, alternative=alternative)

        self.alpha = 1 - ci / 100
        self.dof = self.n_obs - 2

        if effect_size == "pearson":
            correlation_output: float = st.pearsonr(
                self._x_np, self._y_np, alternative=alternative
            )
            self._symbol_correl = "\\rho"
        elif effect_size == "kendall":
            correlation_output: float = st.kendalltau(
                self._x_np, self._y_np, alternative=alternative
            )
            self._symbol_correl = "\\tau"
        elif effect_size == "spearman":
            correlation_output: float = st.spearmanr(
                self._x_np, self._y_np, alternative=alternative
            )
            self._symbol_correl = "\\rho"

        self.correlation = correlation_output.statistic
        self.pvalue = regression.pvalue
        self.intercept = regression.intercept
        self.slope = regression.slope
        self.stderr_slope = regression.stderr
        self._t_critical = st.t.ppf(1 - self.alpha / 2, self.dof)
        self.ci_lower = self.slope - self._t_critical * self.stderr_slope
        self.ci_upper = self.slope + self._t_critical * self.stderr_slope

        ci_decimal: int = _count_n_decimals(ci)

        expr_list: list[str] = [
            "$",
            f"t_{{Student}}({self.dof}) = {self.slope:.2f}, ",
            f"CI_{{{ci:.{ci_decimal}f}\\%}} = [{self.ci_lower:.2f}, {self.ci_upper:.2f}], ",
            f"p = {self.pvalue:.4f}, ",
            f"{self._symbol_correl}_{{{effect_size.title()}}} = {self.correlation:.2f}, ",
            f"n_{{obs}} = {self.n_obs}",
            "$",
        ]
        self._expression = "".join(expr_list)

        expr_list: list[str] = [
            "$",
            f"\\hat{{y}}_i = {self.intercept:.2f}",
            f" {'+' if self.slope >= 0 else '-'} ",
            f"{abs(self.slope):.2f}x_i",
            "$",
        ]
        self._expression_model = "".join(expr_list)

    def plot(
        self,
        *,
        bins: int | list[int] | None = None,
        hist: bool = True,
        scatter: bool = True,
        line: bool = True,
        area: bool = True,
        scatter_kws: dict | None = None,
        line_kws: dict | None = None,
        area_kws: dict | None = None,
        hist_kws: dict | None = None,
        subplot_mosaic_kwargs: dict | None = None,
        show_stats: bool = True,
    ) -> Figure:
        r"""
        Plot a scatter plot of two variables, with a linear regression
        line and annotate it with main statistical results.

        Args:
            bins: Number of bins for the marginal distributions. This can be an integer or a list of two integers (the first for the top distribution and the second for the other).
            hist: Whether to include histograms of marginal distributions.
            scatter: Whether to include the scatter plot.
            line: Whether to include the line of the regression.
            area: Whether to include the area of the confidence interval.
            line_kws: Additional parameters which will be passed to the `plot()` function in matplotlib.
            scatter_kws: Additional parameters which will be passed to the `scatter()` function in matplotlib.
            area_kws: Additional parameters which will be passed to the `fill_between()` function in matplotlib.
            hist_kws: Additional parameters which will be passed to the `hist()` function in matplotlib.
            subplot_mosaic_kwargs: Additional keyword arguments to pass to `plt.subplot_mosaic()`. Default is `None`.
            show_stats: If True, display statistics on the plot.
        """
        if not hist and any([bins is not None, hist_kws is not None]):
            warnings.warn(
                "bins/hist_kws arguments are ignored when hist=False.",
                category=UserWarning,
            )

        default_subplot_mosaic_kwargs: dict = dict(
            width_ratios=(5, 1), height_ratios=(1, 5), figsize=(8, 6)
        )
        if subplot_mosaic_kwargs is None:
            subplot_mosaic_kwargs: dict = {}
        default_subplot_mosaic_kwargs.update(subplot_mosaic_kwargs)

        if hist:
            scheme = """
    B.
    AC
    """
            fig, axs = plt.subplot_mosaic(scheme, **default_subplot_mosaic_kwargs)
            fig.subplots_adjust(wspace=0, hspace=0)
            ax: Axes = axs["A"]  # main Axes of the Figure
        else:
            ax: Axes = plt.gca()
            fig: Figure = plt.gcf()

        self.fig = fig
        self.ax = ax

        if scatter_kws is None:
            scatter_kws: dict = {}
        if line_kws is None:
            line_kws: dict = {}
        if area_kws is None:
            area_kws: dict = {}
        area_default_kws: dict = {
            "color": plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            "alpha": 0.2,
        }
        area_default_kws.update(area_kws)

        x_values = np.linspace(np.min(self._x_np), np.max(self._x_np), 100)
        y_values = self.slope * x_values + self.intercept
        residuals = self._y_np - (self.slope * self._x_np + self.intercept)
        rse = np.sqrt(np.sum(residuals**2) / self.dof)
        x_mean = np.mean(self._x_np)
        x_var = np.sum((self._x_np - x_mean) ** 2)
        y_err = (
            self._t_critical
            * rse
            * np.sqrt(1 / self.n_obs + (x_values - x_mean) ** 2 / x_var)
        )

        if area:
            ax.fill_between(
                x_values,
                y_values - y_err,
                y_values + y_err,
                **area_default_kws,
            )
        if scatter:
            ax.scatter(self._x_np, self._y_np, **scatter_kws)
        if line:
            ax.plot(x_values, y_values, **line_kws)

        ax: Axes = self._themify(ax)

        if hist:
            if bins is None:
                bins: Literal[12] = 12

            if hist_kws is None:
                hist_kws: dict = {}

            if isinstance(bins, list):
                binsB: int = bins[0]
                binsC: int = bins[1]
            else:
                binsB = binsC = bins

            axs["B"].hist(self._x_np, bins=binsB, **hist_kws)
            axs["C"].hist(self._y_np, orientation="horizontal", bins=binsC, **hist_kws)

            axs["B"].axis("off")
            axs["C"].axis("off")

        if show_stats:
            annotation_params: dict = dict(transform=fig.transFigure, va="top")
            fig.text(x=0.1, y=0.95, s=self._expression, size=9, **annotation_params)
            fig.text(
                x=0.75,
                y=0.05,
                s=self._expression_model,
                ha="right",
                weight="bold",
                style="normal",
                size=12,
                **annotation_params,
            )

        return self.fig

    def _themify(self, ax: Axes) -> Axes:
        """
        Set the theme to a matplotlib Axes.

        Args
            ax: The matplotlib Axes to which you want to apply the theme.

        Returns
            The matplotlib Axes.
        """
        ax.grid(color="#525252", alpha=0.2, zorder=-5)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(size=0, labelsize=8)
        return ax

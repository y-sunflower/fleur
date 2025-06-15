import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import numpy as np

import warnings
from numbers import Number
from typing import Union, Optional, Iterable, List
from narwhals.typing import SeriesT, Frame

from .input_data_handling import InputDataHandler
from fleur._utils import _count_n_decimals
from fleur.utils import themify


class ScatterStats:
    """
    Statistical correlation and plotting class for numerical variables.

    Attributes:
        n_obs (int): Total number of observations.
        alpha (float): Probability of rejecting a true null hypothesis
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
        x: Union[str, SeriesT, Iterable],
        y: Union[str, SeriesT, Iterable],
        data: Optional[Frame] = None,
        alternative: str = "two-sided",
        correlation_measure: str = "pearson",
        ci: Number = 95,
    ):
        """
        Initialize a `ScatterStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like.
            y: Colname of `data` or a Series or array-like.
            data: An optional dataframe.
            alternative: Defines the alternative hypothesis. Default is 'two-sided'. Must be one of 'two-sided', 'less' and 'greater'.
            correlation_measure: The correlation measure to use. Default is 'pearson'. Must be one of 'pearson', 'kendall', 'spearman'.
            ci: Confidence level for the top label and the regression plot. The default value is 95 (for a 95% confidence level).
        """
        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError(
                "alternative argument must be one of: 'two-sided', 'less', 'greater'."
            )

        if correlation_measure not in ["pearson", "kendall", "spearman"]:
            raise ValueError(
                "correlation_measure argument must be one of: 'pearson', 'kendall', 'spearman'."
            )

        self._data_info = InputDataHandler(x=x, y=y, data=data).get_info()

        x_name = self._data_info["x_name"]
        y_name = self._data_info["y_name"]
        df = self._data_info["dataframe"]

        self._x_np = df[x_name].to_numpy()
        self._y_np = df[y_name].to_numpy()

        regression = st.linregress(self._x_np, self._y_np, alternative=alternative)

        self.n_obs = len(df)
        self.alpha = 1 - ci / 100
        self.dof = self.n_obs - 2

        if correlation_measure == "pearson":
            correlation = st.pearsonr(self._x_np, self._y_np).statistic
            self._symbol_correl = "\\rho"
        elif correlation_measure == "kendall":
            correlation = st.kendalltau(self._x_np, self._y_np).statistic
            self._symbol_correl = "\\tau"
        elif correlation_measure == "spearman":
            correlation = st.spearmanr(self._x_np, self._y_np).statistic
            self._symbol_correl = "\\rho"

        self.pvalue = regression.pvalue
        self.intercept = regression.intercept
        self.slope = regression.slope
        self.stderr_slope = regression.stderr
        self._t_critical = st.t.ppf(1 - self.alpha / 2, self.dof)
        self.ci_lower = self.slope - self._t_critical * self.stderr_slope
        self.ci_upper = self.slope + self._t_critical * self.stderr_slope

        ci_decimal = _count_n_decimals(ci)

        expr_list = [
            "$",
            f"t_{{Student}}({self.dof}) = {self.slope:.2f}, ",
            f"CI_{{{ci:.{ci_decimal}f}\\%}} = [{self.ci_lower:.2f}, {self.ci_upper:.2f}], ",
            f"p = {self.pvalue:.4f}, ",
            f"{self._symbol_correl}_{{{correlation_measure.title()}}} = {correlation:.2f}, ",
            f"n_{{obs}} = {self.n_obs}",
            "$",
        ]
        self._expression = "".join(expr_list)

        expr_list = [
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
        marginal: bool = True,
        bins: Union[int, List[int]] = None,
        scatter_kws: Union[dict, None] = None,
        line_kws: Union[dict, None] = None,
        area_kws: Union[dict, None] = None,
        hist_kws: Union[dict, None] = None,
        subplot_mosaic_kwargs: Union[dict, None] = None,
    ) -> matplotlib.figure.Figure:
        r"""
        Plot a scatter plot of two variables, with a linear regression
        line and annotate it with main statistical results.

        Args:
            bins: Number of bins for the marginal distributions. This can be an integer or a list of two integers (the first for the top distribution and the second for the other).
            marginal: Whether to include marginal histograms. Default is `True`.
            line_kws: Additional parameters which will be passed to the `plot()` function in matplotlib.
            scatter_kws: Additional parameters which will be passed to the `scatter()` function in matplotlib.
            area_kws: Additional parameters which will be passed to the `fill_between()` function in matplotlib.
            hist_kws: Additional parameters which will be passed to the `hist()` function in matplotlib.
            subplot_mosaic_kwargs: Additional keyword arguments to pass to `plt.subplot_mosaic()`. Default is `None`.
        """
        if not marginal and any([bins is not None, hist_kws is not None]):
            warnings.warn(
                "bins/hist_kws arguments are ignored when marginal=False.",
                category=UserWarning,
            )

        default_subplot_mosaic_kwargs = dict(
            width_ratios=(5, 1), height_ratios=(1, 5), figsize=(8, 6)
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
            ax = plt.gca()
            fig = plt.gcf()

        self.fig = fig
        self.ax = ax

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

        ax.fill_between(
            x_values,
            y_values - y_err,
            y_values + y_err,
            **area_default_kws,
        )
        ax.scatter(self._x_np, self._y_np, **scatter_kws)
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

            axs["B"].hist(self._x_np, bins=binsB, **hist_kws)
            axs["C"].hist(self._y_np, orientation="horizontal", bins=binsC, **hist_kws)

            axs["B"].axis("off")
            axs["C"].axis("off")

        annotation_params = dict(transform=fig.transFigure, va="top")
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

    def summary(self):
        """
        Print a text summary of the statistical test performed.

        Displays the formatted test statistic with p-value, CI, etc.

        Raises:
            RuntimeError: If `plot()` was not called before `summary()`.
        """
        print("Correlation stats\n")

        clean_expression = (
            self._expression.replace("$", "")
            .replace("{", "")
            .replace("}", "")
            .replace("/", "")
        )

        info_about_test = "Student t test of the coefficient on x"
        print(f"Test: {info_about_test}")
        print(f"{clean_expression}")

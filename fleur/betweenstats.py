import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection
from typing import Iterable, Any, cast
from narwhals.typing import SeriesT, Frame

from ._utils import (
    _infer_types,
    _beeswarm,
    _InputDataHandler,
    _get_first_n_colors,
)

import warnings


class BetweenStats:
    """
    Statistical comparison and plotting class for between-group analysis.

    This class provides functionality to visualize and statistically compare
    numerical data across two or more categorical groups. It supports t-tests
    for two groups and one-way ANOVA for three or more groups. Visualization
    options include violin plots, box plots, and swarm plots.

    Attributes:
        statistic (float): The computed test statistic (t or F).
        pvalue (float): The p-value of the statistical test.
        main_stat (str): The formatted test statistic string for display.
        expression (str): Full LaTeX-style annotation string.
        is_ANOVA (bool): True if test is ANOVA, False if t-test.
        is_paired (bool): Whether a paired test was used.
        dof (int): Degrees of freedom for t-tests.
        dof_between (int): Between-group degrees of freedom for ANOVA.
        dof_within (int): Within-group degrees of freedom for ANOVA.
        n_cat (int): Number of unique categories in the group column.
        n_obs (int): Total number of observations.
        means (list): A list with means.
        test_output: The output of the statistical test.
        ax (matplotlib.axes.Axes): The matplotlib axes used for plotting.
    """

    def __init__(
        self,
        x: str | SeriesT | Iterable,
        y: str | SeriesT | Iterable,
        data: Frame | None = None,
        paired: bool = False,
        approach: str = "parametric",
        **kwargs: Any,
    ):
        """
        Initialize a `BetweenStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like.
            y: Colname of `data` or a Series or array-like.
            data: An optional dataframe used if `x` and `y` are colnames.
            paired: Whether comparing the same observations or not.
            approach: A character specifying the type of statistical approach:
                "parametric" (default), "nonparametric", "robust", "bayes".
            kwargs: Additional arguments passed to the scipy test function.
                Either `scipy.stats.ttest_rel()`, `scipy.stats.ttest_ind()`,
                `scipy.stats.f_oneway()`, `scipy.stats.wilcoxon()`
        """
        valid_approachs: list[str] = ["parametric", "nonparametric", "robust", "bayes"]
        if approach not in valid_approachs:
            raise ValueError(
                f"`approach` must be one of {valid_approachs}, not {approach}"
            )

        self._data_info = _InputDataHandler(x=x, y=y, data=data).get_info()
        self.is_paired = paired

        x_name: str = self._data_info["x_name"]
        y_name: str = self._data_info["y_name"]
        df = self._data_info["dataframe"]

        cat_col, num_col = _infer_types(x_name, y_name, df)
        self._cat_col = cat_col
        self._num_col = num_col
        self._result = [sub_df[num_col].to_list() for _, sub_df in df.group_by(cat_col)]
        self._sample_sizes = [len(sub_df) for _, sub_df in df.group_by(cat_col)]
        self._cat_labels = df[cat_col].unique().to_list()
        self.n_cat = df[cat_col].n_unique()
        self.n_obs = len(df)
        self.means = [np.mean(group) for group in self._result]

        self._fit(approach=approach, **kwargs)

    def _fit(self, approach: str, **kwargs: Any):
        """
        Internal method to compute all the statistics and store
        them as attributes.

        Args:
            approach: A character specifying the type of statistical approach:
                "parametric" (default), "nonparametric", "robust", "bayes".
            kwargs: Additional arguments passed to the scipy test function.
                Either `scipy.stats.ttest_rel()`, `scipy.stats.ttest_ind()`,
                or `scipy.stats.f_oneway()`.
        """
        if "trim" in kwargs and approach != "robust":
            warnings.warn(
                'Using `trim` argument without expliciting `approach="robust"` is not recommended.'
            )

        if self.n_cat < 2:
            raise ValueError(
                "You must have at least 2 distinct categories in your category column"
            )
        elif self.n_cat == 2:
            self.is_ANOVA = False

            if self.is_paired:
                if approach == "parametric":
                    self.test_output = st.ttest_rel(
                        self._result[0], self._result[1], **kwargs
                    )
                    self._letter = "t"
                    self.name = "Paired t-test"
                elif approach == "nonparametric":
                    self.test_output = st.wilcoxon(
                        self._result[0], self._result[1], **kwargs
                    )
                    self._letter = "T"
                    self.name = "Wilcoxon"
                else:
                    raise NotImplementedError(
                        (
                            'Only `approach="parametric"` and `approach="nonparametric"` '
                            "have been implemented for paired samples."
                        )
                    )
            else:  # not paired
                if approach == "parametric":
                    self.test_output = st.ttest_ind(
                        self._result[0], self._result[1], **kwargs
                    )
                    self._letter = "t"
                    if "equal_var" in kwargs:
                        equal_var: bool = kwargs["equal_var"]
                        if equal_var:
                            self.name = "Student"
                        else:
                            self.name = "Welch"
                    else:
                        self.name = "Student"
                elif approach == "nonparametric":
                    self.test_output = st.mannwhitneyu(
                        self._result[0], self._result[1], **kwargs
                    )
                    self._letter = "U"
                    self.name = "Mann-Whitney"
                elif approach == "robust":
                    trim_warn_message = (
                        "Setting `approach='robust'` without setting a value "
                        "of `trim` above 0 is equivalent of using default "
                        "`approach='parametric'`. "
                        "Remove `approach='robust'` or set a value of `trim` "
                        "to hide this warning."
                    )
                    if "trim" not in kwargs:
                        warnings.warn(trim_warn_message)
                    else:
                        trim: float = kwargs["trim"]
                        if trim <= 0:
                            warnings.warn(trim_warn_message)
                    self.test_output = st.ttest_ind(
                        self._result[0], self._result[1], **kwargs
                    )
                    self._letter = "t"
                    self.name = "Yuen"
                else:
                    raise NotImplementedError(
                        (
                            'Only `approach="parametric"`, `approach="nonparametric"` '
                            'and `approach="robust"` have been implemented for '
                            "independant samples."
                        )
                    )

            self.statistic = self.test_output.statistic
            self.pvalue = self.test_output.pvalue
            if hasattr(self.test_output, "df"):  # only for t-tests
                self.dof = int(self.test_output.df)
                self.main_stat = f"t_{{Student}}({self.dof}) = {self.statistic:.2f}"
            else:
                self.dof = None
                if self.name == "Wilcoxon":
                    self._letter = "W"
                elif self.name == "Mann-Whitney":
                    self._letter = "U"

        else:  # n >= 3
            self.is_ANOVA = True
            if self.is_paired:
                raise NotImplementedError(
                    "Repeated measures ANOVA has not been implemented yet."
                )
            else:  # not paired
                if approach == "parametric":
                    if "equal_var" in kwargs:
                        if not kwargs["equal_var"]:
                            raise NotImplementedError(
                                "Welch's ANOVA is not implemented yet."
                            )
                    else:
                        self.name = "One-way"
                    self.test_output = st.f_oneway(*self._result, **kwargs)
                    self._letter = "F"
                elif approach == "nonparametric":
                    self.test_output = st.kruskal(*self._result, **kwargs)
                    self._letter = "H"
                    self.name = "Kruskal-Wallis"
                else:
                    raise NotImplementedError(
                        'Only `approach="parametric"` and `approach="nonparametric"` are implemented.'
                    )
            self.statistic = self.test_output.statistic
            self.pvalue = self.test_output.pvalue
            self.dof_between = self.n_cat - 1
            self.dof_within = self.n_obs - self.n_cat
            self.main_stat = (
                f"F({self.dof_between}, {self.dof_within}) = {self.statistic:.2f}"
            )

        self.main_stat = f"{self._letter}_{{{self.name}}} = {self.statistic:.2f}"

        expr_list: list[str] = [
            "$",
            f"{self.main_stat}, ",
            f"p = {self.pvalue:.4f}, ",
            f"n_{{obs}} = {self.n_obs}",
            "$",
        ]
        self._expression = "".join(expr_list)

    def plot(
        self,
        *,
        orientation: str = "vertical",
        colors: list | None = None,
        show_stats: bool = True,
        show_means: bool = True,
        jitter_amount: float = 0.25,
        violin: bool = True,
        box: bool = True,
        scatter: bool = True,
        violin_kws: dict | None = None,
        box_kws: dict | None = None,
        scatter_kws: dict | None = None,
        mean_kws: dict | None = dict(
            fontsize=7,
            color="black",
            bbox=dict(boxstyle="round", facecolor="#fefae0", alpha=0.7),
            zorder=50,
        ),
        mean_line_kws: dict | None = dict(ls="--", lw=0.6, color="black"),
        ax: Axes | None = None,
    ) -> Figure:
        """
        Plot and fit the `BetweenStats` class to data and render a statistical
        comparison plot. It detects how many groups you have and apply the required
        test for this number. All arguments must be passed as keyword arguments.

        Args:
            orientation: 'vertical' or 'horizontal' orientation of plots.
            colors: List of colors for each group.
            show_stats: If True, adds statistics on the plot.
            show_means: If True, adds mean labels on the plot.
            jitter_amount: Controls the horizontal spread of dots to prevent
                overlap; 0 aligns them, higher values increase spacing.
            violin: Whether to include violin plot.
            box: Whether to include box plot.
            scatter: Whether to include scatter plot of raw data.
            violin_kws: Keyword args for violinplot customization.
            box_kws: Keyword args for boxplot customization.
            scatter_kws: Keyword args for scatter plot customization.
            mean_kws: Keyword args for mean labels customization.
            mean_line_kws: Keyword arguments for the line connecting the mean
                point and the mean label.
            ax (matplotlib.axes.Axes, ): Existing Axes to plot on. If None, uses
                current Axes.

        Returns:
            A matplotlib Figure.
        """
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("`orientation` must be one of: 'vertical', 'horizontal'.")

        colors = _get_first_n_colors(colors, self.n_cat)

        if ax is None:
            ax: Axes = plt.gca()
        if violin_kws is None:
            violin_kws: dict = {}
        if box_kws is None:
            box_kws: dict = {}
        if scatter_kws is None:
            scatter_kws: dict = {}
        violin_default_kws: dict = {"orientation": orientation, "showextrema": False}
        violin_default_kws.update(violin_kws)
        box_default_kws: dict = {"orientation": orientation}
        box_default_kws.update(box_kws)
        scatter_default_kws: dict = {"alpha": 0.5}
        scatter_default_kws.update(scatter_kws)

        if violin:
            violin_artists: dict = ax.violinplot(self._result, **violin_default_kws)
            bodies: list[PolyCollection] = cast(
                list[PolyCollection], violin_artists["bodies"]
            )
            for patch, color in zip(bodies, colors):
                patch.set(color=color)

        if box:
            box_style: dict = {"color": "#3b3b3b"}
            ax.boxplot(
                self._result,
                boxprops=box_style,
                medianprops=box_style,
                capprops=box_style,
                whiskerprops=box_style,
                **box_default_kws,
            )

        if scatter:
            for i, (values, label, color) in enumerate(
                zip(self._result, self._cat_labels, colors)
            ):
                x_offsets = _beeswarm(values, width=jitter_amount)
                x_coords = np.full(len(values), i + 1) + x_offsets

                if orientation == "vertical":
                    ax.scatter(x_coords, values, color=color, **scatter_default_kws)
                else:  # "horizontal"
                    ax.scatter(values, x_coords, color=color, **scatter_default_kws)

        mean_scatter_kwargs = dict(color="#c1121f", s=100, zorder=50)
        for pos, mean in zip(range(1, len(self._result) + 1), self.means):
            if orientation == "vertical":
                ax.scatter(pos, mean, **mean_scatter_kwargs)
            else:  # horizontal
                ax.scatter(mean, pos, **mean_scatter_kwargs)

        if show_stats:
            annotation_params: dict = dict(transform=ax.transAxes, va="top")
            ax.text(x=0.05, y=1.09, s=self._expression, size=9, **annotation_params)

        ax: Axes = self._themify(ax)

        ticks: list[int] = [i + 1 for i in range(len(self._sample_sizes))]
        labels: list[str] = [
            f"{label}\nn = {n}"
            for n, label in zip(self._sample_sizes, self._cat_labels)
        ]
        if orientation == "vertical":
            ax.set_xticks(ticks, labels=labels)
        elif orientation == "horizontal":
            ax.set_yticks(ticks, labels=labels)

        self.ax = ax

        if show_means:
            shift = 1.3
            for i, mean in enumerate(self.means):
                label = f"$\\hat{{\\mu}}_{{mean}} = {mean:.2f}$"
                if orientation == "vertical":
                    ax.text(
                        x=i + shift,
                        y=mean,
                        s=label,
                        va="center",
                        ha="left",
                        **mean_kws,
                    )
                    ax.plot([i + 1, i + shift], [mean, mean], **mean_line_kws)
                else:  # horizontal
                    ax.text(
                        x=mean,
                        y=i + shift,
                        s=label,
                        va="bottom",
                        ha="center",
                        **mean_kws,
                    )
                    ax.plot([mean, mean], [i + 1, i + shift], **mean_line_kws)

        return plt.gcf()

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

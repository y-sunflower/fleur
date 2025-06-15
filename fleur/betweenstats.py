import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import numpy as np

from typing import Union, Optional, Iterable
from narwhals.typing import SeriesT, Frame

from ._utils import _infer_types
from .input_data_handling import InputDataHandler
from .utils import themify
from .upstream import _beeswarm


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
        dof_between (int): Between-group degrees of freedom (for ANOVA).
        dof_within (int): Within-group degrees of freedom (for ANOVA).
        n_cat (int): Number of unique categories in the group column.
        n_obs (int): Total number of observations.
        ax (matplotlib.axes.Axes): The matplotlib axes used for plotting.
    """

    def __init__(
        self,
        x: Union[str, SeriesT, Iterable],
        y: Union[str, SeriesT, Iterable],
        data: Optional[Frame] = None,
        paired: bool = False,
    ):
        """
        Initialize a `BetweenStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like.
            y: Colname of `data` or a Series or array-like.
            data: An optional dataframe.
            paired: If True, perform paired t-test (only for 2 groups).
        """
        self._data_info = InputDataHandler(x=x, y=y, data=data).get_info()
        self._is_fitted = False
        self.is_paired = paired

        x_name = self._data_info["x_name"]
        y_name = self._data_info["y_name"]
        df = self._data_info["dataframe"]

        cat_col, num_col = _infer_types(x_name, y_name, df)
        self._result = [sub_df[num_col].to_list() for _, sub_df in df.group_by(cat_col)]
        self._sample_sizes = [len(sub_df) for _, sub_df in df.group_by(cat_col)]
        self._cat_labels = df[cat_col].unique().to_list()
        self.n_cat = df[cat_col].n_unique()
        self.n_obs = len(df)

        if self.n_cat < 2:
            raise ValueError(
                "You must have at least 2 distinct categories in your category column"
            )
        elif self.n_cat == 2:
            self.is_ANOVA = False
            if self.is_paired:
                ttest = st.ttest_rel(self._result[0], self._result[1])
                self.name = "Paired t-test"
            else:
                ttest = st.ttest_ind(self._result[0], self._result[1])
                self.name = "T-test"
            self.statistic = ttest.statistic
            self.pvalue = ttest.pvalue
            self.dof = int(ttest.df)
            self.main_stat = f"t_{{Student}}({self.dof}) = {self.statistic:.2f}"
        else:  # n >= 3
            self.is_ANOVA = True
            if self.is_paired:
                raise NotImplementedError(
                    "Repeated measures ANOVA has not been implemented yet."
                )
            else:
                anova = st.f_oneway(*self._result)
                self.name = "One-way ANOVA"
            self.statistic = anova.statistic
            self.pvalue = anova.pvalue
            self.dof_between = self.n_cat - 1
            self.dof_within = self.n_obs - self.n_cat
            self.main_stat = (
                f"F({self.dof_between}, {self.dof_within}) = {self.statistic:.2f}"
            )

        expr_list = [
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
        colors: Optional[list] = None,
        show_stats: bool = True,
        violin: bool = True,
        box: bool = True,
        scatter: bool = True,
        violin_kws: Union[dict, None] = None,
        box_kws: Union[dict, None] = None,
        scatter_kws: Union[dict, None] = None,
        ax: Union[matplotlib.axes.Axes, None] = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot and fit the `BetweenStats` class to data and render a statistical
        comparison plot. It detects how many groups you have and apply the required
        test for this number. All arguments must be passed as keyword arguments.

        Args:
            orientation: 'vertical' or 'horizontal' orientation of plots.
            colors: List of colors for each group.
            show_stats: If True, display statistics on the plot.
            violin: Whether to include violin plot.
            box: Whether to include box plot.
            scatter: Whether to include scatter plot of raw data.
            violin_kws: Keyword args for violinplot customization.
            box_kws: Keyword args for boxplot customization.
            scatter_kws: Keyword args for scatter plot customization.
            ax (matplotlib.axes.Axes, ): Existing Axes to plot on. If None, uses current Axes.

        Returns:
            A matplotlib Figure.
        """
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError(
                "orientation argument must be one of: 'vertical', 'horizontal'."
            )

        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: self.n_cat]
        else:
            if len(colors) < self.n_cat:
                raise ValueError(
                    f"`colors` argument must have at least {self.n_cat} elements, "
                    f"not {len(colors)}"
                )
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
        scatter_default_kws.update(scatter_kws)

        if violin:
            violin_artists = ax.violinplot(self._result, **violin_default_kws)
            for patch, color in zip(violin_artists["bodies"], colors):
                patch.set(color=color)

        if box:
            box_style = {"color": "#3b3b3b"}
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
                x_offsets = _beeswarm(values)
                x_coords = np.full(len(values), i + 1) + x_offsets

                if orientation == "vertical":
                    ax.scatter(x_coords, values, color=color, **scatter_default_kws)
                else:  # "horizontal"
                    ax.scatter(values, x_coords, color=color, **scatter_default_kws)

        if show_stats:
            annotation_params = dict(transform=ax.transAxes, va="top")
            ax.text(x=0.05, y=1.09, s=self._expression, size=9, **annotation_params)

        ax = themify(ax)

        ticks = [i + 1 for i in range(len(self._sample_sizes))]
        labels = [
            f"{label}\nn = {n}"
            for n, label in zip(self._sample_sizes, self._cat_labels)
        ]
        if orientation == "vertical":
            ax.set_xticks(ticks, labels=labels)
        elif orientation == "horizontal":
            ax.set_yticks(ticks, labels=labels)

        self.ax = ax

        return plt.gcf()

    def summary(self):
        """
        Print a text summary of the statistical test performed.

        Displays the type of test conducted (t-test or ANOVA), number of groups,
        and the formatted test statistic with p-value and sample size.
        """
        print("Between stats comparison\n")

        info_about_test = [
            f"{self.name} ",
            f"with {self.n_cat} groups" if self.is_ANOVA else "",
        ]
        info_about_test = "".join(info_about_test)

        clean_expression = (
            self._expression.replace("$", "").replace("{", "").replace("}", "")
        )
        print(f"Test: {info_about_test}")
        print(clean_expression)

import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import numpy as np

from typing import Union, Optional, Iterable
from narwhals.typing import SeriesT, Frame

from ._utils import _infer_types
from .input_data_handling import InputDataHandler
from .utils import themify


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
    ):
        """
        Initialize a `BetweenStats()` instance.

        Args:
            x: Colname of `data` or a Series or array-like.
            y: Colname of `data` or a Series or array-like.
            data: An optional dataframe.
        """
        self._data_info = InputDataHandler(x=x, y=y, data=data).get_info()
        self._is_fitted = False

    def plot(
        self,
        orientation: str = "vertical",
        paired: bool = False,
        colors: Optional[list] = None,
        violin: bool = True,
        box: bool = True,
        scatter: bool = True,
        violin_kws: Union[dict, None] = None,
        box_kws: Union[dict, None] = None,
        scatter_kws: Union[dict, None] = None,
        ax: Union[matplotlib.axes.Axes, None] = None,
    ):
        """
        Plot and fit the BetweenStats class to data and render a statistical
        comparison plot.

        `plot()` detects how many groups you have and apply the required test
        for this number. For paired groups, use the `paired` argument.

        Args:
            orientation (str): 'vertical' or 'horizontal' orientation of plots.
            paired (bool): If True, perform paired t-test (only for 2 groups).
            colors (list, optional): List of colors for each group.
            violin (bool): Whether to include violin plot.
            box (bool): Whether to include box plot.
            scatter (bool): Whether to include scatter plot of raw data.
            violin_kws (dict, optional): Keyword args for violinplot customization.
            box_kws (dict, optional): Keyword args for boxplot customization.
            scatter_kws (dict, optional): Keyword args for scatter plot customization.
            ax (matplotlib.axes.Axes, optional): Existing Axes to plot on. If None, uses current Axes.

        Raises:
            ValueError: If the orientation is invalid or less than 2 categories are provided.
            NotImplementedError: If a repeated measures ANOVA is requested.
        """
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError(
                "orientation argument must be one of: 'vertical', 'horizontal'."
            )

        x_name = self._data_info["x_name"]
        y_name = self._data_info["y_name"]
        df = self._data_info["dataframe"]

        cat_col, num_col = _infer_types(x_name, y_name, df)
        result = [sub_df[num_col].to_list() for _, sub_df in df.group_by(cat_col)]
        sample_sizes = [len(sub_df) for _, sub_df in df.group_by(cat_col)]
        cat_labels = df[cat_col].unique().to_list()
        n_cat = df[cat_col].n_unique()
        n = len(df)

        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_cat]
        else:
            if len(colors) < n_cat:
                raise ValueError(
                    f"`colors` argument must have at least {n_cat} elements, "
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
        scatter_default_kws.update(box_kws)

        if violin:
            violin_artists = ax.violinplot(result, **violin_default_kws)

        if box:
            box_style = {"color": "#3b3b3b"}
            ax.boxplot(
                result,
                boxprops=box_style,
                medianprops=box_style,
                capprops=box_style,
                whiskerprops=box_style,
                **box_default_kws,
            )

        if scatter:
            for i, (values, label, color) in enumerate(zip(result, cat_labels, colors)):
                jitter = np.random.uniform(low=-0.1, high=0.1, size=len(values))
                x_coords = np.full(len(values), i) + jitter + 1
                if orientation == "vertical":
                    ax.scatter(x_coords, values, color=color, **scatter_default_kws)
                else:  # "horizontal"
                    ax.scatter(values, x_coords, color=color, **scatter_default_kws)

        for patch, color in zip(violin_artists["bodies"], colors):
            patch.set(color=color)

        if n_cat < 2:
            raise ValueError(
                "You must have at least 2 distinct categories in your category column"
            )
        elif n_cat == 2:
            self.is_ANOVA = False
            if paired:
                ttest = st.ttest_rel(result[0], result[1])
                self.name = "Paired t-test"
            else:
                ttest = st.ttest_ind(result[0], result[1])
                self.name = "T-test"
            statistic = ttest.statistic
            pvalue = ttest.pvalue
            dof = int(ttest.df)
            self.dof = dof
            main_stat = f"t_{{Student}}({dof}) = {statistic:.2f}"
        else:  # n >= 3
            self.is_ANOVA = True
            if paired:
                raise NotImplementedError(
                    "Repeated measures ANOVA has not been implemented yet."
                )
            else:
                anova = st.f_oneway(*result)
                self.name = "One-way ANOVA"
            statistic = anova.statistic
            pvalue = anova.pvalue
            dof_between = n_cat - 1
            dof_within = n - n_cat
            self.dof_between = dof_between
            self.dof_within = dof_within
            main_stat = f"F({dof_between}, {dof_within}) = {statistic:.2f}"

        expr_list = [
            "$",
            f"{main_stat}, ",
            f"p = {pvalue:.4f}, ",
            f"n_{{obs}} = {n}",
            "$",
        ]

        all_expr = "".join(expr_list)

        annotation_params = dict(transform=ax.transAxes, va="top")
        ax.text(x=0.05, y=1.09, s=all_expr, size=9, **annotation_params)

        ax = themify(ax)

        ticks = [i + 1 for i in range(len(sample_sizes))]
        labels = [f"{label}\nn = {n}" for n, label in zip(sample_sizes, cat_labels)]
        if orientation == "vertical":
            ax.set_xticks(ticks, labels=labels)
        elif orientation == "horizontal":
            ax.set_yticks(ticks, labels=labels)

        self.statistic = statistic
        self.pvalue = pvalue
        self.main_stat = main_stat
        self.ax = ax
        self.is_paired = paired
        self.expression = all_expr
        self.n_cat = n_cat
        self.n_obs = n
        self._is_fitted = True

        return self

    def summary(self):
        """
        Print a text summary of the statistical test performed.

        Displays the type of test conducted (t-test or ANOVA), number of groups,
        and the formatted test statistic with p-value and sample size.

        Raises:
            RuntimeError: If `plot()` was not called before `summary()`.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call 'plot()' before calling 'summary()'.")

        print("Between stats comparison\n")

        info_about_test = [
            f"{self.name} ",
            f"with {self.n_cat} groups" if self.is_ANOVA else "",
        ]
        info_about_test = "".join(info_about_test)

        clean_expression = (
            self.expression.replace("$", "").replace("{", "").replace("}", "")
        )
        print(f"Test: {info_about_test}")
        print(clean_expression)


if __name__ == "__main__":
    from fleur import datasets

    df = datasets.load_iris()
    df = df.rename(columns={"species": "x", "sepal_length": "y"})

    fig, ax = plt.subplots(dpi=200)
    bs = BetweenStats(x=df["x"], y=df["y"])
    bs.plot()
    bs.summary()

    fig.savefig("cache.png", dpi=300, bbox_inches="tight")

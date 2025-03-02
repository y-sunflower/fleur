import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import numpy as np
import seaborn as sns
from typing import Union
import pandas as pd


def ttestviz(
    x: str,
    y: str,
    data: pd.DataFrame,
    kind: Union[str, list] = "violin",
    violin_kwargs: Union[dict, None] = None,
    ax: matplotlib.axes.Axes = None,
) -> matplotlib.axes.Axes:
    if ax is None:
        ax = plt.gca()

    if violin_kwargs is None:
        violin_kwargs = {}

    group_var = y if pd.api.types.is_numeric_dtype(data[x]) else x
    value_var = x if pd.api.types.is_numeric_dtype(data[x]) else y

    groups = data[group_var].unique()

    sample1 = data[data[group_var] == groups[0]][value_var].values
    sample2 = data[data[group_var] == groups[1]][value_var].values

    ttest_output = stats.ttest_ind(sample1, sample2)
    pvalue = ttest_output.pvalue

    if kind == "violin":
        sns.violinplot(x=x, y=y, data=data, ax=ax, **violin_kwargs)

    ax.text(x=0.1, y=1, s=f"pvalue: {pvalue:.4f}", transform=ax.transAxes)

    return ttest_output


if __name__ == "__main__":
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "group": ["A"] * 500 + ["B"] * 500,
            "value": np.concatenate(
                [
                    stats.norm.rvs(loc=5, scale=10, size=500),
                    stats.norm.rvs(loc=15, scale=10, size=500),
                ]
            ),
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    out1 = ttestviz("group", "value", data, ax=ax1)
    out2 = ttestviz("value", "group", data, ax=ax2)

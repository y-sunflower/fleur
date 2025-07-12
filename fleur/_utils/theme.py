import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


def _get_first_n_colors(colors: list[str] | None, n_cat: int) -> list[str]:
    if colors is None:
        colors: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_cat]
    else:
        if len(colors) < n_cat:
            raise ValueError(
                f"`colors` argument must have at least {n_cat} elements, "
                f"not {len(colors)}"
            )

    return colors


def set_rcParams():
    params = {
        "axes.prop_cycle": cycler(
            "color",
            [
                "#855C75FF",
                "#D9AF6BFF",
                "#AF6458FF",
                "#736F4CFF",
                "#526A83FF",
                "#625377FF",
                "#68855CFF",
                "#9C9C5EFF",
                "#A06177FF",
                "#8C785DFF",
                "#467378FF",
                "#7C7C7CFF",
            ],
        ),
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.color": "#525252",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "ytick.labelsize": 8,
        "xtick.labelsize": 8,
    }
    plt.rcParams.update(params)


def reset_rcParams():
    plt.rcParams.update(mpl.rcParamsDefault)

from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def _themify(ax: Axes) -> Axes:
    """
    Set the fleur theme to a matplotlib Axes.

    Args
    ax: The matplotlib Axes to which you want to apply the theme.

    Returns
        The matplotlib Axes.
    """
    ax.grid(color="#525252", alpha=0.2, zorder=-5)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(size=0, labelsize=8)
    return ax


def _get_first_n_colors(colors, n_cat):
    if colors is None:
        colors: list = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n_cat]
    else:
        if len(colors) < n_cat:
            raise ValueError(
                f"`colors` argument must have at least {n_cat} elements, "
                f"not {len(colors)}"
            )

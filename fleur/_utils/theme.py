from matplotlib.axes import Axes


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

import matplotlib as mpl


def themify(ax: mpl.axes.Axes) -> mpl.axes.Axes:
    """
    Set the fleur theme to a matplotlib Axes.

    :param ax: The matplotlib Axes to which you want to apply the theme.
    :return: The matplotlib Axes.

    Examples
    --------

    .. plot::

        import matplotlib.pyplot as plt
        from fleur.utils import themify

        fig, ax = plt.subplots()
        ax = themify(ax)
    """
    ax.grid(color="#525252", alpha=0.2, zorder=-5)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(size=0, labelsize=8)
    return ax

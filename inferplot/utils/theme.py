import matplotlib as mpl


def themify(ax: mpl.axes.Axes):
    ax.grid(color="#525252", alpha=0.2, zorder=-5)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(size=0, labelsize=8)
    return ax

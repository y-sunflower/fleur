import matplotlib.pyplot as plt


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

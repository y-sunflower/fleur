import numpy as np


def _beeswarm(y, width):
    """
    Computes x-coordinates for a beeswarm plot, given a set of y-values.

    Args:
        y (array-like): The y-values (e.g., numerical data).
        nbins (int, optional): Number of bins for dividing the y-axis.
        width (float): Maximum horizontal spread of the swarm.

    Returns:
        np.ndarray: x-offsets corresponding to each y-value.
    """
    y = np.asarray(y)
    nbins = int(np.ceil(len(y) / 6))

    counts, bin_edges = np.histogram(y, bins=nbins)
    max_bin_count = counts.max()

    x_offsets = np.zeros_like(y, dtype=float)
    dx = width / (max_bin_count // 2 + 1e-5)  # add epsilon to avoid div by zero

    for ymin, ymax in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = np.where((y > ymin) & (y <= ymax))[0]
        if len(in_bin) == 0:
            continue

        sorted_indices = in_bin[np.argsort(y[in_bin])]
        left = sorted_indices[::2]
        right = sorted_indices[1::2]

        x_offsets[left] = dx * (0.5 + np.arange(len(left)))
        x_offsets[right] = -dx * (0.5 + np.arange(len(right)))

    return x_offsets

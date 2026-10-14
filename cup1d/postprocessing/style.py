"""Style plotting implementations; scientific objects retain compatibility wrappers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_discrete_cmap(n, base_cmap="jet"):
    """Construct a discrete colormap sampled from a Matplotlib colormap.

    Parameters
    ----------
    n : int
        Number of equally spaced colors in the output lookup table.
    base_cmap : str, default: "jet"
        Matplotlib colormap name to sample.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap with exactly ``n`` colors.
    """
    # ``matplotlib.cm.get_cmap`` was removed in Matplotlib 3.10.  The pyplot
    # interface remains supported and accepts the requested lookup-table size.
    cmap = plt.get_cmap(base_cmap, lut=n)
    return ListedColormap(cmap(np.linspace(0, 1, n)))

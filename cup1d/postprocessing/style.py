"""House style of the cup1d figures.

Nothing here runs at import time: the plotting entry points call
:func:`apply_house_style` (or use :func:`house_style` as a context manager),
so the fonts no longer depend on which module happened to be imported first.
"""

from contextlib import contextmanager

import matplotlib as mpl
import numpy as np

# STIX text and mathtext, as in the DESI DR1 P1D papers
HOUSE_STYLE = {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}


def apply_house_style():
    """Set the cup1d house style (STIX fonts) in the global ``rcParams``.

    Plotting classes call it when they start drawing; call it yourself before
    drawing with plain matplotlib to get the same fonts.
    """
    mpl.rcParams.update(HOUSE_STYLE)


@contextmanager
def house_style():
    """Context manager: the house style inside the block, the previous settings after.

    Examples
    --------
    >>> with house_style():
    ...     fig = plot_something()
    """
    with mpl.rc_context(HOUSE_STYLE):
        yield


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
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = plt.get_cmap(base_cmap, lut=n)
    return ListedColormap(cmap(np.linspace(0, 1, n)))


def styled(function):
    """Decorator: apply the house style before a plotting function draws."""
    from functools import wraps

    @wraps(function)
    def wrapper(*args, **kwargs):
        apply_house_style()
        return function(*args, **kwargs)

    return wrapper

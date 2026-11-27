"""Saving figures and the arrays behind them."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, stem, formats=("pdf", "png"), close=False, **kwargs):
    """Save a figure in several formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    stem : str or path-like
        Path without extension; its folder is created if needed.
    formats : sequence of str, default=("pdf", "png")
        File extensions.
    close : bool, default=False
        Close the figure afterwards.
    **kwargs
        :meth:`matplotlib.figure.Figure.savefig` options
        (``bbox_inches="tight"`` by default).

    Returns
    -------
    list of pathlib.Path
        Written files.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    paths = []
    for fmt in formats:
        path = stem.with_name(f"{stem.name}.{fmt}")
        fig.savefig(path, **kwargs)
        paths.append(path)
    if close:
        plt.close(fig)
    return paths


def zenodo_directory():
    """Folder of the arrays published with the papers (``data/zenodo`` of cup1d).

    The environment variable ``CUP1D_ZENODO_DIR`` overrides it (e.g. to
    check the figure notebooks without touching the published files).
    """
    from cup1d.utils.utils import get_path_repo

    override = os.environ.get("CUP1D_ZENODO_DIR")
    if override:
        return Path(override)
    return Path(get_path_repo("cup1d")) / "data" / "zenodo"


def save_figure_data(data, filename, directory=None):
    """Save the arrays behind a figure (``np.save`` of a dict).

    Parameters
    ----------
    data : mapping
        Plotted arrays.
    filename : str
        File name, e.g. ``"fig_19.npy"``.
    directory : str or path-like, optional
        Output folder; :func:`zenodo_directory` by default.

    Returns
    -------
    pathlib.Path
        Written file.
    """
    directory = Path(directory) if directory is not None else zenodo_directory()
    os.makedirs(directory, exist_ok=True)
    path = directory / filename
    np.save(path, dict(data), allow_pickle=True)
    return path

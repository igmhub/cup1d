"""Best-fitting star parameters against the emulator training simulations."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from cup1d.utils.utils import get_path_repo

STAR = ("Delta2_star", "n_star", "alpha_star")


def simulation_star_params(emulator, nyx_version="Jul2024"):
    """Star parameters of the training simulations of an emulator.

    Parameters
    ----------
    emulator : object
        cup1d/LaCE emulator exposing ``list_sim_cube``.
    nyx_version : str, default: "Jul2024"
        Version of the Nyx cosmology file.

    Returns
    -------
    dict
        ``{simulation label: {"Delta2_star", "n_star", "alpha_star"}}``.
    """
    suite = emulator.list_sim_cube[0][:3]
    if suite == "mpg":
        fname = os.path.join(get_path_repo("lace"), "data", "sim_suites", "Australia20", "mpg_emu_cosmo.npy")
    elif suite == "nyx":
        from lace.configuration import get_nyx_path

        fname = os.path.join(get_nyx_path(), f"nyx_emu_cosmo_{nyx_version}.npy")
    else:
        raise ValueError(f"unknown simulation suite {suite!r}")
    cosmo = np.load(fname, allow_pickle=True).item()
    return {label: {name: float(entry["star_params"][name]) for name in STAR}
            for label, entry in cosmo.items()}


def _ellipse(ax, center, covariance, nsigma, color, alpha):
    """Gaussian contour of a 2x2 covariance (nsigma: radius in sigmas)."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues < 0):
        raise ValueError("covariance of the star parameters is not positive semidefinite")
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    ax.add_patch(Ellipse(center, width=2 * nsigma * np.sqrt(eigenvalues[1]),
                         height=2 * nsigma * np.sqrt(eigenvalues[0]), angle=angle,
                         facecolor=color, edgecolor=color, alpha=alpha, lw=1.5, zorder=2))


def plot_star_simulations(simulations, fits, nsigma=(1, 2), fontsize=16, annotate=True, ax=None):
    """Delta2_star--n_star of the simulations with best fits and their errors.

    Parameters
    ----------
    simulations : dict
        Output of :func:`simulation_star_params`.
    fits : sequence of dict
        One per best fit: ``label``, ``star`` (``{name: value}``) and
        optionally ``covariance`` (3x3, order of :data:`STAR`, e.g. the
        Gauss--Newton ``mle_cosmo_covariance``), ``color`` and ``marker``.
    nsigma : sequence of float, default: (1, 2)
        Radii of the error ellipses, in standard deviations (Gaussian
        approximation: 1 sigma here is a 39% contour in 2D).
    annotate : bool, default: True
        Write the simulation numbers next to the points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    d2 = np.array([s["Delta2_star"] for s in simulations.values()])
    ns = np.array([s["n_star"] for s in simulations.values()])
    ax.scatter(d2, ns, color="0.5", s=25, label="training simulations", zorder=1)
    if annotate:
        dx = 0.01 * np.ptp(d2)
        for label, x, y in zip(simulations, d2, ns):
            ax.annotate(label.split("_")[-1], (x + dx, y), fontsize=8, color="0.4")

    for i, fit in enumerate(fits):
        color = fit.get("color", f"C{i}")
        center = (fit["star"]["Delta2_star"], fit["star"]["n_star"])
        covariance = fit.get("covariance")
        if covariance is not None:
            for j, n in enumerate(sorted(nsigma)):
                _ellipse(ax, center, np.asarray(covariance)[:2, :2], n, color, alpha=0.35 / (j + 1))
        ax.scatter(*center, marker=fit.get("marker", "X"), s=120, color=color, edgecolor="k",
                   zorder=3, label=fit["label"])

    ax.set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
    ax.set_ylabel(r"$n_\star$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 4)
    ax.legend(fontsize=fontsize - 5, loc="best")
    fig.tight_layout()
    return fig

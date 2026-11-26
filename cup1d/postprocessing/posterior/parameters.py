"""Model parameters against the compressed parameters, and their correlations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cup1d.postprocessing.posterior.samples import STAR_PARAMETERS, family_of
from cup1d.postprocessing.posterior.stats import density_contours, padded_range

# group name -> history families, in plotting order. Families absent from a
# run are skipped; parameters in no group end up in "Other".
PARAMETER_GROUPS = {
    "Cosmology": ["As", "ns", "nrun"],
    "Mean flux": ["tau_eff"],
    "Thermal broadening": ["sigT_kms"],
    "Temperature-density slope": ["gamma"],
    "Pressure smoothing": ["kF_kms"],
    r"Ly$\alpha$-Si metals": ["f_Lya_SiIII", "s_Lya_SiIII", "f_Lya_SiII", "s_Lya_SiII"],
    "Si-Si metals": ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb", "f_SiIIa_SiIII", "f_SiIIb_SiIII"],
    "Kaiser metals": ["b_SiIII", "b_SiIIa", "b_SiIIb", "b_SiIIc", "beta_SiIII", "beta_SiII",
                      "sig_SiIII", "sig_SiII", "f_SiIIxSiII", "sig_SiIIxSiII"],
    "HCD": ["HCD_damp1", "HCD_damp2", "HCD_damp3", "HCD_damp4", "HCD_const"],
    "Resolution": ["R_coeff"],
}


def group_parameters(samples, groups=None, include_cosmology=False):
    """Split the sampled parameters of a posterior into plotting groups.

    Parameters
    ----------
    samples : PosteriorSamples
        Posterior; only its sampled parameters are grouped.
    groups : mapping, optional
        Group name to list of families; :data:`PARAMETER_GROUPS` by default.
    include_cosmology : bool, default=False
        Keep the "Cosmology" group (``As``, ``ns``), which is closely tied to
        the compressed parameters themselves.

    Returns
    -------
    dict
        Group name to the parameter names present, ordered by family then
        index. Parameters in no group are collected under "Other".
    """
    groups = PARAMETER_GROUPS if groups is None else groups
    remaining = [name for name in samples.sampled if name not in STAR_PARAMETERS]
    out = {}
    for group, families in groups.items():
        members = [name for family in families for name in remaining if family_of(name) == family]
        remaining = [name for name in remaining if name not in members]
        if members and (include_cosmology or group != "Cosmology"):
            out[group] = members
    if remaining:
        out["Other"] = remaining
    return out


def _range_with_bounds(values, weights, bounds):
    """Plot range of a parameter, reaching a prior bound when the posterior does."""
    low, high = padded_range(values, weights)
    if bounds is not None:
        a, b = bounds
        width = b - a
        if low < a + 0.05 * width:
            low = a - 0.02 * width
        if high > b - 0.05 * width:
            high = b + 0.02 * width
    return low, high


def plot_parameters_vs_star(samples, names, title=None, fontsize=16, show_correlation=True):
    """Plot the joint posterior of each parameter with Delta2_star and n_star.

    Parameters
    ----------
    samples : PosteriorSamples
        Must contain ``Delta2_star``, ``n_star`` and ``names``.
    names : sequence of str
        Parameters, one column of panels each.
    title : str, optional
        Figure title.
    fontsize : float, default=16
        Base font size.
    show_correlation : bool, default=True
        Write the weighted correlation coefficient in every panel.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Rows are Delta2_star (top) and n_star (bottom); contours enclose 68 and 95
    per cent. Dotted vertical lines mark the flat-prior bounds when they fall
    within the plotted range.
    """
    w = samples.weights
    stars = [samples.column(name) for name in STAR_PARAMETERS]
    star_ranges = [padded_range(v, w) for v in stars]
    fig, axes = plt.subplots(2, len(names), figsize=(2.8 * len(names) + 1.2, 5.6),
                             sharey="row", squeeze=False)
    for j, name in enumerate(names):
        values = samples.column(name)
        bounds = samples.bounds.get(name)
        xrange = _range_with_bounds(values, w, bounds)
        for i, star in enumerate(stars):
            ax = axes[i, j]
            X, Y, H, thresholds = density_contours(values, star, w, ranges=[xrange, star_ranges[i]])
            ax.contourf(X, Y, H, levels=thresholds, cmap="Blues", alpha=0.8)
            ax.contour(X, Y, H, levels=thresholds[:-1], colors="C0", linewidths=1)
            if bounds is not None:
                for edge in bounds:
                    if xrange[0] <= edge <= xrange[1]:
                        ax.axvline(edge, color="k", ls=":", lw=1)
            if show_correlation:
                r = np.cov(values, star, aweights=w)
                r = r[0, 1] / np.sqrt(r[0, 0] * r[1, 1])
                ax.text(0.95 if r < 0 else 0.05, 0.93, rf"$r={r:.2f}$", transform=ax.transAxes,
                        ha="right" if r < 0 else "left", va="top", fontsize=fontsize - 4)
            ax.set_xlim(xrange)
            ax.set_ylim(star_ranges[i])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(labelsize=fontsize - 5)
            if i == 0:
                ax.tick_params(labelbottom=False)
        axes[1, j].set_xlabel(samples.label_of(name), fontsize=fontsize)
    for i, name in enumerate(STAR_PARAMETERS):
        axes[i, 0].set_ylabel(samples.label_of(name), fontsize=fontsize)
    if title:
        fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08, hspace=0.05)
    return fig


def plot_correlation_matrix(samples, names=None, threshold=0.35, fontsize=12):
    """Plot the posterior correlation matrix of all the parameters.

    Parameters
    ----------
    samples : PosteriorSamples
        Posterior.
    names : sequence of str, optional
        Parameters in plotting order; by default Delta2_star and n_star (if
        present) followed by the sampled parameters, grouped as in
        :func:`group_parameters` (cosmology included).
    threshold : float, default=0.35
        Coefficients with larger absolute value are written in their cell.
    fontsize : float, default=12
        Base font size.

    Returns
    -------
    matplotlib.figure.Figure
        Lower triangle of the weighted correlation matrix.
    """
    if names is None:
        names = [name for name in STAR_PARAMETERS if name in samples]
        for members in group_parameters(samples, include_cosmology=True).values():
            names += members
    matrix = samples.correlation(names)
    n = len(names)
    size = max(8.0, 0.32 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    masked = np.ma.masked_where(np.triu(np.ones((n, n), dtype=bool), k=1), matrix)
    image = ax.imshow(masked, cmap="bwr", vmin=-1, vmax=1)
    labels = [samples.label_of(name) for name in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize - 2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=fontsize - 2)
    for i in range(n):
        for j in range(i):
            if abs(matrix[i, j]) > threshold:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=max(fontsize - 7, 4))
    # group separators
    families = [family_of(name) for name in names]
    for i in range(1, n):
        if families[i] != families[i - 1]:
            ax.axhline(i - 0.5, color="0.6", lw=0.5)
            ax.axvline(i - 0.5, color="0.6", lw=0.5)
    cax = ax.inset_axes([0.55, 0.9, 0.42, 0.03])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label("Correlation", fontsize=fontsize)
    bar.ax.tick_params(labelsize=fontsize - 2)
    if samples.label:
        ax.set_title(samples.label, fontsize=fontsize + 2)
    fig.tight_layout()
    return fig

"""Posterior of the compressed linear-power parameters Delta2_star and n_star."""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2 as chi2_distribution, norm

from cup1d.postprocessing.labels import STAR_LABELS
from cup1d.postprocessing.literature import (  # noqa: F401 - re-exported
    DR1, LITERATURE, gaussian_chi2 as _gaussian_chi2, gaussian_constraint,
)
from cup1d.postprocessing.posterior.samples import STAR_PARAMETERS
from cup1d.postprocessing.style import styled
from cup1d.postprocessing.posterior.stats import (
    density_contours, format_quantiles, marginal_density, padded_range,
)

# Delta chi2 of the 68 and 95 per cent regions of a two-parameter Gaussian
GAUSSIAN_LEVELS = tuple(chi2_distribution.ppf([0.68, 0.95], 2))



def _contourf(ax, x, y, weights, cmap="Blues", color="C0", ranges=None, bins=40, smooth=1.0):
    """Filled 68/95 per cent contours of weighted samples."""
    X, Y, H, thresholds = density_contours(x, y, weights, bins=bins, smooth=smooth, ranges=ranges)
    ax.contourf(X, Y, H, levels=thresholds, cmap=cmap, alpha=0.8)
    ax.contour(X, Y, H, levels=thresholds[:-1], colors=color, linewidths=1.5)


@styled
def plot_star_posterior(samples, fontsize=16, color="C0", truth=None):
    """Plot the posterior of Delta2_star and n_star with their marginals.

    Parameters
    ----------
    samples : PosteriorSamples
        Must contain ``Delta2_star`` and ``n_star``.
    fontsize : float, default=16
        Base font size.
    color : str, default="C0"
        Colour of the marginals and contour lines.
    truth : mapping, optional
        ``{"Delta2_star": value, "n_star": value}`` drawn as dotted lines.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The titles give the median and the 16 and 84 per cent weighted quantiles;
    the contours enclose 68 and 95 per cent of the posterior.
    """
    x, y = (samples.column(name) for name in STAR_PARAMETERS)
    w = samples.weights
    ranges = [padded_range(x, w), padded_range(y, w)]
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    axes[0, 1].set_visible(False)
    for i, (values, name) in enumerate(zip((x, y), STAR_PARAMETERS)):
        ax = axes[i, i]
        centres, density = marginal_density(values, w, value_range=ranges[i])
        ax.plot(centres, density, color=color, lw=2)
        for q, ls in zip(samples.quantiles(name), ("--", "-", "--")):
            ax.axvline(q, color=color, ls=ls, lw=1, alpha=0.7)
        ax.set_title(rf"{samples.label_of(name)} $= {format_quantiles(values, w)}$",
                     fontsize=fontsize)
        ax.set_yticks([])
        ax.set_xlim(ranges[i])
        ax.set_ylim(0, 1.1)
    joint = axes[1, 0]
    _contourf(joint, x, y, w, color=color, ranges=ranges)
    joint.set_xlim(ranges[0])
    joint.set_ylim(ranges[1])
    joint.set_xlabel(samples.label_of("Delta2_star"), fontsize=fontsize)
    joint.set_ylabel(samples.label_of("n_star"), fontsize=fontsize)
    axes[1, 1].set_xlabel(samples.label_of("n_star"), fontsize=fontsize)
    axes[0, 0].tick_params(labelbottom=False)
    r = samples.correlation(STAR_PARAMETERS)[0, 1]
    joint.text(0.05, 0.95, rf"$r={r:.2f}$", transform=joint.transAxes, va="top", fontsize=fontsize - 2)
    if truth is not None:
        joint.axvline(truth["Delta2_star"], color="k", ls=":")
        joint.axhline(truth["n_star"], color="k", ls=":")
        axes[0, 0].axvline(truth["Delta2_star"], color="k", ls=":")
        axes[1, 1].axvline(truth["n_star"], color="k", ls=":")
    for ax in axes.ravel():
        ax.tick_params(labelsize=fontsize - 4)
        # avoid colliding tick labels where the bottom panels meet
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    if samples.label:
        fig.suptitle(samples.label, fontsize=fontsize)
    return fig


@styled
def plot_star_literature(samples, literature=tuple(LITERATURE), include_dr1=True,
                         fontsize=16, xlim=(0.23, 0.52), ylim=(-2.39, -2.24), color="C0"):
    """Compare the Delta2_star--n_star posterior with published constraints.

    Parameters
    ----------
    samples : PosteriorSamples
        Must contain ``Delta2_star`` and ``n_star`` without blinding offsets.
    literature : sequence of str, default: every key of :data:`LITERATURE`
        Earlier constraints (SDSS, BOSS, eBOSS), drawn as Gaussians.
    include_dr1 : bool, default=True
        Overlay DESI DR1 (Chaves-Montero+26) as a two-dimensional Gaussian with
        the published means, errors and correlation, not the full posterior.
    fontsize : float, default=16
        Base font size.
    xlim, ylim : tuple of float
        Initial axis ranges (those of the DR1 paper), enlarged to contain the
        95 per cent region of ``samples``.
    color : str, default="C0"
        Colour of this posterior.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the star parameters of ``samples`` are blinded: a comparison with
        external measurements is only meaningful without blinding offsets.
    """
    if samples.blinded:
        raise ValueError("the star parameters are blinded; unblind them to compare with the literature")
    x, y = (samples.column(name) for name in STAR_PARAMETERS)
    w = samples.weights
    lo, hi = padded_range(x, w, 0.01, 0.99, 0.2)
    xlim = (min(xlim[0], lo), max(xlim[1], hi))
    lo, hi = padded_range(y, w, 0.01, 0.99, 0.2)
    ylim = (min(ylim[0], lo), max(ylim[1], hi))

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    joint, amplitude, slope = axes[1, 0], axes[0, 0], axes[1, 1]
    axes[0, 1].axis("off")
    n_grid, d_grid = np.mgrid[ylim[0]:ylim[1]:300j, xlim[0]:xlim[1]:300j]
    x_grid = np.linspace(*xlim, 500)
    y_grid = np.linspace(*ylim, 500)
    handles = []
    references = [LITERATURE[name] for name in literature]
    if include_dr1:
        references.append(DR1)
    for function_name, label, line_color, ls in references:
        joint.contour(d_grid, n_grid, _gaussian_chi2(function_name, n_grid, d_grid),
                      levels=GAUSSIAN_LEVELS, colors=line_color, linewidths=[2.5, 1.5],
                      linestyles=ls, alpha=0.8)
        summary = gaussian_constraint(function_name)
        pdf = norm.pdf(x_grid, summary["Delta2_star"], summary["Delta2_star_err"])
        amplitude.plot(x_grid, pdf / pdf.max(), color=line_color, ls=ls, lw=2, alpha=0.8)
        pdf = norm.pdf(y_grid, summary["n_star"], summary["n_star_err"])
        slope.plot(y_grid, pdf / pdf.max(), color=line_color, ls=ls, lw=2, alpha=0.8)
        handles.append(mlines.Line2D([], [], color=line_color, ls=ls, lw=2, label=label))

    _contourf(joint, x, y, w, color=color, ranges=[xlim, ylim], bins=60)
    centres, density = marginal_density(x, w, value_range=xlim, bins=120)
    amplitude.plot(centres, density, color=color, lw=3)
    centres, density = marginal_density(y, w, value_range=ylim, bins=120)
    slope.plot(centres, density, color=color, lw=3)
    handles.insert(0, mlines.Line2D([], [], color=color, lw=3, label=samples.label or "This work"))

    joint.set_xlim(xlim)
    joint.set_ylim(ylim)
    amplitude.set_xlim(xlim)
    slope.set_xlim(ylim)
    for ax in (amplitude, slope):
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
    amplitude.tick_params(labelbottom=False)
    amplitude.set_title(f"{STAR_LABELS['Delta2_star'][:-1]} = {format_quantiles(x, w)}$",
                        fontsize=fontsize)
    slope.set_title(f"{STAR_LABELS['n_star'][:-1]} = {format_quantiles(y, w)}$", fontsize=fontsize)
    joint.set_xlabel(STAR_LABELS["Delta2_star"], fontsize=fontsize)
    joint.set_ylabel(STAR_LABELS["n_star"], fontsize=fontsize)
    slope.set_xlabel(STAR_LABELS["n_star"], fontsize=fontsize)
    for ax in (joint, amplitude, slope):
        ax.tick_params(labelsize=fontsize - 4)
    axes[0, 1].legend(handles=handles, loc="center", fontsize=fontsize - 4, frameon=False)
    return fig


def overlay_gaussian(fig, mean, covariance, nsigma=1, color="C3", label="MLE Gaussian",
                     error_label=None, posterior_label=None):
    """Overlay a Gaussian (e.g. a minimizer's Gauss--Newton errors) on :func:`plot_star_posterior`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Output of :func:`plot_star_posterior`.
    mean : mapping or array_like
        ``{"Delta2_star": ..., "n_star": ...}`` or the pair of values.
    covariance : array_like, shape (2, 2)
        Covariance of (Delta2_star, n_star), same coordinates as the samples.
    nsigma : float, default=1
        Ellipse in the 1D-interval convention (Delta chi2 = nsigma**2), see
        :func:`cup1d.postprocessing.primitives.gaussian_ellipse`.
    color : str, default="C3"
        Colour.
    label : str, default="MLE Gaussian"
        Legend label of the centre.
    error_label : str, optional
        Legend label of the ellipse (default: ``"<nsigma>$\\sigma$"``).
    posterior_label : str, optional
        Legend entry for the posterior contours (colour C0).

    Returns
    -------
    matplotlib.figure.Figure
        The same figure. With zero errors (e.g. fixed cosmology) only the
        centre is drawn: a marker and vertical lines in the marginals.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from cup1d.postprocessing.primitives import gaussian_ellipse

    if isinstance(mean, dict):
        mean = [mean[name] for name in STAR_PARAMETERS]
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0):
        raise ValueError("covariance must be a finite 2x2 matrix with non-negative variances")
    errors = np.sqrt(np.diag(covariance))
    has_errors = bool(np.all(errors > 0))
    axes = np.asarray(fig.axes).reshape(2, 2)
    joint = axes[1, 0]
    if has_errors:
        gaussian_ellipse(joint, mean, covariance, nsigma=nsigma, facecolor=color, edgecolor=color,
                         alpha=0.3)
    joint.plot(*mean, "X", color=color)
    for index, ax in enumerate((axes[0, 0], axes[1, 1])):
        if has_errors:
            grid = np.linspace(*ax.get_xlim(), 300)
            ax.plot(grid, np.exp(-0.5 * ((grid - mean[index]) / errors[index]) ** 2), color=color)
        else:
            ax.axvline(mean[index], color=color)
    handles = [Line2D([], [], marker="X", color=color, linestyle="", label=label)]
    if has_errors:
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.3,
                             label=error_label or f"{nsigma:g}$\\sigma$"))
    legend_axis = axes[0, 1]
    legend_axis.set_visible(True)
    legend_axis.axis("off")
    if posterior_label is not None:
        handles.insert(0, Line2D([], [], color="C0", label=posterior_label))
    legend_axis.legend(handles=handles, loc="center")
    return fig

"""Drawing primitives shared by the cup1d figures.

Each primitive draws on axes it is given (or on a new figure it returns) and
has no other side effect: no ``plt.show``, no printing, no saving.

Functions
---------
contour_delta_chi2, ellipse_offsets, gaussian_ellipse
    Gaussian error ellipses, with the contour chosen explicitly either as a
    1D interval (``nsigma``: Delta chi2 = nsigma**2) or as an enclosed 2D
    probability (``level``: Delta chi2 = chi2.ppf(level, 2)).
draw_contour_polygons
    Saved 68/95 per cent contour polygons (``line_sigmas.npy``).
draw_band
    Median line and quantile bands of samples against a coordinate.
draw_correlation_matrix
    Lower-triangle correlation (or any) matrix with labels.
plot_traces
    Walker/chain traces (log posterior or parameters), stuck ones highlighted.
"""

import matplotlib.pyplot as plt
import numpy as np


# ----- Gaussian ellipses --------------------------------------------------------


def contour_delta_chi2(nsigma=None, level=None):
    """Delta chi2 of a 2D Gaussian contour.

    ``nsigma``: the contour whose projection onto each axis is the 1D
    ``nsigma`` interval (Delta chi2 = nsigma**2; 39%/86% of the 2D probability
    for 1/2 sigma). ``level``: the contour enclosing that 2D probability
    (e.g. 0.6827 -> Delta chi2 = 2.30).
    """
    if (nsigma is None) == (level is None):
        raise ValueError("give exactly one of nsigma and level")
    if nsigma is not None:
        return float(nsigma) ** 2
    from scipy.stats import chi2

    return float(chi2.ppf(level, 2))


def ellipse_offsets(covariance, delta_chi2, n_points=200):
    """Outline of a 2D Gaussian contour around the origin, shape ``(2, n_points)``.

    Points ``x`` with ``x^T C^-1 x = delta_chi2`` for the 2x2 covariance ``C``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    if np.any(eigenvalues < -1e-12 * max(1.0, np.max(np.abs(eigenvalues)))):
        raise ValueError("covariance is not positive semidefinite")
    angle = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(angle), np.sin(angle)])
    return eigenvectors @ (np.sqrt(delta_chi2 * np.maximum(eigenvalues, 0))[:, None] * circle)


def gaussian_ellipse(ax, mean, cov, nsigma=None, level=None, n_points=200, **kwargs):
    """Draw a Gaussian error ellipse.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    mean : array_like, shape (2,)
        Centre.
    cov : array_like, shape (2, 2)
        Covariance of the two plotted coordinates.
    nsigma, level : float
        Exactly one: the 1D interval convention (``nsigma``) or the enclosed
        2D probability (``level``), see :func:`contour_delta_chi2`.
    n_points : int, default=200
        Points of the outline.
    **kwargs
        :class:`matplotlib.patches.Polygon` options (``facecolor``,
        ``edgecolor``, ``alpha``, ``lw``, ``label``, ``fill``...).

    Returns
    -------
    matplotlib.patches.Polygon
        The added patch.
    """
    from matplotlib.patches import Polygon

    outline = ellipse_offsets(cov, contour_delta_chi2(nsigma=nsigma, level=level), n_points)
    vertices = (np.asarray(mean, dtype=float)[:, None] + outline).T
    patch = Polygon(vertices, closed=True, **kwargs)
    ax.add_patch(patch)
    return patch


# ----- contours, bands, matrices --------------------------------------------------


def draw_contour_polygons(ax, contours, offset=(0.0, 0.0), scale=(1.0, 1.0), cmap="Blues",
                          label=None, levels=(0.68, 0.95), line_alpha=0.75, fill_alpha=0.5,
                          ls="-", hatch=None):
    """Draw saved credible-region polygons (``{level: [(x, y), ...]}``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    contours : mapping
        Level to a list of ``(x, y)`` outlines, as written by
        :func:`cup1d.postprocessing.chains.products.save_contours`.
    offset, scale : pair of float
        Plotted coordinates are ``(x - offset[0]) * scale[0]`` (same for y),
        e.g. to remove blinding offsets or plot ``A_s`` in units of 1e-9.
    cmap : str, default="Blues"
        Colour map; the inner level takes the darker colour.
    label : str, optional
        Legend label (first outline only).
    levels : sequence of float, default=(0.68, 0.95)
        Levels drawn, inner first.
    line_alpha, fill_alpha : float
        Opacity of outlines and fills.
    ls : str, default="-"
        Outline line style.
    hatch : str, optional
        Fill hatch.

    Returns
    -------
    tuple of list
        ``(x, y)`` arrays of everything drawn (for axis limits).
    """
    colormap = plt.colormaps[cmap]
    xs, ys = [], []
    for index, level in enumerate(levels):
        color = colormap((0.7, 0.3)[min(index, 1)])
        for x, y in contours[level]:
            x = (np.asarray(x) - offset[0]) * scale[0]
            y = (np.asarray(y) - offset[1]) * scale[1]
            ax.plot(x, y, color=color, lw=(3, 2)[min(index, 1)], ls=ls, alpha=line_alpha, label=label)
            ax.fill(x, y, color=color, alpha=fill_alpha, hatch=hatch)
            label = None
            xs.append(x)
            ys.append(y)
    return xs, ys


def draw_band(ax, x, samples=None, quantiles=None, weights=None, color="C0",
              levels=((0.16, 0.84), (0.025, 0.975)), alphas=(0.5, 0.2), median=False,
              label=None, **line_kwargs):
    """Draw quantile bands of samples (or given quantiles) against ``x``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : array_like, shape (nx,)
        Abscissa.
    samples : array_like, shape (n, nx), optional
        Samples at every ``x``.
    quantiles : mapping, optional
        Instead of samples: ``{(low, high): (y_low, y_high)}`` and optionally
        ``{"median": y}``.
    weights : array_like, shape (n,), optional
        Sample weights.
    color : str, default="C0"
        Band colour.
    levels : sequence of (float, float)
        Quantile pairs, innermost first.
    alphas : sequence of float
        Opacity of every band.
    median : bool, default=False
        Also draw the median line.
    label : str, optional
        Legend label of the innermost band.
    **line_kwargs
        Options of the median line.

    Returns
    -------
    dict
        The plotted quantiles: ``{(low, high): (y_low, y_high), "median": y}``.
    """
    from cup1d.postprocessing.posterior.stats import weighted_quantile

    x = np.asarray(x)
    if quantiles is None:
        samples = np.asarray(samples, dtype=float)
        probs = sorted({p for pair in levels for p in pair} | {0.5})
        q = np.array([weighted_quantile(samples[:, i], probs, weights) for i in range(samples.shape[1])]).T
        table = dict(zip(probs, q))
        quantiles = {pair: (table[pair[0]], table[pair[1]]) for pair in levels}
        quantiles["median"] = table[0.5]
    for index, (pair, alpha) in enumerate(zip(levels, alphas)):
        low, high = quantiles[pair]
        ax.fill_between(x, low, high, color=color, alpha=alpha, lw=0,
                        label=label if index == 0 else None)
    if median and "median" in quantiles:
        ax.plot(x, quantiles["median"], color=color, **line_kwargs)
    return quantiles


def correlation_from_covariance(covariance):
    """Correlation matrix of a covariance (zero where the covariance is zero)."""
    covariance = np.asarray(covariance, dtype=float)
    sigma = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(sigma, sigma)
    correlation[covariance == 0] = 0
    return correlation


def draw_correlation_matrix(ax, matrix, labels=None, lower=True, threshold=None, cmap="bwr",
                            vmin=-1, vmax=1, fontsize=12, colorbar="inset", separators=None,
                            colorbar_label="Correlation"):
    """Draw a (correlation) matrix.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    matrix : array_like, shape (n, n)
        Matrix to show.
    labels : sequence of str, optional
        Tick labels (no ticks without them).
    lower : bool, default=True
        Mask the upper triangle.
    threshold : float, optional
        Write the coefficients with larger absolute value in their cells.
    cmap, vmin, vmax
        Colour scale.
    fontsize : float, default=12
        Base font size.
    colorbar : {"inset", "side", None}, default="inset"
        Horizontal colour bar inside the empty upper triangle, a vertical one
        on the side, or none.
    separators : sequence of int, optional
        Indices where a new group starts (thin grey lines).
    colorbar_label : str, default="Correlation"
        Label of the colour bar.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    shown = np.ma.masked_where(np.triu(np.ones((n, n), dtype=bool), k=1), matrix) if lower else matrix
    image = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    if labels is not None:
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=fontsize - 2)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=fontsize - 2)
    if threshold is not None:
        for i in range(n):
            for j in range(i + 1 if lower else n):
                if i != j and abs(matrix[i, j]) > threshold:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=max(fontsize - 4, 6))
    for i in separators or ():
        ax.axhline(i - 0.5, color="0.6", lw=0.5)
        ax.axvline(i - 0.5, color="0.6", lw=0.5)
    if colorbar == "inset":
        cax = ax.inset_axes([0.55, 0.9, 0.42, 0.03])
        bar = ax.figure.colorbar(image, cax=cax, orientation="horizontal")
    elif colorbar == "side":
        bar = ax.figure.colorbar(image, ax=ax)
    else:
        bar = None
    if bar is not None:
        bar.set_label(colorbar_label, fontsize=fontsize)
        bar.ax.tick_params(labelsize=fontsize - 2)
    return image


# ----- chain traces -----------------------------------------------------------------


def plot_traces(values, stuck=None, names=None, ax=None, ylabel=None, xlabel="step",
                ylim=None, show_mean=False, max_points=2000, fontsize=12):
    """Plot chain or walker traces, highlighting stuck ones.

    Parameters
    ----------
    values : array_like or sequence of array_like
        A 2D array of shape ``(n_steps, n_chains)`` (e.g. emcee ``lnprob``) or
        one 1D trace per chain (chains of different lengths, e.g. Cobaya).
    stuck : collection, optional
        Indices (or names, with ``names``) of chains drawn in colour and
        labelled; the others are thin grey lines. Without ``stuck`` every
        chain is drawn in colour.
    names : sequence of str, optional
        Chain names (legend labels of the stuck ones).
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created otherwise.
    ylabel, xlabel : str
        Axis labels.
    ylim : pair of float, optional
        Y limits.
    show_mean : bool, default=False
        Also draw the mean and median over chains (equal lengths only).
    max_points : int, default=2000
        Traces longer than this are thinned for drawing.
    fontsize : float, default=12
        Font size of labels and legend.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    if isinstance(values, np.ndarray) and values.ndim == 2:
        traces = [values[:, i] for i in range(values.shape[1])]
    else:
        traces = [np.asarray(v) for v in values]
    names = list(names) if names is not None else [str(i) for i in range(len(traces))]
    stuck = None if stuck is None else set(stuck)
    any_stuck = False
    for i, (name, trace) in enumerate(zip(names, traces)):
        step = max(1, len(trace) // max_points)
        x = np.arange(len(trace))[::step]
        if stuck is None:
            ax.plot(x, trace[::step], lw=0.6, alpha=0.8)
        elif i in stuck or name in stuck:
            any_stuck = True
            ax.plot(x, trace[::step], lw=0.6, alpha=0.9, label=name)
        else:
            ax.plot(x, trace[::step], lw=0.6, alpha=0.25, color="0.4")
    if show_mean:
        lengths = {len(t) for t in traces}
        if len(lengths) == 1:
            stacked = np.column_stack(traces)
            ax.plot(np.mean(stacked, axis=1), lw=3, color="k", label="mean")
            ax.plot(np.median(stacked, axis=1), lw=3, color="C3", label="median")
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if any_stuck or show_mean:
        ax.legend(fontsize=fontsize - 5, ncol=4, frameon=False)
    return fig

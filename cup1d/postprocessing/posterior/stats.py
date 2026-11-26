"""Weighted statistics shared by the posterior figures.

MCMC samplers such as Cobaya write weighted samples (the weight of a row is the
number of steps the chain stayed there), so every summary here accepts
weights. Unweighted samples (emcee) use unit weights.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def weighted_quantile(values, quantiles, weights=None):
    """Return weighted quantiles of one-dimensional samples.

    Parameters
    ----------
    values : array_like, shape (n,)
        Samples.
    quantiles : float or array_like
        Quantiles in ``[0, 1]``.
    weights : array_like, shape (n,), optional
        Non-negative sample weights; unit weights by default.

    Returns
    -------
    float or numpy.ndarray
        Quantiles, linearly interpolated in the weighted empirical
        distribution (midpoint convention).
    """
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values)


def weighted_correlation(values, weights=None):
    """Return the weighted Pearson correlation matrix of sample columns.

    Parameters
    ----------
    values : array_like, shape (n, p)
        Samples, one parameter per column.
    weights : array_like, shape (n,), optional
        Non-negative sample weights.

    Returns
    -------
    numpy.ndarray, shape (p, p)
        Correlation matrix.
    """
    covariance = np.cov(np.asarray(values, dtype=float), rowvar=False, aweights=weights)
    sigma = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigma, sigma)


def density_contours(x, y, weights=None, levels=(0.68, 0.95), bins=40, smooth=1.0,
                     ranges=None):
    """Estimate highest-density contours of two-dimensional samples.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Paired samples.
    weights : array_like, shape (n,), optional
        Sample weights.
    levels : sequence of float, default=(0.68, 0.95)
        Enclosed probability of each contour.
    bins : int, default=40
        Histogram bins per axis.
    smooth : float, default=1.0
        Gaussian smoothing of the histogram, in bins (0 disables it).
    ranges : sequence of two (min, max) pairs, optional
        Histogram ranges; by default the 0.05--99.95 weighted percentiles,
        padded by 10 per cent.

    Returns
    -------
    X, Y : numpy.ndarray, shape (bins, bins)
        Bin-centre grid for :func:`matplotlib.pyplot.contour`.
    H : numpy.ndarray, shape (bins, bins)
        Smoothed density on the grid (rows follow ``y``).
    thresholds : numpy.ndarray
        Increasing density thresholds, from the widest to the narrowest level,
        followed by a value above the maximum (ready for ``contourf``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if ranges is None:
        ranges = [padded_range(v, weights) for v in (x, y)]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=ranges, weights=weights)
    if smooth:
        H = gaussian_filter(H, smooth)
    flat = np.sort(H.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    thresholds = [flat[min(np.searchsorted(cumulative, level), flat.size - 1)]
                  for level in sorted(levels, reverse=True)]
    thresholds = np.append(np.maximum.accumulate(thresholds), H.max() * (1 + 1e-6))
    # contourf requires strictly increasing levels
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] * (1 + 1e-9) + 1e-300
    X, Y = np.meshgrid(0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1]))
    return X, Y, H.T, thresholds


def padded_range(values, weights=None, low=0.0005, high=0.9995, pad=0.1):
    """Return a weighted-percentile range of samples with relative padding.

    Parameters
    ----------
    values : array_like
        Samples.
    weights : array_like, optional
        Sample weights.
    low, high : float
        Quantiles bounding the range.
    pad : float, default=0.1
        Padding as a fraction of the range on each side.

    Returns
    -------
    tuple of float
        ``(min, max)``.
    """
    a, b = weighted_quantile(values, [low, high], weights)
    if b <= a:
        a, b = a - 0.5 * abs(a) - 1e-12, b + 0.5 * abs(b) + 1e-12
    width = b - a
    return a - pad * width, b + pad * width


def marginal_density(values, weights=None, bins=60, smooth=1.0, value_range=None):
    """Return a smoothed one-dimensional marginal normalized to peak one.

    Parameters
    ----------
    values : array_like
        Samples.
    weights : array_like, optional
        Sample weights.
    bins : int, default=60
        Histogram bins.
    smooth : float, default=1.0
        Gaussian smoothing in bins.
    value_range : tuple of float, optional
        Histogram range; by default :func:`padded_range`.

    Returns
    -------
    centres, density : numpy.ndarray
        Bin centres and densities with maximum one.
    """
    if value_range is None:
        value_range = padded_range(values, weights)
    density, edges = np.histogram(values, bins=bins, range=value_range, weights=weights)
    density = density.astype(float)
    if smooth:
        density = gaussian_filter1d(density, smooth)
    return 0.5 * (edges[1:] + edges[:-1]), density / density.max()


def format_quantiles(values, weights=None, quantiles=(0.16, 0.5, 0.84)):
    """Format the median and the 16--84 per cent interval as LaTeX.

    Parameters
    ----------
    values : array_like
        Samples.
    weights : array_like, optional
        Sample weights.
    quantiles : sequence of three float, default=(0.16, 0.5, 0.84)
        Lower, central and upper quantiles.

    Returns
    -------
    str
        ``median^{+upper}_{-lower}`` with two significant digits in the
        smaller error (no surrounding ``$``).
    """
    low, mid, high = weighted_quantile(values, quantiles, weights)
    lower, upper = mid - low, high - mid
    error = min(lower, upper)
    decimals = 2 if error <= 0 else max(1 - int(np.floor(np.log10(error))), 0)
    return f"{mid:.{decimals}f}^{{+{upper:.{decimals}f}}}_{{-{lower:.{decimals}f}}}"

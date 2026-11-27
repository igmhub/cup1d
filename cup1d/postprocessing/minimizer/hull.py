"""Emulator inputs of a fit against the emulator training hull."""

import numpy as np

from cup1d.postprocessing.style import styled


def hull_points(like, point):
    """Emulator inputs of every redshift at one physical point, shape ``(nz, n_params)``.

    Parameters
    ----------
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood whose theory has a training hull (``theory.hull``).
    point : mapping
        Physical value of every free parameter.

    Returns
    -------
    numpy.ndarray
        Columns in the order of ``like.theory.hull.params``, one row per
        redshift of the first data set.
    """
    result = like.get_p1d_kms(point, return_emu_params=True, apply_hull=False)
    if result is None:
        raise ValueError("the point is outside the star priors")
    key = next(iter(like.data))
    emu_params = result[1][key][-1]
    return np.column_stack([np.asarray(emu_params[name], dtype=float)
                            for name in like.theory.hull.params])


def _projection_outlines(hull):
    """2D outlines of the hull, ``{(i, j): (x, y) segments}`` with x the j-th and y the i-th input."""
    from scipy.spatial import ConvexHull

    outlines = {}
    if getattr(hull, "hulls", None) is not None:
        for projection in hull.hulls:
            dims = (projection.dim0, projection.dim1)
            outlines[dims] = [(projection.points[s, 1], projection.points[s, 0])
                              for s in projection.simplices]
    elif getattr(hull, "hull", None) is not None:
        n = hull.hull.points.shape[1]
        vertices = hull.hull.points[hull.hull.vertices]
        for i in range(n):
            for j in range(i):
                projected = vertices[:, [j, i]]
                flat = ConvexHull(projected)
                outlines[(i, j)] = [(projected[s, 0], projected[s, 1]) for s in flat.simplices]
    return outlines


@styled
def plot_point_in_hull(hull, points, labels=None, fontsize=12):
    """Pairwise projections of points over the training hull of the emulator.

    Parameters
    ----------
    hull : cup1d.emulator.hull.Hull
        Training hull (2D projections ``hulls`` or the full ``hull``) with
        ordered input names ``params``.
    points : array_like, shape (n_points, n_params)
        Emulator inputs, e.g. :func:`hull_points` of a best fit (one row per
        redshift).
    labels : sequence of str, optional
        Axis labels (default: ``hull.params``).
    fontsize : float, default=12
        Font size of the labels.

    Returns
    -------
    matplotlib.figure.Figure
        Lower triangle: points (coloured by row) and hull outlines; diagonal:
        histograms of every input.
    """
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=float)
    n = points.shape[1]
    labels = list(hull.params) if labels is None else list(labels)
    outlines = _projection_outlines(hull)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n + 2, 2.4 * n + 2), constrained_layout=True,
                             squeeze=False)
    colours = plt.cm.jet(np.linspace(0, 1, len(points)))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
            elif i == j:
                ax.hist(points[:, i])
            else:
                for key in ((i, j), (j, i)):
                    for x, y in outlines.get(key, []):
                        if key == (i, j):
                            ax.plot(x, y, "k-", lw=1)
                        else:
                            ax.plot(y, x, "k-", lw=1)
                ax.scatter(points[:, j], points[:, i], s=15, c=colours, zorder=3)
    for j in range(n):
        axes[-1, j].set_xlabel(labels[j], fontsize=fontsize)
        axes[j, 0].set_ylabel(labels[j], fontsize=fontsize)
    return fig

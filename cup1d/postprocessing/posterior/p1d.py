"""Best-fitting P1D model against every data set, one data set per row."""

import matplotlib.pyplot as plt
import numpy as np

from cup1d.postprocessing.p1d import (
    P1DBin, _dataset_titles, _finish, plot_p1d_residuals, plot_p1d_spectra,
)


def stacked_bins(entries):
    """Colour and offset P1D bins for one-axis-per-data-set figures.

    Parameters
    ----------
    entries : sequence of P1DBin
        Bins of every data set (any colour and shift). Data sets keep the
        order of their first bin; redshifts are sorted within a data set.

    Returns
    -------
    list of P1DBin
        Copies coloured with the ``jet`` colour map by redshift index and
        shifted by ``4 i / (nz - 1)`` in the residual figure, as
        :class:`cup1d.postprocessing.p1d.P1DPlotter` does for one likelihood.
    """
    keys = list(dict.fromkeys(entry.key for entry in entries))
    out = []
    for key in keys:
        own = sorted((e for e in entries if e.key == key), key=lambda e: e.z)
        nz = len(own)
        for i, e in enumerate(own):
            color = plt.cm.jet(i / (nz - 1)) if nz > 1 else "C0"
            shift = 4 * i / (nz - 1) if nz > 1 else 0.0
            out.append(P1DBin(e.key, e.index, e.z, e.k, e.data, e.model, e.error, e.chi2, e.dof,
                              color, shift, e.model_error, e.realizations))
    return out


def plot_best_fit(entries, chi2, dof, ndata=None, residuals=True, title=None, fontsize=18,
                  plot_fname=None):
    """Plot the best-fitting model against the data, one row per data set.

    Parameters
    ----------
    entries : sequence of P1DBin
        One bin per data set and redshift, in native velocity units, e.g.
        DESI DR1 first and the high-resolution data second.
    chi2 : float
        Correlated total chi-squared of the fit (figure title).
    dof : int
        Degrees of freedom.
    ndata : int, optional
        Total number of data points.
    residuals : bool, default=True
        Plot data/model offset by redshift (as in the DR1 paper); otherwise
        ``k P1D / pi`` with the model as dashed lines.
    title : str, optional
        Line prepended to the chi-squared title.
    fontsize : float, default=18
        Base font size.
    plot_fname : str or path-like, optional
        File root for PDF and PNG output; the figure is returned either way.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Each data-set axis is titled with the sum of its per-redshift chi-squared
    (marginal blocks), which need not add up to the correlated total.
    """
    bins = stacked_bins(entries)
    renderer = plot_p1d_residuals if residuals else plot_p1d_spectra
    fig, axes = renderer(bins, panels=False, fontsize=fontsize, print_chi2=True)
    stats = {}
    for item in bins:
        entry = stats.setdefault(item.key, [0.0, 0])
        entry[0] += float(item.chi2)
        entry[1] += int(np.count_nonzero(item.data))
    _dataset_titles(axes, [b.key for b in bins], stats, fontsize)
    _finish(fig, axes, chi2, dof, fontsize=fontsize, ylims=None, plot_fname=plot_fname,
            show=False, ndata=ndata)
    if title:
        fig.suptitle(title + "\n" + fig._suptitle.get_text(), fontsize=fontsize)
    return fig

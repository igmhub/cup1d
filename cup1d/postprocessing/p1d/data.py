"""The P1D measurement alone (``BaseDataP1D.plot_p1d``)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from cup1d.postprocessing.labels import K_KMS_LABEL
from cup1d.postprocessing.style import styled


@styled
def plot_p1d_data(
    zs,
    k_kms,
    Pk_kms,
    cov_Pk_kms,
    use_dimensionless=True,
    xlog=False,
    ylog=True,
    fname=None,
    cov_ext=None,
    ftsize=18,
    store_data=False,
):
    """Plot P1D measurements and covariance-diagonal uncertainties.

    Parameters
    ----------
    zs : array_like of float
        Redshifts associated with the P1D blocks.
    k_kms, Pk_kms : sequence of ndarray
        Per-redshift wavenumber grids in ``s / km`` and P1D vectors in
        ``km / s``.
    cov_Pk_kms : sequence of ndarray
        Per-redshift covariance blocks with shape ``(nk, nk)`` in
        ``(km / s)**2``.
    use_dimensionless : bool, default: True
        Plot ``k P1D / pi`` rather than P1D itself.
    xlog, ylog : bool, default: False, True
        Use logarithmic wavenumber and ordinate axes.
    fname : str or path-like, optional
        File root for PDF and PNG outputs.
    cov_ext : sequence of ndarray, optional
        Optional covariance blocks whose diagonal errors replace those from
        ``cov_Pk_kms``.
    ftsize : float, default: 18
        Base font size.
    store_data : bool, default: False
        Return plotted x, y, and diagonal-error arrays.

    Returns
    -------
    matplotlib.figure.Figure or dict
        The figure, or the keys ``xN``, ``yN``, and ``errN`` for every
        redshift block when ``store_data`` is true.
    """
    if store_data:
        out_data = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    N = len(zs)
    for ii in range(N):
        if cov_ext is None:
            err_Pk_kms = np.sqrt(np.diagonal(cov_Pk_kms[ii]))
        else:
            err_Pk_kms = np.sqrt(np.diagonal(cov_ext[ii]))
        if use_dimensionless:
            fact = k_kms[ii] / np.pi
        else:
            fact = 1.0

        if store_data:
            out_data["x" + str(ii)] = k_kms[ii]
            out_data["y" + str(ii)] = fact * Pk_kms[ii]
            out_data["err" + str(ii)] = fact * err_Pk_kms

        ax.errorbar(
            k_kms[ii],
            fact * Pk_kms[ii],
            yerr=fact * err_Pk_kms,
            label=r"$z = {}$".format(np.round(zs[ii], 3)),
            color=colormaps["tab20"].colors[ii],
        )

    ax.legend(ncol=4, fontsize=ftsize - 4)
    if ylog:
        ax.set_yscale("log", nonpositive="clip")
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(K_KMS_LABEL, fontsize=ftsize)
    if use_dimensionless:
        ax.set_ylabel(r"$\mathrm{\pi}^{-1}k_\parallel\,P(k)$", fontsize=ftsize)
    else:
        ax.set_ylabel(r"$P(k)\,[\mathrm{km\,s^{-1}}]$", fontsize=ftsize)

    ax.tick_params(axis="both", which="major", labelsize=ftsize)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(str(fname) + ".pdf")
        fig.savefig(str(fname) + ".png")

    if store_data:
        return out_data
    return fig

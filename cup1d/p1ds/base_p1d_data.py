"""Shared container for observed 1D power spectrum measurements.

This module provides the base class used by observational and mock P1D loaders.
It stores per-redshift wavenumbers, power spectra, covariance matrices, and
optional flattened arrays for analyses with cross-redshift covariance.

"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import numpy.typing as npt

from cup1d.utils.utils import get_path_repo

# Type aliases
Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]
Float = Union[float, int]


def _drop_zbins(
    z_in: Array1D,
    k_in: list[Array1D],
    Pk_in: list[Array1D],
    cov_in: list[Array2D],
    z_min: float,
    z_max: float,
    full_zs: Array1D | None = None,
    full_Pk_kms: Array1D | None = None,
    full_cov_kms: Array2D | None = None,
    full_cov_stat_kms: Array2D | None = None,
    Pksmooth_kms: list[Array1D] | None = None,
    cov_stat: list[Array2D] | None = None,
    kmin_in: list[Array1D] | None = None,
    kmax_in: list[Array1D] | None = None,
) -> tuple:
    """Drop redshift bins outside ``[z_min, z_max]`` and trim empty k bins.

    Parameters
    ----------
    z_in : Array1D
        Input redshift values.
    k_in : List[Array1D]
        Input wavenumber values.
    Pk_in : List[Array1D]
        Input power spectrum values.
    cov_in : List[Array2D]
        Input covariance matrices.
    z_min : float
        Minimum redshift.
    z_max : float
        Maximum redshift.
    full_zs : Optional[Array1D], optional
        Full redshift array.
    full_Pk_kms : Optional[Array1D], optional
        Full power spectrum.
    full_cov_kms : Optional[Array2D], optional
        Full covariance.
    full_cov_stat_kms : Optional[Array2D], optional
        Full statistical covariance.
    Pksmooth_kms : Optional[List[Array1D]], optional
        Smooth power spectrum.
    cov_stat : Optional[List[Array2D]], optional
        Statistical covariance.
    kmin_in : Optional[List[Array1D]], optional
        Minimum k values.
    kmax_in : Optional[List[Array1D]], optional
        Maximum k values.

    Returns
    -------
    tuple
        Processed per-redshift arrays and optional flattened full-covariance
        arrays, in the order consumed by :class:`BaseDataP1D`.
    """

    # k_in center of the kbin
    # kmin_in starting of the kbin
    # kmax_in ending of the kbin

    z_in = np.array(z_in)
    ind = np.argwhere((z_in >= z_min) & (z_in <= z_max))[:, 0]
    z_out = z_in[ind]

    k_out = []
    Pk_out = []
    cov_out = []
    cov_stat_out = []
    kmin_out = []
    kmax_out = []
    if Pksmooth_kms is None:
        Pksmooth_out = None
    else:
        Pksmooth_out = []
    if Pksmooth_kms is None:
        Pksmooth_out = None
    for jj in ind:
        # remove tailing zeros
        ind2 = np.argwhere(Pk_in[jj] != 0)[:, 0]
        k_out.append(k_in[jj][ind2])
        if kmin_in is not None:
            kmin_out.append(kmin_in[jj][ind2])
        else:
            kdiff = 0.5 * (k_in[jj][ind2][1] - k_in[jj][ind2][0])
            kmin_out.append(k_in[jj][ind2] - kdiff)
        if kmax_in is not None:
            kmax_out.append(kmax_in[jj][ind2])
        else:
            kdiff = 0.5 * (k_in[jj][ind2][1] - k_in[jj][ind2][0])
            kmax_out.append(k_in[jj][ind2] + kdiff)
        Pk_out.append(Pk_in[jj][ind2])
        if Pksmooth_kms is not None:
            Pksmooth_out.append(Pksmooth_kms[jj][ind2])
        cov_out.append(cov_in[jj][ind2, :][:, ind2])
        if cov_stat is not None:
            cov_stat_out.append(cov_stat[jj][ind2, :][:, ind2])

    if full_zs is not None:
        ind = np.argwhere((full_zs >= z_min) & (full_zs <= z_max))[:, 0]
        full_zs = full_zs[ind]
        full_Pk_kms = full_Pk_kms[ind]
        full_cov_kms = full_cov_kms[ind, :][:, ind]
        full_cov_stat_kms = full_cov_stat_kms[ind, :][:, ind]

    return (
        z_out,
        k_out,
        Pk_out,
        cov_out,
        full_zs,
        full_Pk_kms,
        full_cov_kms,
        full_cov_stat_kms,
        Pksmooth_out,
        cov_stat_out,
        kmin_out,
        kmax_out,
    )


class BaseDataP1D:
    """Base class to store measurements of the 1D power spectrum.

    Parameters
    ----------
    z : Array1D
        Redshift values.
    _k_kms : Union[Array1D, List[Array1D]]
        Wavenumber values in km/s.
    Pk_kms : List[Array1D]
        Power spectrum values.
    cov_Pk_kms : List[Array2D]
        Covariance matrices.
    z_min : float, optional
        Minimum redshift.
    z_max : float, optional
        Maximum redshift.
    full_zs : Optional[Array1D], optional
        Full redshift array for combined analysis.
    full_Pk_kms : Optional[Array1D], optional
        Full power spectrum.
    full_cov_kms : Optional[Array2D], optional
        Full covariance matrix.
    full_cov_stat_kms : Optional[Array2D], optional
        Full statistical covariance.
    Pksmooth_kms : Optional[List[Array1D]], optional
        Smooth power spectrum.
    cov_stat : Optional[List[Array2D]], optional
        Statistical covariance.
    k_kms_min : Optional[List[Array1D]], optional
        Minimum k values.
    k_kms_max : Optional[List[Array1D]], optional
        Maximum k values.
    """

    BASEDIR = os.path.join(get_path_repo("cup1d"), "data", "p1d_measurements")

    def __init__(
        self,
        z: Array1D,
        _k_kms: Array1D | list[Array1D],
        Pk_kms: list[Array1D],
        cov_Pk_kms: list[Array2D],
        z_min: float = 0,
        z_max: float = 10,
        full_zs: Array1D | None = None,
        full_Pk_kms: Array1D | None = None,
        full_cov_kms: Array2D | None = None,
        full_cov_stat_kms: Array2D | None = None,
        Pksmooth_kms: list[Array1D] | None = None,
        cov_stat: list[Array2D] | None = None,
        k_kms_min: list[Array1D] | None = None,
        k_kms_max: list[Array1D] | None = None,
    ) -> None:

        ## if multiple z, ensure that k_kms for each redshift
        # more than one z, and k_kms is different for each z
        if (len(z) > 1) & (len(np.atleast_1d(_k_kms[0])) != 1):
            k_kms = []
            for iz in range(len(z)):
                k_kms.append(_k_kms[iz])
        # more than one z, and kms is the same for all z
        elif (len(z) > 1) & (len(np.atleast_1d(_k_kms[0])) == 1):
            k_kms = []
            for iz in range(len(z)):
                k_kms.append(_k_kms)
        # only one z
        else:
            k_kms = _k_kms

        # drop zbins below z_min and above z_max
        res = _drop_zbins(
            z,
            k_kms,
            Pk_kms,
            cov_Pk_kms,
            z_min,
            z_max,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            full_cov_stat_kms=full_cov_stat_kms,
            Pksmooth_kms=Pksmooth_kms,
            cov_stat=cov_stat,
            kmin_in=k_kms_min,
            kmax_in=k_kms_max,
        )

        (
            self.z,
            self.k_kms,
            self.Pk_kms,
            self.cov_Pk_kms,
            self.full_zs,
            self.full_Pk_kms,
            self.full_cov_Pk_kms,
            self.full_cov_stat_Pk_kms,
            self.Pksmooth_kms,
            self.covstat_Pk_kms,
            self.k_kms_min,
            self.k_kms_max,
        ) = res

        self.full_k_kms = np.concatenate(self.k_kms)

        # decide if applying blinding
        self.apply_blinding = False
        if hasattr(self, "blinding"):
            if self.blinding is not None:
                self.apply_blinding = True

    def get_Pk_iz(self, iz):
        """Return P1D in km/s units for redshift bin ``iz``."""

        return self.Pk_kms[iz]

    def get_cov_iz(self, iz):
        """Return the P1D covariance for redshift bin ``iz``."""

        return self.cov_Pk_kms[iz]

    def get_icov_iz(self, iz):
        """Return the inverse P1D covariance for redshift bin ``iz``."""

        return self.icov_Pk_kms[iz]

    def cull_data(self, kmin_kms=0, kmax_kms=10):
        """Remove bins with wavenumber outside ``[kmin_kms, kmax_kms]``."""

        if (kmin_kms is None) & (kmax_kms is None):
            return

        for iz in range(len(self.z)):
            ind = np.argwhere(
                (self.k_kms[iz] >= kmin_kms) & (self.k_kms[iz] <= kmax_kms)
            )[:, 0]
            sli = slice(ind[0], ind[-1] + 1)
            self.k_kms[iz] = self.k_kms[iz][sli]
            self.Pk_kms[iz] = self.Pk_kms[iz][sli]
            self.cov_Pk_kms[iz] = self.cov_Pk_kms[iz][sli, sli]
            self.icov_Pk_kms[iz] = self.icov_Pk_kms[iz][sli, sli]

    def plot_p1d(
        self,
        use_dimensionless=True,
        xlog=False,
        ylog=True,
        fname=None,
        cov_ext=None,
        ftsize=18,
        store_data=False,
    ):
        """Plot the P1D measurement.

        If ``use_dimensionless`` is true, the y-axis is ``k P(k) / pi``.
        When ``store_data`` is true, return the plotted arrays instead of only
        creating the figure.
        """

        import matplotlib.pyplot as plt
        from matplotlib import colormaps, rcParams

        rcParams["mathtext.fontset"] = "stix"
        rcParams["font.family"] = "STIXGeneral"

        if store_data:
            out_data = {}

        fig, ax = plt.subplots(figsize=(8, 6))

        N = len(self.z)
        for ii in range(N):
            k_kms = self.k_kms[ii]
            Pk_kms = self.get_Pk_iz(ii)
            if cov_ext is None:
                err_Pk_kms = np.sqrt(np.diagonal(self.get_cov_iz(ii)))
            else:
                err_Pk_kms = np.sqrt(np.diagonal(cov_ext[ii]))
            if use_dimensionless:
                fact = k_kms / np.pi
            else:
                fact = 1.0

            if store_data:
                out_data["x" + str(ii)] = k_kms
                out_data["y" + str(ii)] = fact * Pk_kms
                out_data["err" + str(ii)] = fact * err_Pk_kms

            ax.errorbar(
                k_kms,
                fact * Pk_kms,
                yerr=fact * err_Pk_kms,
                label=rf"$z = {np.round(self.z[ii], 3)}$",
                color=colormaps["tab20"].colors[ii],
            )

        ax.legend(ncol=4, fontsize=ftsize - 4)
        if ylog:
            plt.yscale("log", nonpositive="clip")
        if xlog:
            plt.xscale("log")
        plt.xlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
        if use_dimensionless:
            plt.ylabel(r"$\mathrm{\pi}^{-1}k_\parallel\,P(k)$", fontsize=ftsize)
        else:
            plt.ylabel(r"$P(k) [km/s]$", fontsize=ftsize)

        ax.tick_params(axis="both", which="major", labelsize=ftsize)
        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname + ".pdf")
            plt.savefig(fname + ".png")
        else:
            plt.show()

        if store_data:
            return out_data

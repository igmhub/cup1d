import os
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from cup1d.utils.utils import get_path_cup1d


def _drop_zbins(
    z_in,
    k_in,
    Pk_in,
    cov_in,
    z_min,
    z_max,
    full_z=None,
    full_Pk_kms=None,
    full_cov_kms=None,
):
    """Drop redshift bins below z_min or above z_max"""

    z_in = np.array(z_in)
    ind = np.argwhere((z_in >= z_min) & (z_in <= z_max))[:, 0]
    z_out = z_in[ind]

    k_out = []
    Pk_out = []
    cov_out = []
    for jj in ind:
        # remove tailing zeros
        ind = np.argwhere(Pk_in[jj] != 0)[:, 0]
        k_out.append(k_in[jj][ind])
        Pk_out.append(Pk_in[jj][ind])
        cov_out.append(cov_in[jj][ind, :][:, ind])

    if full_z is not None:
        ind = np.argwhere((full_z >= z_min) & (full_z <= z_max))[:, 0]
        full_z = full_z[ind]
        full_Pk_kms = full_Pk_kms[ind]
        full_cov_kms = full_cov_kms[ind, :][:, ind]

    return z_out, k_out, Pk_out, cov_out, full_Pk_kms, full_cov_kms


class BaseDataP1D(object):
    """Base class to store measurements of the 1D power spectrum"""

    BASEDIR = get_path_cup1d() + "/data/p1d_measurements/"

    def __init__(
        self,
        z,
        _k_kms,
        Pk_kms,
        cov_Pk_kms,
        z_min=0,
        z_max=10,
        full_z=None,
        full_Pk_kms=None,
        full_cov_kms=None,
    ):
        """Construct base P1D class, from measured power and covariance"""

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
            full_z=full_z,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
        )

        (
            self.z,
            self.k_kms,
            self.Pk_kms,
            self.cov_Pk_kms,
            self.full_Pk_kms,
            self.full_cov_kms,
        ) = res

    def get_Pk_iz(self, iz):
        """Return P1D in units of km/s for redshift bin iz"""

        return self.Pk_kms[iz]

    def get_cov_iz(self, iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz"""

        return self.cov_Pk_kms[iz]

    def get_icov_iz(self, iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz"""

        return self.icov_Pk_kms[iz]

    def cull_data(self, kmin_kms=0, kmax_kms=10):
        """Remove bins with wavenumber k < kmin_kms and k > kmin_kms"""

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
        self, use_dimensionless=True, xlog=False, ylog=True, fname=None
    ):
        """Plot P1D mesurement. If use_dimensionless, plot k*P(k)/pi."""

        N = len(self.z)
        for i in range(N):
            k_kms = self.k_kms[i]
            Pk_kms = self.get_Pk_iz(i)
            err_Pk_kms = np.sqrt(np.diagonal(self.get_cov_iz(i)))
            if use_dimensionless:
                fact = k_kms / np.pi
            else:
                fact = 1.0
            plt.errorbar(
                k_kms,
                fact * Pk_kms,
                yerr=fact * err_Pk_kms,
                label="z = {}".format(np.round(self.z[i], 3)),
            )

        plt.legend(ncol=4)
        if ylog:
            plt.yscale("log", nonpositive="clip")
        if xlog:
            plt.xscale("log")
        plt.xlabel("k [s/km]")
        if use_dimensionless:
            plt.ylabel(r"$k P(k)/ \pi$")
        else:
            plt.ylabel("P(k) [km/s]")

        if fname is not None:
            plt.savefig(fname)
        else:
            plt.show()

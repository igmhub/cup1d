import os
import numpy as np
import matplotlib.pyplot as plt

import cup1d
from lace.utils.smoothing_manager import apply_smoothing


class BaseDataP1D(object):
    """Base class to store measurements of the 1D power spectrum"""

    BASEDIR = cup1d.__path__[0] + "/data/p1d_measurements/"

    def __init__(self, z, k_kms, Pk_kms, cov_Pk_kms):
        """Construct base P1D class, from measured power and covariance"""

        self.z = z
        self.k_kms = k_kms
        self.Pk_kms = Pk_kms
        self.cov_Pk_kms = cov_Pk_kms
        self.icov_Pk_kms = []
        for ii in range(len(z)):
            self.icov_Pk_kms.append(np.linalg.inv(cov_Pk_kms[ii]))

    def get_Pk_iz(self, iz):
        """Return P1D in units of km/s for redshift bin iz"""

        return self.Pk_kms[iz]

    def get_cov_iz(self, iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz"""

        return self.cov_Pk_kms[iz]

    def get_icov_iz(self, iz):
        """Return covariance of P1D in units of (km/s)^2 for redshift bin iz"""

        return self.icov_Pk_kms[iz]

    def _cull_data(self, kmin_kms):
        """Remove bins with wavenumber k < kmin_kms"""

        if kmin_kms is None:
            return

        # figure out number of bins to cull
        Ncull = np.sum(self.k_kms < kmin_kms)

        # cull wavenumbers, power spectra, and covariances
        self.k_kms = self.k_kms[Ncull:]
        self.Pk_kms = self.Pk_kms[:, Ncull:]
        for i in range(len(self.z)):
            self.cov_Pk_kms[i] = self.cov_Pk_kms[i][Ncull:, Ncull:]

        return

    def set_smoothing_kms(self, emulator):
        """Smooth data"""

        list_data_Mpc = []
        for ii in range(len(self.z)):
            data = {}
            data["k_Mpc"] = self.k_kms * self.dkms_dMpc[ii]
            data["p1d_Mpc"] = self.Pk_kms[ii] * self.dkms_dMpc[ii]
            list_data_Mpc.append(data)

        apply_smoothing(emulator, list_data_Mpc)

        for ii in range(len(self.z)):
            self.Pk_kms[ii] = (
                list_data_Mpc[ii]["p1d_Mpc_smooth"] / self.dkms_dMpc[ii]
            )

    def set_smoothing_Mpc(self, emulator, list_data_Mpc):
        """Smooth data"""

        apply_smoothing(emulator, list_data_Mpc)
        for ii in range(len(list_data_Mpc)):
            if "p1d_Mpc_smooth" in list_data_Mpc[ii]:
                list_data_Mpc[ii]["p1d_Mpc"] = list_data_Mpc[ii][
                    "p1d_Mpc_smooth"
                ]

        return list_data_Mpc

    def plot_p1d(self, use_dimensionless=True, xlog=False, ylog=True):
        """Plot P1D mesurement. If use_dimensionless, plot k*P(k)/pi."""

        N = len(self.z)
        for i in range(N):
            k_kms = self.k_kms
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
                label="z = {}".format(self.z[i]),
            )

        plt.legend()
        if ylog:
            plt.yscale("log", nonpositive="clip")
        if xlog:
            plt.xscale("log")
        plt.xlabel("k [s/km]")
        if use_dimensionless:
            plt.ylabel(r"$k P(k)/ \pi$")
        else:
            plt.ylabel("P(k) [km/s]")
        plt.show()


def _drop_zbins(z_in, k_in, Pk_in, cov_in, zmin, zmax):
    """Drop redshift bins below zmin or above zmax"""

    # size of input arrays
    Nz_in = len(z_in)
    Nk = len(k_in)

    # figure out how many z to keep
    keep = np.ones(Nz_in, dtype=bool)
    if zmin:
        keep = np.logical_and(keep, z_in >= zmin)
    if zmax:
        keep = np.logical_and(keep, z_in <= zmax)
    Nz_out = np.sum(keep)

    # setup new arrays
    z_out = np.empty(Nz_out)
    Pk_out = np.empty((Nz_out, Nk))
    cov_out = []
    i = 0
    for j in range(Nz_in):
        if keep[j]:
            z_out[i] = z_in[j]
            Pk_out[i] = Pk_in[j]
            Pk_out[i] = Pk_in[j]
            cov_out.append(cov_in[j])
            i += 1
    return z_out, k_in, Pk_out, cov_out

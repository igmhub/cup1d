import numpy as np
import os

from cup1d.likelihood import CAMB_model
from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_Chabanier2019(BaseDataP1D):
    """Class containing P1D from Chabanier et al. (2019)."""

    def __init__(self, z_min=0, z_max=10, add_syst=True):
        """Read measured P1D from Chabanier et al. (2019)."""

        # folder storing P1D measurements
        datadir = BaseDataP1D.BASEDIR + "/Chabanier2019/"

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(datadir, add_syst)

        super().__init__(z, k, Pk, cov, z_min=z_min, z_max=z_max)

        return


def read_from_file(datadir, add_syst):
    """Reconstruct covariance matrix from files."""

    # start by reading Pk file
    p1d_file = datadir + "/Pk1D_data.dat"
    inz, ink, inPk, inPkstat, _, _ = np.loadtxt(p1d_file, unpack=True)

    # store unique values of redshift and wavenumber
    z = np.unique(inz)
    Nz = len(z)
    k_kms = np.unique(ink)
    Nk = len(k_kms)

    # re-shape matrices, and compute variance (statistics only for now)
    Pk_kms = np.reshape(inPk, [Nz, Nk])
    var_Pk_kms = np.reshape(inPkstat**2, [Nz, Nk])

    # if asked to, add systematic variance
    if add_syst:
        # read file with systematic uncertainties
        syst_file = datadir + "Pk1D_syst.dat"
        insyst = np.loadtxt(syst_file, unpack=True)
        # add in quadrature 8 different systematics
        syst_var = np.sum(insyst**2, axis=0)
        var_Pk_kms += np.reshape(syst_var, [Nz, Nk])

    # now read correlation matrices
    corr_file = datadir + "Pk1D_cor.dat"
    incorr = np.loadtxt(corr_file, unpack=True)
    # note strange order
    allcorr = np.reshape(incorr, [Nk, Nz, Nk])

    # compute covariance matrices with statistics and systematic errors
    cov_Pk_kms = []
    for i in range(Nz):
        corr = allcorr[:, i, :]
        sigma = np.sqrt(var_Pk_kms[i])
        zcov = np.multiply(corr, np.outer(sigma, sigma))
        cov_Pk_kms.append(zcov)

    return z, k_kms, Pk_kms, cov_Pk_kms

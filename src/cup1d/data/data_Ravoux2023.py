import numpy as np
import os

from cup1d.data.base_p1d_data import BaseDataP1D

class P1D_Ravoux2023(BaseDataP1D):
    """Class containing P1D from Ravoux et al. (2023)."""

    def __init__(self, zmin=None, zmax=None, velunits=True):
        """Read measured P1D from Ravoux et al. (2023)."""

        # folder storing P1D measurements
         datadir=BaseDataP1D.BASEDIR + "/Ravoux2023/"

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(datadir, velunits)

        # drop low-z or high-z bins
        if zmin or zmax:
            z, k, Pk, cov = base_p1d_data._drop_zbins(z, k, Pk, cov, zmin, zmax)

        super().__init__(self, z, k, Pk, cov)

        return


def read_from_file(datadir, velunits):
    """Reconstruct covariance matrix from files."""

    # start by reading Pk file
    if velunits:
        p1d_file = datadir + "/p1d_measurement_kms.txt"
    else:
        p1d_file = datadir + "/p1d_measurement.txt"

    inz, ink, inPk = np.loadtxt(
        p1d_file,
        unpack=True,
        usecols=range(
            3,
        ),
    )
    # store unique values of redshift and wavenumber
    z = np.unique(inz)
    Nz = len(z)

    mask = inz == z[0]
    k = ink[mask]
    Nk = len(k)

    # re-shape matrices, and compute variance (statistics only for now)
    if velunits:
        Pk = []
        for i in range(len(z)):
            mask = inz == z[i]
            Pk.append(inPk[mask][:Nk] * np.pi / k)
        Pk = np.array(Pk)
    else:
        Pk = np.reshape(inPk * np.pi / k, [Nz, Nk])

    # now read correlation matrices
    if velunits:
        cov_file = datadir + "/covariance_matrix_kms.txt"
    else:
        cov_file = datadir + "/covariance_matrix.txt"

    inzcov, ink1, _, incov = np.loadtxt(
        cov_file,
        unpack=True,
        usecols=range(
            4,
        ),
    )
    if velunits:
        cov_Pk = []
        for i in range(Nz):
            mask = inzcov == z[i]
            k1 = np.unique(ink1[mask])
            cov_Pk_z = []
            for j in range(Nk):
                mask_k = mask & (ink1 == k1[j])
                cov_Pk_z.append(incov[mask_k][:Nk])
            cov_Pk.append(cov_Pk_z)
        cov_Pk = np.array(cov_Pk)
    else:
        cov_Pk = np.reshape(
            incov,
            [Nz, Nk, Nk],
        )

    return z, k, Pk, cov_Pk

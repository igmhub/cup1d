import os

import pandas
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_Karacayli2022(BaseDataP1D):
    def __init__(self, diag_cov=True, kmax_kms=0.09, zmin=None, zmax=None):
        """Read measured P1D from file.
        - diag_cov: for now, use diagonal covariance
        - kmax_kms: limit to low-k where we trust emulator"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(diag_cov, kmax_kms)

        # drop low-z or high-z bins
        if zmin or zmax:
            z, k, Pk, cov = _drop_zbins(z, k, Pk, cov, zmin, zmax)

        super().__init__(z, k, Pk, cov)

        return


def read_from_file(diag_cov, kmax_kms):
    """Read file containing mock P1D"""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/Karacayli2022/"

    # start by reading the file with measured band power
    # z, k, P, e
    data = pandas.read_table(
        datadir + "final-conservative-p1d-karacayli_etal2021.txt",
        delimiter="|",
        skipinitialspace=True,
        usecols=[1, 2, 3, 4],
        names=["z", "k", "P", "e"],
        header=0,
    ).to_records(index=False)

    zbins = np.unique(data["z"])
    kbins = np.unique(data["k"])
    Nk = kbins.size
    Nz = zbins.size

    w = kbins < kmax_kms
    kbins = kbins[w]
    print("Nz = {} , Nk = {}".format(Nz, Nk))
    Pk = data["P"].reshape(Nz, Nk)[:, w]
    ek = data["e"].reshape(Nz, Nk)[:, w]

    # for now only use diagonal elements
    assert diag_cov, "implement code to read full covariance"
    # for now only use diagonal elements
    cov = [np.diag(_**2) for _ in ek]

    return zbins, kbins, Pk, cov

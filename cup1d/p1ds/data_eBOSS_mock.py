import os

import pandas
import numpy as np

from cup1d.p1ds.base_p1d_mock import BaseMockP1D
from cup1d.p1ds.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_eBOSS_mock(BaseMockP1D):
    def __init__(
        self,
        diag_cov=False,
        kmax_kms=None,
        zmin=None,
        zmax=None,
        input_sim="nyx_central",
        add_noise=False,
        seed=0,
    ):
        """Read measured P1D from file.
        - diag_cov: for now, use diagonal covariance
        - kmax_kms: limit to low-k where we trust emulator"""

        self.input_sim = input_sim

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(diag_cov, input_sim, kmax_kms=kmax_kms)

        # drop low-z or high-z bins
        if zmin or zmax:
            z, k, Pk, cov = _drop_zbins(z, k, Pk, cov, zmin, zmax)

        super().__init__(z, k, Pk, cov, add_noise=add_noise, seed=seed)

        return


def read_from_file(diag_cov, input_sim, kmax_kms=None, old_cov=False):
    """Read file containing mock P1D"""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/eBOSS_mock/"

    all_input_sim = ["nyx_central"]
    try:
        assert input_sim in all_input_sim
    except AssertionError:
        raise ValueError(
            "Mock from input_sim="
            + input_sim
            + " not included. Available options: ",
            all_input_sim,
        )
    else:
        if input_sim == "nyx_central":
            fname = datadir + "/pk_1d_Nyx_emu_fiducial_mock.out"

    result = np.loadtxt(fname, unpack=True)
    z, k, Pk, sPk, nPk, bPk, tPk = result
    # z == Redshift
    # k == Wavenumber (usually in s/km)
    # Pk == 1D flux power spectrum
    # sPk == standard deviation of flux power spectrum from
    #       observations (statistical error)
    # tPk == systematic error on flux power spectrum,
    #       can be added in quadrature to sPk to give total error
    # As far as things currently stand, neither sPk nor tPk is
    #       currently used, as they are superseded by the
    #       actual inverse covmat file
    # nPk == noise Pk, used to derive corrections from noise
    #       (we have a fiducial noise power spectrum that we add
    #       with a redshift-dependent amplitude parameter in order
    #       to marginalize). We can discuss at the workshop how sensible
    #       this currently is (can we add it as a speaking point in one
    #       of the sessions about nusiance treatments, i.e. 1.C?).
    # bPk == background of the Pk, not sure what this was used for.
    #       In the original C code sent to me by Christophe Yeche it
    #       was already not being used, but he should know more!

    fname = datadir + "/pk_1d_DR12_13bins_invCov.out"
    inv_covmat = np.loadtxt(fname)

    zbins = np.unique(z)
    kbins = np.unique(k)
    Nk = kbins.size
    Nz = zbins.size
    print("Nz = {} , Nk = {}".format(Nz, Nk))

    Pkbins = []
    covbins = []

    for iz in range(Nz):
        if kmax_kms is None:
            mask = np.argwhere(z == zbins[iz])[:, 0]
        else:
            mask = np.argwhere((z == zbins[iz]) & (k <= kmax_kms))[:, 0]

        Pkbins.append(Pk[mask])

        if old_cov:
            cov_shape = np.zeros((mask.shape[0], mask.shape[0]))
            cov_shape[:, :] = sPk[mask] ** 2
            # systematics
            cov_shape[:, :] += tPk[mask] ** 2
            # noise (very large)
            cov_shape[:, :] += nPk[mask] ** 2
            covbins.append(cov_shape)
        else:
            covbins.append(np.linalg.inv(inv_covmat[mask][:, mask]))
            # covbins.append(np.linalg.inv(inv_covmat)[mask][:, mask])

    return zbins, kbins, Pkbins, covbins

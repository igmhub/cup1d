import os

import pandas
import numpy as np

from cup1d.data.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_eBOSS_mock(BaseDataP1D):
    def __init__(
        self,
        diag_cov=False,
        kmax_kms=0.09,
        zmin=2.19,
        zmax=None,
    ):
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


def read_from_file(diag_cov, kmax_kms, old_cov=False):
    """Read file containing mock P1D"""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/eBOSS_mock/"
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
        _ = np.argwhere(z == zbins[iz])[:, 0]
        Pkbins.append(Pk[_])

        if old_cov:
            cov_shape = np.zeros((_.shape[0], _.shape[0]))
            cov_shape[:, :] = sPk[_] ** 2
            # systematics
            cov_shape[:, :] += tPk[_] ** 2
            # noise (very large)
            cov_shape[:, :] += nPk[_] ** 2
            cov_shape[:, :] = np.sqrt(cov_shape)
            covbins.append(cov_shape)
        else:
            covbins.append(np.linalg.inv(inv_covmat[_][:, _]))

    return zbins, kbins, Pkbins, covbins

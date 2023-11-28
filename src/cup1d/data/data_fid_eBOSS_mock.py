import os

import pandas
import numpy as np

from cup1d.data.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_eBOSS_mock(BaseDataP1D):
    def __init__(self, diag_cov=False, kmax_kms=0.09, zmin=, zmax=None):
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
    datadir = BaseDataP1D.BASEDIR + "/eBOSS_mock/"
    fname = datadir + "/pk_1d_Nyx_emu_fiducial_mock.out"

    result = np.loadtxt(fname,unpack=True)
    try:
        z,k,Pk,sPk,nPk,bPk,tPk = result
    except ValueError as ve:
        # Possibly different format? Check!
        if("not enough values to unpack" in str(ve)):
            z,k,Pk,sPk = result
            nPk,bPk,tPk = np.zeros_like(z),np.zeros_like(z),np.zeros_like(z)
        else:
            raise
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

    zbins = np.unique(data["z"])
    kbins = np.unique(data["kc"])
    Nk = kbins.size
    Nz = zbins.size

    print("Nz = {} , Nk = {}".format(Nz, Nk))
    Pk = []
    cov = []
    for iz in range(Nz):
        w = np.isclose(data["z"], zbins[iz])
        tmp_d = np.zeros(Nk)
        _nk = w.sum()
        tmp_d[:_nk] = data["p_final"][w]
        Pk.append(tmp_d)

        tmp_cov = np.zeros((Nk, Nk))
        # Fill non-existing k bins with inf
        np.fill_diagonal(tmp_cov, 1e12)
        tmp_cov[:_nk, :_nk] = cov_full[w, :][:, w]
        cov.append(tmp_cov)

    return zbins, kbins, Pk, cov

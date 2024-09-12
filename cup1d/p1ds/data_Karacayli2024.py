import os

import pandas
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_Karacayli2024(BaseDataP1D):
    def __init__(self, diag_cov=False, kmax_nyq=0.5, z_min=2.19, z_max=10):
        """Read measured P1D from file.
        - diag_cov: for now, use diagonal covariance
        - kmax_nyq: High k cut wrt the Nyquist frequency
        - z_min: z=2.0 bin is not recommended by Karacayli2024
        - z_max: maximum redshift to include"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z, k, Pk, cov = read_from_file(diag_cov, kmax_nyq)

        super().__init__(z, k, Pk, cov, z_min=z_min, z_max=z_max)

        return


def read_from_file(diag_cov, kmax_nyq):
    """Read file containing P1D"""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/Karacayli2024/"
    fname = datadir + "/desi-edrp-lyasb1subt-p1d-detailed-results.txt"

    with open(fname) as _:
        names = _.readline()[1:].strip().split()

    # start by reading the file with measured band power
    # z k1 k2 kc Pfid ThetaP p_final e_stat p_raw p_noise p_fid_qmle p_smooth
    # e_n_syst e_res_syst e_cont_syst e_dla_syst p_sb1 e_sb1_stat e_total
    data = pandas.read_table(
        fname, comment="#", names=names, delim_whitespace=True
    ).to_records(index=False)

    if diag_cov:
        cov_full = np.diag(data["e_total"] ** 2)
    else:
        fname_cov = (
            datadir + "/desi-edrp-lyasb1subt-cov-total-offdiag-results.txt"
        )
        cov_full = np.loadtxt(fname_cov)

    zbins = np.unique(data["z"])
    kbins = np.unique(data["kc"])
    Nk = kbins.size
    Nz = zbins.size

    print("Nz = {} , Nk = {}".format(Nz, Nk))
    Pk = []
    cov = []
    for iz in range(Nz):
        z = zbins[iz]
        # moving Nyquist frequency of the DESI wavelength
        # grid. dlambda = 0.8 A
        dv = 2.99792458e5 * 0.8 / 1215.67 / (1 + z)
        kmax = kmax_nyq * np.pi / dv

        w = np.isclose(data["z"], z) & (data["kc"] < kmax)
        tmp_d = np.zeros(Nk)
        _nk = w.sum()
        tmp_d[:_nk] = data["p_final"][w]
        Pk.append(tmp_d)

        tmp_cov = np.zeros((Nk, Nk))
        # Fill non-existing k bins with inf
        # such that covariance is still invertible
        np.fill_diagonal(tmp_cov, np.inf)
        tmp_cov[:_nk, :_nk] = cov_full[w, :][:, w]
        cov.append(tmp_cov)

    return zbins, kbins, Pk, cov

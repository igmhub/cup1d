import os
import numpy as np
from scipy.linalg import block_diag
from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_Walther2018(BaseDataP1D):
    """Class containing P1D from Walther et al. (2018)."""

    def __init__(self, kmax_kms=1.0, z_min=0, z_max=10, diag_cov=True):
        """Read measured P1D from Walther et al. (2018).

        Problems for the off-diagonal terms of the covariance matrix
        """

        # # optimize
        # kmax_kms = 0.2

        # read redshifts, wavenumbers, power spectra and covariance matrices
        res = read_from_file(kmax_kms, diag_cov=diag_cov)

        (
            zs,
            k_kms,
            Pk_kms,
            cov,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
            Pksmooth_kms,
            cov_stat,
            k_kms_min,
            k_kms_max,
        ) = res

        super().__init__(
            zs,
            k_kms,
            Pk_kms,
            cov,
            z_min=z_min,
            z_max=z_max,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            full_cov_stat_kms=full_cov_stat_kms,
            Pksmooth_kms=Pksmooth_kms,
            cov_stat=cov_stat,
            k_kms_min=k_kms_min,
            k_kms_max=k_kms_max,
        )

        return


def read_from_file(kmax_kms, diag_cov=True):
    """Reconstruct covariance matrix from files."""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/Walther2018/"

    # table 5: measurement masking metals
    # table 6: measurement without masking metals
    # table 7: correlation matrix masking metals
    # table 8: correlation matrix without masking metals

    # start by reading Pk file
    p1d_file = datadir + "/table5.dat"
    # note that the file contains    k P1D(k) / pi
    zs_raw, k_kms_raw, inkPk, inkPkstat = np.loadtxt(p1d_file, unpack=True)

    # now read correlation matrices
    corr_file = datadir + "/table7.dat"
    _ = np.loadtxt(corr_file, unpack=True)
    z_incorr = _[0]
    k_kms_incorr = _[1]
    corr_incorr = _[2:]

    # store unique values of redshift and wavenumber
    z_unique = np.unique(zs_raw)
    Nz = len(z_unique)

    # divide by wavenumber and multiply by pi to get flux power (and error)
    Pk_kms_raw = inkPk / k_kms_raw * np.pi
    err_Pk_kms_raw = inkPkstat / k_kms_raw * np.pi

    zs = []
    k_kms = []
    k_kms_min = []
    k_kms_max = []
    Pk_kms = []
    Pksmooth_kms = []
    cov = []
    cov_stat = []
    mask_raw = np.zeros(len(k_kms_raw), dtype=bool)

    for z in z_unique:
        zs.append(z)
        mask = np.argwhere((zs_raw == z) & (k_kms_raw < kmax_kms))[:, 0]
        mask_raw[mask] = True

        k_kms.append(np.array(k_kms_raw[mask]))
        dk_kms = 0.5 * (k_kms[-1][1:] - k_kms[-1][:-1])
        dk_kms = np.append(dk_kms, dk_kms[-1])
        k_kms_min.append(k_kms[-1] - dk_kms)
        k_kms_max.append(k_kms[-1] + dk_kms)

        _pk = np.array(Pk_kms_raw[mask])
        _err_Pk = np.array(err_Pk_kms_raw[mask])

        # get correlation matrix for this redshift bin
        maskz = np.argwhere(np.abs(z_incorr - z) < 0.05)[:, 0]
        k_kms_allz = k_kms_incorr[maskz]
        ind_k_kms_cut = np.argwhere((k_kms_allz < kmax_kms))[:, 0]
        corr_allz = corr_incorr[:, maskz]
        corr = corr_allz[ind_k_kms_cut, :][:, ind_k_kms_cut]

        if diag_cov:
            _cov = np.diag(_err_Pk**2)
        else:
            _cov = np.multiply(_err_Pk, np.multiply(corr, _err_Pk))
        _cov_stat = _cov

        # TBD (smooth pk)
        _pksmooth = np.array(_pk)

        Pk_kms.append(_pk)
        cov.append(_cov)
        cov_stat.append(_cov_stat)
        Pksmooth_kms.append(_pksmooth)

    full_zs = zs_raw[mask_raw]
    full_Pk_kms = Pk_kms_raw[mask_raw]
    full_cov_kms = block_diag(*cov)
    full_cov_stat_kms = block_diag(*cov_stat)

    return (
        zs,
        k_kms,
        Pk_kms,
        cov,
        full_zs,
        full_Pk_kms,
        full_cov_kms,
        full_cov_stat_kms,
        Pksmooth_kms,
        cov_stat,
        k_kms_min,
        k_kms_max,
    )

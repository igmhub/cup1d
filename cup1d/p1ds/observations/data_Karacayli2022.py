import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_Karacayli2022(BaseDataP1D):
    def __init__(self, kmax_kms=0.1, z_min=0, z_max=10):
        """Read measured P1D from file.
        - diag_cov: for now, use diagonal covariance
        - kmax_kms: limit to low-k where we trust emulator"""

        # optimize
        # kmax_kms = 0.07

        # read redshifts, wavenumbers, power spectra and covariance matrices
        res = read_from_file(kmax_kms)

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


def read_from_file(kmax_kms):
    """Read file containing mock P1D"""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/Karacayli2022/"

    data = np.loadtxt(
        datadir + "final-conservative-p1d-karacayli_etal2021.txt",
        skiprows=1,
        usecols=(1, 2, 3, 4),
        delimiter="|",
    )
    zs_raw = data[:, 0]
    z_unique = np.unique(zs_raw)
    k_kms_raw = data[:, 1]
    Pk_kms_raw = data[:, 2]

    cov_raw = np.loadtxt(
        datadir + "final-conservative-covariance-karacayli_etal2021.txt",
    )
    cov_stat_raw = cov_raw

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
        slice_cov = slice(mask[0], mask[-1] + 1)

        k_kms.append(np.array(k_kms_raw[mask]))
        dk_kms = 0.5 * (k_kms[-1][1:] - k_kms[-1][:-1])
        dk_kms = np.append(dk_kms, dk_kms[-1])
        k_kms_min.append(k_kms[-1] - dk_kms)
        k_kms_max.append(k_kms[-1] + dk_kms)

        _pk = np.array(Pk_kms_raw[mask])
        _cov = np.array(cov_raw[slice_cov, slice_cov])
        _cov_stat = np.array(cov_stat_raw[slice_cov, slice_cov])

        # TBD (smooth pk)
        _pksmooth = np.array(_pk)

        Pk_kms.append(_pk)
        cov.append(_cov)
        cov_stat.append(_cov_stat)
        Pksmooth_kms.append(_pksmooth)

    full_zs = zs_raw[mask_raw]
    full_Pk_kms = Pk_kms_raw[mask_raw]
    full_cov_kms = cov_raw[mask_raw, :][:, mask_raw]
    full_cov_stat_kms = cov_stat_raw[mask_raw, :][:, mask_raw]

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

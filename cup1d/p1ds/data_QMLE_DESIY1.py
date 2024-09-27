import os
from astropy.io import fits
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D, _drop_zbins


class P1D_QMLE_DESIY1(BaseDataP1D):
    def __init__(self, full_cov=False, z_min=2, z_max=10):
        """Read measured P1D from file.
        - full_cov: for now, no covariance between redshift bins
        - z_min: z=2.0 bin is not recommended by Karacayli2024
        - z_max: maximum redshift to include"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        zs, k_kms, Pk_kms, cov = read_from_file(full_cov=full_cov)

        super().__init__(zs, k_kms, Pk_kms, cov, z_min=z_min, z_max=z_max)

        return


def read_from_file(full_cov=False):
    """Read file containing P1D"""

    # folder storing P1D measurement
    datadir = BaseDataP1D.BASEDIR + "/QMLE_DESIY1/"
    fname = datadir + "/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"

    hdu = fits.open(fname)

    zs_raw = hdu[1].data["z"]
    k_kms_raw = hdu[1].data["k"]
    Pk_kms_raw = hdu[1].data["Plya"]
    cov_raw = hdu[2].data.copy()
    diag_cov_raw = np.diag(cov_raw)

    z_unique = np.unique(zs_raw)

    zs = []
    k_kms = []
    Pk_kms = []
    cov = []

    for z in z_unique:
        zs.append(z)
        mask = np.argwhere((zs_raw == z) & (diag_cov_raw > 0))[:, 0]
        slice_cov = slice(mask[0], mask[-1] + 1)
        k_kms.append(np.array(k_kms_raw[mask]))
        Pk_kms.append(np.array(Pk_kms_raw[mask]))
        cov.append(np.array(cov_raw[slice_cov, slice_cov]))

    return zs, k_kms, Pk_kms, cov

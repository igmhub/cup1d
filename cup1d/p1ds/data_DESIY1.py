import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


class P1D_DESIY1(BaseDataP1D):
    def __init__(
        self,
        p1d_fname=None,
        z_min=0,
        z_max=10,
        cov_only_diag=False,
        sys_only_diag=False,
    ):
        """Read measured P1D from file.
        - full_cov: for now, no covariance between redshift bins
        - z_min: z=2.0 bin is not recommended by Karacayli2024
        - z_max: maximum redshift to include"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        res = read_from_file(
            p1d_fname=p1d_fname,
            cov_only_diag=cov_only_diag,
            sys_only_diag=sys_only_diag,
        )
        (
            zs,
            k_kms,
            Pk_kms,
            cov,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            self.blinding,
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
        )

        return


def read_from_file(
    p1d_fname=None,
    kmin=1e-3,
    nknyq=0.5,
    max_cov=1e3,
    cov_only_diag=False,
    sys_only_diag=False,
):
    """Read file containing P1D"""

    # folder storing P1D measurement
    try:
        hdu = fits.open(p1d_fname)
    except:
        raise ValueError("Cannot read: ", p1d_fname)

    dict_with_keys = {}
    for ii in range(len(hdu)):
        if "EXTNAME" in hdu[ii].header:
            if hdu[ii].header["EXTNAME"] == "P1D_BLIND":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii
            elif hdu[ii].header["EXTNAME"] == "COVARIANCE":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii
            elif hdu[ii].header["EXTNAME"] == "COVARIANCE_STAT":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii
            elif hdu[ii].header["EXTNAME"] == "COVARIANCE_SYST":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii

    if "P1D_BLIND" not in dict_with_keys:
        raise ValueError("Cannot find P1D_BLIND in: ", p1d_fname)

    iuse = dict_with_keys["P1D_BLIND"]
    if "VELUNITS" in hdu[iuse].header:
        if hdu[iuse].header["VELUNITS"] == False:
            raise ValueError("Not velocity units in: ", p1d_fname)
    blinding = None
    if "BLINDING" in hdu[iuse].header:
        if hdu[iuse].header["BLINDING"] is not None:
            blinding = hdu[iuse].header["BLINDING"]
    elif "EXTNAME" in hdu[iuse].header:
        if hdu[iuse].header["EXTNAME"] == "P1D_BLIND":
            blinding = True

    if sys_only_diag:
        cov_stat_raw = hdu[dict_with_keys["COVARIANCE_STAT"]].data.copy()
        cov_syst_raw = hdu[dict_with_keys["COVARIANCE_SYST"]].data.copy()
        # add systematic error just to diagonal elements
        cov_raw = cov_stat_raw.copy()
        ind = np.arange(cov_raw.shape[0])
        cov_raw[ind, ind] += np.diag(cov_syst_raw)
    else:
        cov_raw = hdu[dict_with_keys["COVARIANCE"]].data.copy()

    zs_raw = hdu[iuse].data["Z"]
    k_kms_raw = hdu[iuse].data["K"]
    Pk_kms_raw = hdu[iuse].data["PLYA"]
    diag_cov_raw = np.diag(cov_raw)
    if cov_only_diag:
        _cov = np.zeros_like(cov_raw)
        _cov[np.arange(_cov.shape[0]), np.arange(_cov.shape[0])] = diag_cov_raw
        cov_raw = _cov

    z_unique = np.unique(zs_raw)
    mask_raw = np.zeros(len(k_kms_raw), dtype=bool)

    zs = []
    k_kms = []
    Pk_kms = []
    cov = []
    for z in z_unique:
        dv = 2.99792458e5 * 0.8 / 1215.67 / (1 + z)
        k_nyq = np.pi / dv
        zs.append(z)
        mask = np.argwhere(
            (zs_raw == z)
            & (diag_cov_raw > 0)
            & (diag_cov_raw < max_cov)
            & np.isfinite(Pk_kms_raw)
            & np.isfinite(diag_cov_raw)
            & (k_kms_raw > kmin)
            & (k_kms_raw < k_nyq * nknyq)
        )[:, 0]
        mask_raw[mask] = True

        slice_cov = slice(mask[0], mask[-1] + 1)
        k_kms.append(np.array(k_kms_raw[mask]))

        # add emulator error
        _pk = np.array(Pk_kms_raw[mask])
        _cov = np.array(cov_raw[slice_cov, slice_cov])

        Pk_kms.append(_pk)
        cov.append(_cov)

    full_zs = zs_raw[mask_raw]
    full_Pk_kms = Pk_kms_raw[mask_raw]
    full_cov_kms = cov_raw[mask_raw, :][:, mask_raw]

    return zs, k_kms, Pk_kms, cov, full_zs, full_Pk_kms, full_cov_kms, blinding

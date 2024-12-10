import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from cup1d.p1ds.base_p1d_mock import BaseMockP1D


class P1D_challenge_DESIY1(BaseMockP1D):
    def __init__(self, theory, true_cosmo, p1d_fname=None, z_min=0, z_max=10):
        """Read measured P1D from file.
        - full_cov: for now, no covariance between redshift bins
        - z_min: z=2.0 bin is not recommended by Karacayli2024
        - z_max: maximum redshift to include"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        res = read_from_file(p1d_fname=p1d_fname)
        (
            zs,
            k_kms,
            Pk_kms,
            cov_Pk_kms,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            self.blinding,
        ) = res

        # set theory (just to save truth)
        theory.model_igm.set_fid_igm(np.array(zs))
        theory.set_fid_cosmo(np.array(zs), input_cosmo=true_cosmo)

        super().__init__(
            zs,
            k_kms,
            Pk_kms,
            cov_Pk_kms,
            z_min=z_min,
            z_max=z_max,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            theory=theory,
        )

        return


def read_from_file(p1d_fname=None, kmin=1e-3, nknyq=0.5):
    """Read file containing P1D"""

    # folder storing P1D measurement
    print("Reading: ", p1d_fname)
    try:
        hdu = fits.open(p1d_fname)
    except:
        raise ValueError("Cannot read: ", p1d_fname)

    if "VELUNITS" in hdu[1].header:
        if hdu[1].header["VELUNITS"] == False:
            raise ValueError("Not velocity units in: ", p1d_fname)
    blinding = None
    if "BLINDING" in hdu[1].header:
        if hdu[1].header["BLINDING"] is not None:
            blinding = hdu[1].header["BLINDING"]
    # hack
    blinding = None

    # compressed parameters do not agree between codes!!
    # keys = ["modelname", "Delta_star", "N_STAR", "alpha_star"]
    # dict_conv = {
    #     "Delta_star": "Delta2_star",
    #     "N_STAR": "n_star",
    #     "alpha_star":"alpha_star"
    # }
    # for key in keys:
    #     if key == "modelname":
    #         print(hdu[1].header[key])
    #     else:
    #         print(dict_conv[key], hdu[1].header[key])

    # input_sim = hdu[1].header["modelname"]

    zs_raw = hdu[1].data["Z"]
    k_kms_raw = hdu[1].data["K"]
    Pk_kms_raw = hdu[1].data["PLYA"]
    cov_raw = hdu[3].data.copy()
    diag_cov_raw = np.diag(cov_raw)

    z_unique = np.unique(zs_raw)
    mask_raw = np.zeros(len(k_kms_raw), dtype=bool)

    zs = []
    k_kms = []
    Pk_kms = []
    cov = []
    tot = 0
    for z in z_unique:
        dv = 2.99792458e5 * 0.8 / 1215.67 / (1 + z)
        k_nyq = np.pi / dv
        zs.append(z)
        mask = np.argwhere(
            (zs_raw == z)
            & (diag_cov_raw > 0)
            & np.isfinite(Pk_kms_raw)
            & (k_kms_raw > kmin)
            & (k_kms_raw < k_nyq * nknyq)
        )[:, 0]
        tot += len(mask)
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

    return (
        zs,
        k_kms,
        Pk_kms,
        cov,
        full_zs,
        full_Pk_kms,
        full_cov_kms,
        blinding,
    )

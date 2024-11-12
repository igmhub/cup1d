import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D, _drop_zbins
from cup1d.likelihood import lya_theory


class P1D_DESIY1(BaseDataP1D):
    def __init__(
        self,
        p1d_fname=None,
        z_min=0,
        z_max=10,
    ):
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
            cov,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            self.blinding,
        ) = res

        ## TODO, get fiducial cosmo from file!

        if "fiducial" in p1d_fname:
            true_sim_label = "nyx_central"
        elif "CGAN" in p1d_fname:
            true_sim_label = "nyx_seed"
        elif "grid_3" in p1d_fname:
            true_sim_label = "nyx_3"
        else:
            true_sim_label = None

        # set truth if possible
        if true_sim_label is not None:
            self.input_sim = true_sim_label
            model_igm = IGM(np.array(zs), fid_sim_igm=true_sim_label)
            model_cont = Contaminants()
            true_cosmo = self._get_cosmo()
            theory = lya_theory.Theory(
                zs=np.array(zs),
                emulator=None,
                fid_cosmo=true_cosmo,
                model_igm=model_igm,
                model_cont=model_cont,
            )
            self.set_truth(theory, zs)

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


def read_from_file(p1d_fname=None, kmin=1e-3, nknyq=0.5):
    """Read file containing P1D"""

    # folder storing P1D measurement
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

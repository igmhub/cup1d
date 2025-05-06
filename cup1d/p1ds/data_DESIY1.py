import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D


def compute_cov(syst, type_measurement="QMLE", type_analysis="red"):
    if type_measurement == "QMLE":
        sys_labels = [
            "E_DLA_COMPLETENESS",
            "E_BAL_COMPLETENESS",
            "E_RESOLUTION",
            "E_CONTINUUM",
            "E_CONTINUUM_ADD",
            "E_NOISE_SCALE",
            "E_NOISE_ADD",
        ]

        if type_analysis == "fid":
            sys_labels_corr = [
                "E_DLA_COMPLETENESS",
                "E_BAL_COMPLETENESS",
                "E_RESOLUTION",
                "E_CONTINUUM",
                "E_NOISE_SCALE",
            ]
            sys_labels_ucorr = [
                "E_NOISE_ADD",
                "E_CONTINUUM_ADD",
            ]
        elif type_analysis == "red":
            sys_labels_corr = [
                # "E_DLA_COMPLETENESS",
                # "E_BAL_COMPLETENESS",
                "E_RESOLUTION",
                "E_CONTINUUM",
                "E_NOISE_SCALE",
            ]
            sys_labels_ucorr = [
                "E_NOISE_ADD",
                "E_CONTINUUM_ADD",
            ]
        elif type_analysis == "xred":
            sys_labels_corr = [
                # "E_DLA_COMPLETENESS",
                # "E_BAL_COMPLETENESS",
                "E_RESOLUTION",
                "E_CONTINUUM",
                # "E_NOISE_SCALE",
            ]
            sys_labels_ucorr = [
                # "E_NOISE_ADD",
                "E_CONTINUUM_ADD",
            ]
    elif type_measurement == "FFT":
        sys_labels = [
            "E_PSF",
            "E_RESOLUTION",
            "E_SIDE_BAND",
            "E_LINES",
            "E_DLA",
            "E_BAL",
            "E_CONTINUUM",
            "E_DLA_COMPLETENESS",
            "E_BAL_COMPLETENESS",
        ]
        if type_analysis == "fid":
            sys_labels_corr = [
                "E_PSF",
                "E_RESOLUTION",
                "E_SIDE_BAND",
                "E_LINES",
                "E_DLA",
                "E_BAL",
                "E_CONTINUUM",
                "E_DLA_COMPLETENESS",
                "E_BAL_COMPLETENESS",
            ]
            sys_labels_ucorr = []
        elif (type_analysis == "red") | (type_analysis == "xred"):
            sys_labels_corr = [
                "E_PSF",
                "E_RESOLUTION",
                "E_SIDE_BAND",
                "E_LINES",
                "E_DLA",
                "E_BAL",
                "E_CONTINUUM",
                # "E_DLA_COMPLETENESS",
                # "E_BAL_COMPLETENESS",
            ]
            sys_labels_ucorr = []
    else:
        return None

    zz_unique = np.unique(syst["Z"])
    nelem = syst[sys_labels[0]].shape[0]
    cov = np.zeros((nelem, nelem))
    ind_diag = np.diag_indices_from(cov)
    for lab in sys_labels:
        if lab in sys_labels_ucorr:
            cov[ind_diag] += syst[lab] ** 2
        elif lab in sys_labels_corr:
            for z in zz_unique:
                ind = np.argwhere(syst["Z"] == z)[:, 0]
                cov[:, slice(ind[0], ind[-1] + 1)][
                    slice(ind[0], ind[-1] + 1), :
                ] += np.outer(syst[lab][ind], syst[lab][ind])
    return cov


class P1D_DESIY1(BaseDataP1D):
    def __init__(
        self,
        p1d_fname=None,
        z_min=0,
        z_max=10,
        cov_syst_type="red",
    ):
        """Read measured P1D from file.
        - full_cov: for now, no covariance between redshift bins
        - z_min: z=2.0 bin is not recommended by Karacayli2024
        - z_max: maximum redshift to include"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        res = read_from_file(p1d_fname=p1d_fname, cov_syst_type=cov_syst_type)
        (
            zs,
            k_kms,
            Pk_kms,
            cov,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            self.blinding,
            Pksmooth_kms,
            cov_stat,
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
            Pksmooth_kms=Pksmooth_kms,
            cov_stat=cov_stat,
        )

        return


def read_from_file(
    p1d_fname=None,
    kmin=1e-3,
    nknyq=0.5,
    max_cov=1e3,
    cov_syst_type="red",
):
    """Read file containing P1D"""

    # folder storing P1D measurement
    try:
        hdu = fits.open(p1d_fname)
    except:
        raise ValueError("Cannot read: ", p1d_fname)

    if "fft" in p1d_fname:
        type_measurement = "FFT"
    elif "qmle" in p1d_fname:
        type_measurement = "QMLE"
    else:
        raise ValueError("Cannot find type_measurement in: ", p1d_fname)

    dict_with_keys = {}
    for ii in range(len(hdu)):
        if "EXTNAME" in hdu[ii].header:
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

    cov_stat_raw = hdu[dict_with_keys["COVARIANCE_STAT"]].data.copy()
    cov_syst_raw = compute_cov(
        hdu[dict_with_keys["SYSTEMATICS"]].data,
        type_measurement=type_measurement,
        type_analysis=cov_syst_type,
    )
    cov_raw = cov_stat_raw + cov_syst_raw

    zs_raw = hdu[iuse].data["Z"]
    k_kms_raw = hdu[iuse].data["K"]
    Pk_kms_raw = hdu[iuse].data["PLYA"]
    diag_cov_raw = np.diag(cov_raw)

    # smooth Pk, for adding emulator error
    if type_measurement == "QMLE":
        Pksmooth_kms_raw = hdu[iuse].data["PSMOOTH"]
    elif type_measurement == "FFT":
        p1d_fname_qmle = os.path.join(
            os.path.dirname(p1d_fname),
            "desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v2.fits",
        )
        _ = fits.open(p1d_fname_qmle)
        ksmooth_kms_raw = _[1].data["K"]
        zsmooth = _[1].data["Z"]
        Pksmooth_kms_raw = _[1].data["PSMOOTH"]

    z_unique = np.unique(zs_raw)
    mask_raw = np.zeros(len(k_kms_raw), dtype=bool)

    zs = []
    k_kms = []
    Pk_kms = []
    Pksmooth_kms = []
    cov = []
    cov_stat = []
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
        _cov_stat = np.array(cov_stat_raw[slice_cov, slice_cov])

        if type_measurement == "QMLE":
            _pksmooth = np.array(Pksmooth_kms_raw[mask])
        elif type_measurement == "FFT":
            mask2 = zsmooth == z
            _pksmooth = np.interp(
                k_kms_raw[mask], ksmooth_kms_raw[mask2], Pksmooth_kms_raw[mask2]
            )

        Pk_kms.append(_pk)
        cov.append(_cov)
        cov_stat.append(_cov_stat)
        Pksmooth_kms.append(_pksmooth)

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
        Pksmooth_kms,
        cov_stat,
    )

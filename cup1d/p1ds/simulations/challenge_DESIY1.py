from astropy.io import fits
import numpy as np

from cup1d.p1ds.base_p1d_mock import BaseMockP1D


class P1D_challenge_DESIY1(BaseMockP1D):
    """Represent the challenge DESIY1 P1D data product."""
    def __init__(self, theory, true_cosmo, p1d_fname=None, z_min=0, z_max=10):
        """Load a DESI Y1 challenge P1D realization and record its truth.

        Parameters
        ----------
        theory : object
            Cup1D theory object.  Its fiducial IGM and cosmology are reset to
            the challenge truth before that truth is stored in the mock.
        true_cosmo : object
            Cosmology object corresponding to the challenge realization.
        p1d_fname : str or path-like, optional
            Challenge FITS file containing ``P1D`` and ``COVARIANCE`` tables.
        z_min, z_max : float, default: 0, 10
            Inclusive redshift range retained from the file.
        """

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


def read_from_file(p1d_fname=None, kmin=1e-3, nknyq=0.5, max_cov=1e3):
    """Read and quality-filter a DESI Y1 challenge P1D FITS product.

    Parameters
    ----------
    p1d_fname : str or path-like
        FITS file with velocity-space P1D and covariance extensions.
    kmin : float, default: 1e-3
        Strict lower wavenumber cut in ``s / km``.
    nknyq : float, default: 0.5
        Fraction of the redshift-dependent Nyquist frequency retained.
    max_cov : float, default: 1e3
        Retained-diagonal covariance ceiling in ``(km / s)**2``.  This option
        is accepted for reader compatibility; this implementation only uses
        positivity and finite-value filtering.

    Returns
    -------
    tuple
        Per-redshift and concatenated P1D vectors in ``km / s``, covariance
        blocks in ``(km / s)**2``, and blinding metadata.

    Raises
    ------
    ValueError
        If the FITS file cannot be opened, lacks a ``P1D`` extension, or is
        not tagged as velocity-space data.
    """

    # folder storing P1D measurement
    print("Reading: ", p1d_fname)
    try:
        hdu = fits.open(p1d_fname)
    except:
        raise ValueError("Cannot read: ", p1d_fname)

    dict_with_keys = {}
    for ii in range(len(hdu)):
        if "EXTNAME" in hdu[ii].header:
            if hdu[ii].header["EXTNAME"] == "P1D":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii
            elif hdu[ii].header["EXTNAME"] == "COVARIANCE":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii
            elif hdu[ii].header["EXTNAME"] == "COVARIANCE_STAT":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii
            elif hdu[ii].header["EXTNAME"] == "COVARIANCE_SYST":
                dict_with_keys[hdu[ii].header["EXTNAME"]] = ii

    if "P1D" not in dict_with_keys:
        raise ValueError("Cannot find P1D in: ", p1d_fname)

    iuse = dict_with_keys["P1D"]
    if "VELUNITS" in hdu[iuse].header:
        if hdu[iuse].header["VELUNITS"] == False:
            raise ValueError("Not velocity units in: ", p1d_fname)
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

    cov_raw = hdu[dict_with_keys["COVARIANCE"]].data.copy()

    zs_raw = hdu[iuse].data["Z"]
    k_kms_raw = hdu[iuse].data["K"]
    Pk_kms_raw = hdu[iuse].data["PLYA"]
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

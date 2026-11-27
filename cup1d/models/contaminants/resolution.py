import numpy as np
from cup1d.models.contaminants.base_contaminants import Contaminant
from cup1d.utils.batch import as_redshift_rows, unbatch


def get_Rz_Naim(z):
    # 4.1 https://arxiv.org/abs/2306.06316
    """Convert a fixed 0.8-Angstrom width into a velocity width.

    Parameters
    ----------
    z : float or numpy.ndarray
        Absorber redshift.

    Returns
    -------
    float or numpy.ndarray
        Width in km/s, equal to ``c*0.8/((1+z)*1215.67)``.
    """
    c_kms = 2.99792458e5
    lya_AA = 1215.67  # angstroms
    Delta_lambda_AA = 0.8  # angstroms
    # kms2AA = lya_AA * (1 + z) / c_kms
    # k_A = k_kms / kms2AA
    Rz = c_kms * Delta_lambda_AA / (1 + z) / lya_AA
    return Rz


RESOLUTION_FORMS = ("exponential", "linear")


def resolution_correction(r_coeff, z, k_kms, form="exponential"):
    """Multiplicative resolution correction ``exp(x)`` or ``1 + x``.

    ``x = 2 R_coeff R(z)^2 k^2`` with ``R`` from :func:`get_Rz_Naim`; both
    forms agree to first order in ``x``. ``r_coeff`` and ``k_kms`` broadcast.
    """
    x = 2 * np.asarray(r_coeff) * get_Rz_Naim(z) ** 2 * np.asarray(k_kms) ** 2
    return np.exp(x) if form == "exponential" else 1 + x


class Resolution(Contaminant):
    """Model the multiplicative P1D correction for spectrograph resolution.

    The redshift-dependent ``R_coeff`` history controls the correction applied
    by ``get_contamination`` to velocity-space power spectra.
    """

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        z_max_res=10,
        fid_vals=None,
        flat_priors=None,
        null_vals=None,
        Gauss_priors=None,
        form="exponential",
    ):
        """Initialize the resolution-coefficient history and its priors.

        Parameters
        ----------
        coeffs, prop_coeffs, free_param_names
            Optional fixed history, metadata, and free coefficient selection.
        z_0 : float, default=3
            Pivot redshift of the coefficient polynomial.
        z_max_res : float, default=10
            Retained compatibility resolution cutoff.
        fid_vals, flat_priors, null_vals, Gauss_priors : mapping, optional
            History defaults and prior metadata forwarded to :class:`Contaminant`.
        form : {"exponential", "linear"}, default="exponential"
            Correction ``exp(x)`` for a Gaussian resolution kernel whose
            variance is off by ``-2 R_coeff R^2``, or its first-order
            expansion ``1 + x``, with ``x = 2 R_coeff R^2 k^2`` and ``R`` the
            velocity width of a 0.8 Angstrom pixel.
        """
        if form not in RESOLUTION_FORMS:
            raise ValueError(f"resolution form must be one of {RESOLUTION_FORMS}, got {form!r}")
        self.form = form

        list_coeffs = ["R_coeff"]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {"R_coeff": [[-0.5, 0.5], [-0.1, 0.1]]}

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "R_coeff_ztype": "pivot",
                "R_coeff_otype": "const",
            }

        # fiducial values
        if (fid_vals is None) | (len(fid_vals["R_coeff"]) == 0):
            fid_vals = {
                "R_coeff": [0, 0],
            }

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            null_vals=null_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(self, z, k_kms, like_params=None):
        """Multiplicative spectrograph-resolution correction on velocity k grids.

        Parameters
        ----------
        z : array-like
            Redshifts.
        k_kms : sequence of ndarray
            Per-redshift wavenumbers in s/km.
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.

        Returns
        -------
        list of ndarray
            One array per redshift: ``(n_k,)`` for one point,
            ``(n_batch, n_k)`` for a batch.
        """
        z, k_kms = as_redshift_rows(z, k_kms)
        values, scalar = self.histories(z, like_params, null=None)
        output = [
            resolution_correction(values["R_coeff"][:, iz, None], z[iz], k_kms[iz][None, :], self.form)
            for iz in range(len(z))
        ]
        return unbatch(output, scalar)


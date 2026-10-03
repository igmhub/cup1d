import numpy as np
from cup1d.models.contaminants.base_contaminants import Contaminant


def fun_cont(damp, k):
    # Based on Walther+24, their equation is weird
    """Evaluate the BOSS HCD multiplicative power correction.

    Parameters
    ----------
    damp : float or numpy.ndarray
        Dimensionless contamination amplitude.
    k : float or numpy.ndarray
        Line-of-sight wavenumber in s/km.

    Returns
    -------
    float or numpy.ndarray
        Dimensionless factor multiplying P1D. Zero amplitude gives unity.

    Notes
    -----
    The implemented rational profile is singular at ``15000*k = 9.9``.
    This helper applies no domain clipping or regularization.
    """
    return 1 + 1 / (1 - (1 / (15000 * k - 8.9))) * damp


class HCD_BOSS(Contaminant):
    """BOSS HCD multiplicative correction based on Walther et al. (2024).

    The model parameterizes the residual high-column-density absorber effect
    on the velocity-space flux P1D and exposes its redshift evolution through
    the common contaminant interface.
    """

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        flat_priors=None,
        null_vals=None,
        Gauss_priors=None,
    ):
        """Initialize the one-family HCD damping contaminant model.

        Parameters
        ----------
        coeffs : mapping, optional
            Coefficient values supplied to the base contaminant model.
        prop_coeffs : mapping, optional
            Redshift interpolation and output-transform metadata.
        free_param_names : sequence of str, optional
            Likelihood parameter names that are free.
        z_0 : float, default=3.0
            Pivot redshift for coefficient evolution.
        fid_vals, flat_priors, null_vals, Gauss_priors : mapping, optional
            Fiducial values and prior/null definitions. Defaults implement the
            calibrated ``HCD_damp1`` model.
        """
        # list of all coefficients
        list_coeffs = [
            "HCD_damp1",
        ]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {
                "HCD_damp1": [[-0.5, 0.5], [-10.0, -1.0]],
            }

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "HCD_damp1_ztype": "pivot",
                "HCD_damp1_otype": "exp",
            }

        # fiducial values
        if fid_vals is None:
            fid_vals = {
                "HCD_damp1": [0, -20.0],
            }

        # null values
        if null_vals is None:
            null_vals = {
                "HCD_damp1": -21.5,
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
        """Evaluate scalar-point HCD multiplicative corrections.

        Parameters
        ----------
        z : array-like
            Redshifts matching ``k_kms`` rows.
        k_kms : sequence of array-like
            Wavenumber rows in s/km.
        like_params : mapping, optional
            Coefficient overrides in likelihood parameterization.

        Returns
        -------
        list of ndarray or ndarray
            One correction row per redshift, or the row itself for one
            redshift.
        """

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )
            if key in self.null_vals:
                if self.prop_coeffs[key + "_otype"] == "const":
                    null = self.null_vals[key]
                else:
                    null = np.exp(self.null_vals[key])
                _ = vals[key] <= null
                vals[key][_] = 0
        # print(vals)

        dla_corr = []
        for iz in range(len(z)):
            cont = fun_cont(vals[f"HCD_damp1"][iz], k_kms[iz])
            dla_corr.append(cont)

        if len(z) == 1:
            dla_corr = dla_corr[0]

        return dla_corr

    def get_contamination_batch(self, z, k_kms, like_params):
        """Evaluate batched HCD corrections.

        Parameters
        ----------
        z : array-like
            Redshifts matching ``k_kms`` rows.
        k_kms : sequence of array-like
            Wavenumber rows in s/km.
        like_params : mapping
            Columnar parameter values with a leading batch dimension.

        Returns
        -------
        list of ndarray
            One array per redshift with shape ``(n_batch, n_k)``.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        values = self.get_value_batch("HCD_damp1", z, like_params)
        null = np.exp(self.null_vals["HCD_damp1"])
        values = np.where(values <= null, 0.0, values)
        return [
            fun_cont(values[:, iz, None], np.asarray(k_kms[iz])[None, :])
            for iz in range(len(z))
        ]

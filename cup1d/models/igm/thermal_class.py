import numpy as np
from cup1d.models.igm.base_igm import IGM_model
from lace.cosmo import thermal_broadening


class Thermal(IGM_model):
    """Model the IGM thermal history used by cup1d."""
    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_igm=None,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        """Initialize thermal-width and temperature-density slope histories.

        Parameters are forwarded to :class:`IGM_model`; both default histories
        multiply their fiducial values with constant-output spline factors.
        """
        list_coeffs = ["sigT_kms", "gamma"]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "interp_spl"
                prop_coeffs[coeff + "_otype"] = "const"

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-1, 1], [-1.25, 1.25]]

        for coeff in list_coeffs:
            if coeff not in fid_vals:
                if prop_coeffs[coeff + "_ztype"] == "pivot":
                    fid_vals[coeff] = [0, 1]
                else:
                    fid_vals[coeff] = np.ones(
                        len(prop_coeffs[coeff + "_znodes"])
                    )

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
            fid_igm=fid_igm,
        )

    def get_sigT_kms(self, z, like_params=None, name_par="sigT_kms"):
        """Evaluate the thermal broadening width in km/s.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the history.
        like_params : mapping, optional
            Physical named coefficients overriding the configured history.
        name_par : str, default="sigT_kms"
            Coefficient and fiducial-history key.

        Returns
        -------
        float or numpy.ndarray
            Thermal broadening width in km/s. Dividing by ``H(z)/(1+z)``
            converts this width to a comoving length in Mpc.
        """

        sigT_kms = self.get_value(name_par, z, like_params=like_params)
        sigT_kms *= self.fid_interp[name_par](z)
        return sigT_kms

    def get_T0(self, z, like_params=None, name_par="sigT_kms"):
        """Convert the thermal broadening history to temperature.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the history.
        like_params : mapping, optional
            Physical named coefficients passed to ``get_sigT_kms``.
        name_par : str, default="sigT_kms"
            Thermal broadening history key.

        Returns
        -------
        float or numpy.ndarray
            Temperature at mean density in kelvin, using LaCE's thermal
            broadening conversion.
        """

        sigT_kms = self.get_sigT_kms(
            z, like_params=like_params, name_par=name_par
        )
        T0 = thermal_broadening.T0_from_broadening_kms(sigT_kms)
        return T0

    def get_gamma(self, z, like_params=None, name_par="gamma"):
        """Evaluate the temperature--density relation slope.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the history.
        like_params : mapping, optional
            Physical named coefficients overriding the configured history.
        name_par : str, default="gamma"
            Coefficient and fiducial-history key.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless ``gamma`` in ``T = T0 * (rho/rho_mean)**(gamma-1)``.
        """

        gamma = self.get_value(name_par, z, like_params=like_params)
        gamma *= self.fid_interp[name_par](z)
        return gamma

import numpy as np
from cup1d.models.igm.base_igm import IGM_model


class MeanFlux(IGM_model):
    """Model the IGM meanflux history used by cup1d."""
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
        """Initialize the effective-optical-depth history model.

        Parameters are forwarded to :class:`IGM_model`; the default history is
        a spline-interpolated exponential multiplier of fiducial ``tau_eff``.
        """
        list_coeffs = ["tau_eff"]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "interp_spl"
                prop_coeffs[coeff + "_otype"] = "exp"

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-0.5, 0.5], [-0.2, 0.2]]

        for coeff in list_coeffs:
            if coeff not in fid_vals:
                if prop_coeffs[coeff + "_ztype"] == "pivot":
                    fid_vals[coeff] = [0, 0]
                else:
                    fid_vals[coeff] = np.zeros(
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

    def get_tau_eff(self, z, like_params=None, name_par="tau_eff"):
        """Evaluate the effective optical-depth history.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the history.
        like_params : mapping, optional
            Physical named coefficients overriding the configured history.
        name_par : str, default="tau_eff"
            Coefficient and fiducial-history key.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless effective optical depth: the history multiplier
            returned by ``get_value`` times the fiducial optical depth.
        """

        tau_eff = self.get_value(name_par, z, like_params=like_params)
        tau_eff *= self.fid_interp[name_par](z)
        return tau_eff

    def get_mean_flux(self, z, like_params=None):
        """Evaluate the mean transmitted flux fraction.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the history.
        like_params : mapping, optional
            Physical named coefficients passed to ``get_tau_eff``.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless flux fraction ``exp(-tau_eff)`` at ``z``.
        """
        tau = self.get_tau_eff(z, like_params=like_params)
        return np.exp(-tau)

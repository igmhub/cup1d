import numpy as np
from cup1d.models.igm.base_igm import IGM_model


class Pressure(IGM_model):
    """Model the IGM pressure history used by cup1d."""
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
        """Initialize the inverse pressure-smoothing scale history model.

        Parameters are forwarded to :class:`IGM_model`; ``kF_kms`` remains a
        velocity-space inverse length in s/km.
        """
        list_coeffs = ["kF_kms"]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "interp_spl"
                prop_coeffs[coeff + "_otype"] = "const"

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-1, 1], [-1.2, 1.2]]

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

    def get_kF_kms(self, z, like_params=None, name_par="kF_kms"):
        """Return the inverse pressure-smoothing scale in velocity units.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the history.
        like_params : mapping, optional
            Physical named history coefficients. Omitted coefficients use
            the configured history; these values are not sampler coordinates.
        name_par : str, default="kF_kms"
            History key used for both coefficients and fiducial interpolation.

        Returns
        -------
        float or numpy.ndarray
            Inverse smoothing scale in s/km, evaluated at ``z``. Multiplying
            by ``H(z)/(1+z)`` converts it to an inverse comoving length in
            1/Mpc. This quantity is an inverse length, not a broadening width.
        """

        kF_kms = self.get_value(name_par, z, like_params=like_params)
        kF_kms *= self.fid_interp[name_par](z)
        return kF_kms

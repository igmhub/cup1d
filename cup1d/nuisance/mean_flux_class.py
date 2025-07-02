import numpy as np
import os
import lace
from cup1d.nuisance.base_igm import IGM_model


class MeanFlux(IGM_model):
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
        list_coeffs = ["tau_eff"]

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-0.5, 0.5], [-0.2, 0.2]]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "pivot"
                prop_coeffs[coeff + "_otype"] = "exp"

        if fid_igm is None:
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + "data/sim_suites/Australia20/IGM_histories.npy"
            try:
                igm_hist = np.load(fname, allow_pickle=True).item()
            except:
                raise ValueError(
                    fname
                    + " not found. You can produce it using the LaCE"
                    + r" script save_mpg_IGM.py"
                )
            else:
                fid_igm = igm_hist["mpg_central"]

        if fid_vals is None:
            fid_vals = {}
            for coeff in list_coeffs:
                fid_vals[coeff] = [0, 0]

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

    def get_tau_eff(self, z, like_params=[], name_par="tau_eff"):
        """Effective optical depth at the input redshift"""

        tau_eff = self.get_value(name_par, z, like_params=like_params)
        if self.prop_coeffs[name_par + "_ztype"] == "pivot":
            tau_eff *= self.fid_interp(z)
        return tau_eff

    def get_mean_flux(self, z, like_params=[]):
        """Mean transmitted flux fraction at the input redshift"""
        tau = self.get_tau_eff(z, like_params=like_params)
        return np.exp(-tau)

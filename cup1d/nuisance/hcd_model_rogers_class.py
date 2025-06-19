import numpy as np
from cup1d.nuisance.base_contaminants import Contaminant


def fun_damping(k_kms, it):
    a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
    b_0 = np.array([36.449, 81.388, 162.95, 429.58])
    return 1 / (a_0[it] * np.exp(k_kms * b_0[it]) - 1) ** 2


class HCD_Model_Rogers(Contaminant):
    """New model for HCD contamination"""

    def __init__(
        self,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        # list of all coefficients
        list_coeffs = [
            "HCD_damp1",
            "HCD_damp2",
            "HCD_damp3",
            "HCD_damp4",
            "HCD_const",
        ]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {
                "HCD_damp": [[-0.5, 0.5], [-4, 1e-6]],
                "HCD_const": [[-1, 1], [-0.2, 1e-6]],
            }

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "HCD_damp1_ztype": "pivot",
                "HCD_damp2_ztype": "pivot",
                "HCD_damp3_ztype": "pivot",
                "HCD_damp4_ztype": "pivot",
                "HCD_const_ztype": "pivot",
                "HCD_damp1_otype": "exp",
                "HCD_damp2_otype": "exp",
                "HCD_damp3_otype": "exp",
                "HCD_damp4_otype": "exp",
                "HCD_const_otype": "const",
            }

        # fiducial values
        if fid_vals is None:
            fid_vals = {
                "HCD_damp1": [0, -4],
                "HCD_damp2": [0, -4],
                "HCD_damp3": [0, -4],
                "HCD_damp4": [0, -4],
                "HCD_const": [0, 0],
            }

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by HCDs"""
        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )

        dla_corr = []
        for iz in range(len(z)):
            cont = 1 + vals["HCD_const"][iz] + np.zeros_like(k_kms[iz])
            for it in range(4):
                cont += vals[f"HCD_damp{it+1}"][iz] * fun_damping(k_kms[iz], it)
            dla_corr.append(cont)

        if len(z) == 1:
            dla_corr = dla_corr[0]

        return dla_corr

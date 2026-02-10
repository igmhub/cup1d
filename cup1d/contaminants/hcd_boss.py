import numpy as np
from cup1d.contaminants.base_contaminants import Contaminant


def fun_cont(damp, k):
    # Based on Walther+24, their equation is weird
    return 1 + 1 / (1 - (1 / (15000 * k - 8.9))) * damp


class HCD_BOSS(Contaminant):
    """HCD contamination Eq. 5.2 Walther+24"""

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

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by HCDs"""

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

import numpy as np
from cup1d.nuisance.base_contaminants import Contaminant


class MetalModel(Contaminant):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        metal_label,
        coeffs=None,
        prop_coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_X=3."""

        # label identifying the metal line
        self.metal_label = metal_label
        if metal_label == "SiIIa_SiIIb":
            self.lambda_rest = [1190.42, 1193.28]
            self.osc_strength = [0.277, 0.575]
        elif metal_label == "CIVa_CIVb":
            self.lambda_rest = [1548.20, 1550.78]
            self.osc_strength = [0.190, 0.095]
        elif metal_label == "MgIIa_MgIIb":
            self.lambda_rest = [2795.53, 2802.70]
            self.osc_strength = [0.608, 0.303]
        else:
            if lambda_rest is None:
                raise ValueError("need to specify lambda_rest", metal_label)
        c_kms = 299792.458
        self.dv = np.log(self.lambda_rest[1] / self.lambda_rest[0]) * c_kms
        self.ratio_f = np.min(self.osc_strength) / np.max(self.osc_strength)

        list_coeffs = ["f_" + metal_label, "s_" + metal_label]

        if flat_priors is None:
            flat_priors = {
                "f_" + metal_label: [[-3, 3], [-11, -0.5]],
                "s_" + metal_label: [[-1, 1], [-10, 10]],
            }

        if prop_coeffs is None:
            prop_coeffs = {
                "f_" + metal_label + "_ztype": "pivot",
                "s_" + metal_label + "_ztype": "pivot",
                "f_" + metal_label + "_otype": "exp",
                "s_" + metal_label + "_otype": "exp",
            }

        if fid_vals is None:
            fid_vals = {
                "f_" + metal_label: [0, -10.5],
                "s_" + metal_label: [0, -9.5],
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

    def get_contamination(self, z, k_kms, mF, like_params=[]):
        """Multiplicative contamination at a given z and k (in s/km)."""

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )

        metal_corr = []
        for iz in range(len(z)):
            damping = np.exp(
                -1 * vals["s_" + self.metal_label][iz] ** 2 * k_kms[iz] ** 2
            )
            metal_corr.append(
                vals["f_" + self.metal_label][iz]
                * (
                    1
                    + self.ratio_f**2
                    + 2 * self.ratio_f * np.cos(self.dv * k_kms[iz])
                )
                * damping
            )

        return metal_corr

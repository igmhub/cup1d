import numpy as np
import copy
import os
from cup1d.likelihood import likelihood_parameter


class HCD_Model_Rogers2017(object):
    """Model HCD contamination following Rogers et al. (2017)."""

    def __init__(
        self,
        z_0=3.0,
        fid_value=[0, -5],
        null_value=-4,
        ln_A_damp_coeff=None,
        free_param_names=None,
    ):
        self.z_0 = z_0
        self.null_value = null_value
        if ln_A_damp_coeff:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.ln_A_damp_coeff = ln_A_damp_coeff
        else:
            if free_param_names:
                # figure out number of HCD free params
                n_hcd = len([p for p in free_param_names if "ln_A_damp_" in p])
                if n_hcd == 0:
                    n_hcd = 1
            else:
                n_hcd = 1

            self.ln_A_damp_coeff = [0.0] * n_hcd
            self.ln_A_damp_coeff[-1] = fid_value[-1]
            if n_hcd == 2:
                self.ln_A_damp_coeff[-2] = fid_value[-2]

        self.set_parameters()

    def set_parameters(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = []
        Npar = len(self.ln_A_damp_coeff)
        for i in range(Npar):
            name = "ln_A_damp_" + str(i)
            if i == 0:
                # no contamination
                xmin = -4
                # 0 gives 350% contamination low k
                xmax = 0
            else:
                # not optimized
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            value = self.ln_A_damp_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_A_damp_coeff) == len(self.params), "size mismatch"
        return len(self.ln_A_damp_coeff)

    def get_A_damp(self, z, like_params=[]):
        """Amplitude of HCD contamination around z_0"""

        ln_A_damp_coeff = self.get_A_damp_coeffs(like_params=like_params)
        if ln_A_damp_coeff[-1] <= self.null_value:
            return 0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(ln_A_damp_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by HCDs"""
        A_damp = self.get_A_damp(z, like_params=like_params)
        if A_damp == 0:
            return 1

        # values and pivot redshift directly from arXiv:1706.08532
        z_0 = 2
        # parameter order: LLS, Sub-DLA, Small-DLA, Large-DLA
        a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
        a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
        b_0 = np.array([36.449, 81.388, 162.95, 429.58])
        b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
        # compute the z-dependent correction terms
        a_z = a_0 * ((1 + z) / (1 + z_0)) ** a_1
        b_z = b_0 * ((1 + z) / (1 + z_0)) ** b_1
        dla_corr = np.ones(
            k_kms.size
        )  # alpha_0 degenerate with mean flux, set to 1
        # a_lls and a_sub degenerate with each other (as are a_sdla, a_ldla), so only use two values
        dla_corr += (
            A_damp
            * ((1 + z) / (1 + z_0)) ** -3.55
            * (
                (a_z[0] * np.exp(b_z[0] * k_kms) - 1) ** -2
                + (a_z[1] * np.exp(b_z[1] * k_kms) - 1) ** -2
            )
        )
        # dla_corr += (
        #     alpha[1]
        #     * ((1 + z) / (1 + z_0)) ** -3.55
        #     * (
        #         (a_z[2] * np.exp(b_z[2] * k_kms) - 1) ** -2
        #         + (a_z[3] * np.exp(b_z[3] * k_kms) - 1) ** -2
        #     )
        # )

        return dla_corr

    def get_parameters(self):
        """Return likelihood parameters for the HCD model"""
        return self.params

    def get_A_damp_coeffs(self, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            ln_A_damp_coeff = self.ln_A_damp_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_A_damp" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_A_damp_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError(
                    "number of params mismatch in get_A_damp_coeffs"
                )

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_A_damp_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_A_damp_coeff = self.ln_A_damp_coeff

        return ln_A_damp_coeff

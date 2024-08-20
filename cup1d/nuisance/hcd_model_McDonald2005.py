import numpy as np
import copy
import os
from cup1d.likelihood import likelihood_parameter


class HCD_Model_McDonald2005(object):
    """Model HCD contamination following McDonald et al. (2005).
    """

    def __init__(
        self,
        z_0=3.0,
        ln_A_damp_coeff=None,
        free_param_names=None,
    ):

        self.z_0 = z_0
        if ln_A_damp_coeff:
            assert free_param_names is None
            self.ln_A_damp_coeff = ln_A_damp_coeff
        else:
            if free_param_names:
                # figure out number of HCD free params
                n_hcd = len([p for p in free_param_names if "ln_A_damp_" in p])
                if n_hcd > 0:
                    self.ln_A_damp_coeff = [0.0] * n_hcd
                else:
                    # close to no contamination
                    self.ln_A_damp_coeff = [-100]
            else:
                # close to no contamination
                self.ln_A_damp_coeff = [-100]
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_A_damp_coeff) == len(self.params), "size mismatch"
        return len(self.ln_A_damp_coeff)

    def get_A_damp(self, z):
        """Amplitude of HCD contamination around z_0"""
        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(self.ln_A_damp_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_contamination(self, z, k_kms):
        """Multiplicative contamination caused by HCDs"""
        A_damp = self.get_A_damp(z)
        # fitting function from Palanque-Delabrouille et al. (2015)
        # that qualitatively describes Fig 2 of McDonald et al. (2005)
        f_HCD = 0.018 + 1/(15000*k_kms - 8.9)

        return (1 + A_damp * f_HCD)

    def set_parameters(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = []
        Npar = len(self.ln_A_damp_coeff)
        for i in range(Npar):
            name = "ln_A_damp_" + str(i)
            if i == 0:
                xmin = -0.4
                xmax = 0.4
            else:
                xmin = -1
                xmax = 1
            # note non-trivial order in coefficients
            value = self.ln_A_damp_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_parameters(self):
        """Return likelihood parameters for the HCD model"""
        return self.params

    def update_parameters(self, like_params):
        """Update HCD model values using input list of likelihood parameters"""

        Npar = self.get_Nparam()

        # loop over likelihood parameters
        for like_par in like_params:
            if "ln_A_damp" not in like_par.name:
                continue
            # make sure you find the parameter
            found = False
            # loop over parameters in HCD model
            for ip in range(len(self.params)):
                if self.params[ip].name == like_par.name:
                    assert found == False, "can not update parameter twice"
                    self.ln_A_damp_coeff[Npar - ip - 1] = like_par.value
                    found = True
            assert found == True, "could not update parameter " + like_par.name

        return

    def get_new_model(self, like_params=[]):
        """Return copy of model, updating values from list of parameters"""

        hcd_model = HCD_Model_McDonald2005(
            z_0=self.z_0,
            ln_A_damp_coeff=copy.deepcopy(self.ln_A_damp_coeff),
        )
        hcd_model.update_parameters(like_params)
        return hcd_model

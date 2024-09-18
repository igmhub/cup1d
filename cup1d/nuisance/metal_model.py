import numpy as np
import copy
from cup1d.likelihood import likelihood_parameter


class MetalModel(object):
    """Model the contamination from Silicon Lya cross-correlations"""

    def __init__(
        self,
        metal_label,
        lambda_rest=None,
        z_X=3.0,
        ln_X_coeff=None,
        fid_value=-10,
        free_param_names=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_X=3."""

        # label identifying the metal line
        self.metal_label = metal_label
        if metal_label == "SiIII":
            self.lambda_rest = 1206.50  # from McDonald et al. 2006)
            if lambda_rest:
                if lambda_rest != self.lambda_rest:
                    raise ValueError("inconsistent lambda_rest")
        else:
            if lambda_rest is None:
                raise ValueError("need to specify lambda_rest", metal_label)
            self.lambda_rest = lambda_rest

        # power law pivot point
        self.z_X = z_X

        # figure out parameters
        if ln_X_coeff:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.ln_X_coeff = ln_X_coeff
        else:
            if free_param_names:
                # figure out number of free params for this metal line
                param_tag = "ln_" + metal_label
                n_X = len([p for p in free_param_names if param_tag in p])
                if n_X == 0:
                    n_X = 1
            else:
                n_X = 1
            # start with value from McDonald et al. (2006), and no z evolution
            self.ln_X_coeff = [0.0] * n_X
            self.ln_X_coeff[-1] = fid_value

        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        if len(self.ln_X_coeff) != len(self.X_params):
            raise ValueError("parameter size mismatch")
        return len(self.ln_X_coeff)

    def set_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.X_params = []
        Npar = len(self.ln_X_coeff)
        for i in range(Npar):
            name = "ln_" + self.metal_label + "_" + str(i)
            if i == 0:
                # log of overall amplitude at z_X
                # no contamination
                xmin = -11
                # max 10% contamination (oscillations)
                xmax = -4
            else:
                # not optimized
                xmin = -5
                xmax = 5
            # note non-trivial order in coefficients
            value = self.ln_X_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.X_params.append(par)
        return

    def get_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.X_params

    def get_X_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            ln_X_coeff = self.ln_X_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_" + self.metal_label in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_X_coeff
            elif Npar != len(self.X_params):
                raise ValueError("number of params mismatch in get_X_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.X_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.X_params[ip].name
                    )
                else:
                    ln_X_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_X_coeff = self.ln_X_coeff

        return ln_X_coeff

    def get_amplitude(self, z, like_params=[]):
        """Amplitude of contamination at a given z"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)

        ln_X_coeff = self.get_X_coeffs(like_params)

        xz = np.log((1 + z) / (1 + self.z_X))
        ln_poly = np.poly1d(ln_X_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_dv_kms(self):
        """Velocity separation where the contamination is stronger"""

        # these constants should be written elsewhere
        lambda_lya = 1215.67
        c_kms = 2.997e5

        # we should properly compute this (check McDonald et al. 2006)
        # dv_kms = (lambda_lya-self.lambda_rest)/lambda_lya*c_kms
        dv_kms = np.log(lambda_lya / self.lambda_rest) * c_kms

        return dv_kms

    def get_contamination(self, z, k_kms, mF, like_params=[]):
        """Multiplicative contamination at a given z and k (in s/km).
        The mean flux (mF) is used scale it (see McDonald et al. 2006)"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)
        f = self.get_amplitude(z, like_params=like_params)
        a = f / (1 - mF)
        # v3 in McDonald et al. (2006)
        dv = self.get_dv_kms()

        return 1 + a**2 + 2 * a * np.cos(dv * k_kms)

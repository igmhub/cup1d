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
        d_coeff=None,
        fid_value=[[0, 0], [2, -10]],
        null_value=[2, -10],
        free_param_names=None,
    ):
        """Model the evolution of a metal contamination (SiII or SiIII).
        We use a power law around z_X=3."""

        # label identifying the metal line
        self.metal_label = metal_label
        if metal_label == "SiIII":
            self.lambda_rest = 1206.50  # from McDonald et al. 2006
        elif metal_label == "SiII":
            self.lambda_rest = 1192.5  # like in Chabanier+19, Karacali+24
        else:
            if lambda_rest is None:
                raise ValueError("need to specify lambda_rest", metal_label)
            self.lambda_rest = lambda_rest

        # power law pivot point
        self.z_X = z_X
        # value below which no contamination (speed up model)
        self.null_value = null_value
        self.dv = self.get_dv_kms()

        # figure out parameters
        if ln_X_coeff and d_coeff:
            if free_param_names is not None:
                raise ValueError("cannot specify coeff and free_param_names")
            self.ln_X_coeff = ln_X_coeff
            self.d_coeff = d_coeff
        else:
            if free_param_names:
                # figure out number of free params for this metal line
                param_tag = "ln_" + metal_label + "_"
                n_X = len([p for p in free_param_names if param_tag in p])
                if n_X == 0:
                    n_X = 1

                param_tag = "d_" + metal_label + "_"
                n_D = len([p for p in free_param_names if param_tag in p])
                if n_D == 0:
                    n_D = 1
            else:
                n_X = 1
                n_D = 1
            # start with value from McDonald et al. (2006), and no z evolution
            self.ln_X_coeff = [0.0] * n_X
            self.ln_X_coeff[-1] = fid_value[-1][-1]
            if n_X == 2:
                self.ln_X_coeff[-2] = fid_value[-2][-1]

            # this is for the k-dependent damping
            self.d_coeff = [0.0] * n_D
            self.d_coeff[-1] = fid_value[-1][0]
            if n_D == 2:
                self.d_coeff[-2] = fid_value[-2][0]

        # store list of likelihood parameters (might be fixed or free)
        self.set_X_parameters()
        self.set_D_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        if len(self.ln_X_coeff) != len(self.X_params):
            raise ValueError("parameter size mismatch")
        return len(self.ln_X_coeff)

    def set_X_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.X_params = []
        Npar = len(self.ln_X_coeff)
        for i in range(Npar):
            name = "ln_" + self.metal_label + "_" + str(i)
            if i == 0:
                # log of overall amplitude at z_X
                # no contamination
                xmin = -11
                # max 10% contamination (oscillations) -4
                xmax = -2
            else:
                # not optimized
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            value = self.ln_X_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.X_params.append(par)
        return

    def set_D_parameters(self):
        """Setup likelihood parameters for metal model"""

        self.D_params = []
        Npar = len(self.d_coeff)
        for i in range(Npar):
            name = "d_" + self.metal_label + "_" + str(i)
            if i == 0:
                # log of overall amplitude at z_X
                # no contamination
                xmin = 1
                # TBD
                xmax = 6.5
            else:
                # not optimized
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            value = self.d_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.D_params.append(par)
        return

    def get_X_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.X_params

    def get_D_parameters(self):
        """Return likelihood parameters from the metal model"""
        return self.D_params

    def get_X_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            ln_X_coeff = self.ln_X_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_" + self.metal_label + "_" in par.name:
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

    def get_D_coeffs(self, like_params=[]):
        """Return list of coefficients for metal model"""

        if like_params:
            d_coeff = self.d_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "d_" + self.metal_label + "_" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.d_coeff
            elif Npar != len(self.D_params):
                raise ValueError("number of params mismatch in get_D_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.D_params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.D_params[ip].name
                    )
                else:
                    d_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            d_coeff = self.d_coeff

        return d_coeff

    def get_amplitude(self, z, like_params=[]):
        """Amplitude of contamination at a given z"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)

        ln_X_coeff = self.get_X_coeffs(like_params)

        if ln_X_coeff[-1] <= self.null_value[-1]:
            return 0

        xz = np.log((1 + z) / (1 + self.z_X))
        ln_poly = np.poly1d(ln_X_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_damping(self, z, like_params=[]):
        """Damping of contamination at a given z"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)

        d_coeff = self.get_D_coeffs(like_params)

        if d_coeff[-1] <= self.null_value[0]:
            return 0

        xz = np.log((1 + z) / (1 + self.z_X))
        poly = np.poly1d(d_coeff)
        return np.exp(poly(xz))

    def get_dv_kms(self):
        """Velocity separation where the contamination is stronger"""

        # these constants should be written elsewhere
        lambda_lya = 1215.67
        c_kms = 2.997e5

        # we should properly compute this (check McDonald et al. 2006)
        # dv_kms = (lambda_lya-self.lambda_rest)/lambda_lya*c_kms
        # v3 in McDonald et al. (2006)
        dv_kms = np.log(lambda_lya / self.lambda_rest) * c_kms

        return dv_kms

    def get_contamination(self, z, k_kms, mF, like_params=[]):
        """Multiplicative contamination at a given z and k (in s/km).
        The mean flux (mF) is used scale it (see McDonald et al. 2006)"""

        # Note that this represents "f" in McDonald et al. (2006)
        # It is later rescaled by <F> to compute "a" in eq. (15)
        f = self.get_amplitude(z, like_params=like_params)
        if f == 0:
            return 1
        adamp = self.get_damping(z, like_params=like_params)

        a = f / (1 - mF)
        # faster damping than exponential, but still long tail
        alpha = 1.5
        adim_damp = adamp * k_kms
        damping = (1 + adim_damp) ** alpha * np.exp(-1 * adim_damp**alpha)
        cont = 1 + a**2 + 2 * a * np.cos(self.dv * k_kms) * damping
        return cont

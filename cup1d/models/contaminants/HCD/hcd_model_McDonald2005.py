import numpy as np
from cup1d.likelihood import parameter as likelihood_parameter


class HCD_Model_McDonald2005(object):
    """Model HCD contamination following McDonald et al. (2005)."""

    def __init__(
        self,
        z_0=3.0,
        fid_A_damp=[0, -6],
        null_value=-6,
        ln_A_damp_coeff=None,
        free_param_names=None,
    ):
        self.z_0 = z_0
        self.null_value = null_value
        if fid_A_damp is None:
            fid_A_damp = [0, -6]

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
            self.ln_A_damp_coeff[-1] = fid_A_damp[-1]
            if n_hcd == 2:
                self.ln_A_damp_coeff[-2] = fid_A_damp[-2]

        self.set_parameters()

    def set_parameters(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = {}
        Npar = len(self.ln_A_damp_coeff)
        for i in range(Npar):
            name = "ln_A_damp_" + str(i)
            if i == 0:
                # no contamination
                xmin = -7
                # 0 gives 30% contamination low k
                xmax = 2.5
            else:
                # not optimized
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            value = self.ln_A_damp_coeff[Npar - i - 1]
            par = likelihood_parameter.make_parameter(
                name=name, value=value, min_value=xmin, max_value=xmax, hessian_transform="exp"
            )
            self.params[name] = par

        return

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_A_damp_coeff) == len(self.params), "size mismatch"
        return len(self.ln_A_damp_coeff)

    def get_A_damp(self, z, like_params=None):
        """Amplitude of HCD contamination around z_0"""

        ln_A_damp_coeff = self.get_A_damp_coeffs(like_params=like_params)
        if ln_A_damp_coeff[-1] <= self.null_value:
            return 0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(ln_A_damp_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_contamination(self, z, k_kms, like_params=None):
        """Multiplicative contamination caused by HCDs"""
        A_damp = self.get_A_damp(z, like_params=like_params)
        if A_damp == 0:
            return 1

        # fitting function from Palanque-Delabrouille et al. (2015)
        # that qualitatively describes Fig 2 of McDonald et al. (2005)
        f_HCD = 0.018 + 1 / (15000 * k_kms - 8.9)
        return 1 + A_damp * f_HCD

    def get_contamination_batch(self, z, k_kms, like_params):
        """McDonald HCD correction with items shaped ``(batch, k_z)``."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        n_batch = len(next(iter(like_params.values())))
        coeff = np.broadcast_to(np.asarray(self.ln_A_damp_coeff, dtype=float), (n_batch, len(self.ln_A_damp_coeff))).copy()
        for index in range(len(self.ln_A_damp_coeff)):
            name = f"ln_A_damp_{index}"
            if name in like_params:
                coeff[:, -(index + 1)] = np.asarray(like_params[name])
        xz = np.log((1 + z) / (1 + self.z_0))
        log_amplitude = np.zeros((n_batch, len(z)))
        for coefficient in coeff.T:
            log_amplitude = log_amplitude * xz[None, :] + coefficient[:, None]
        amplitude = np.where(coeff[:, -1, None] <= self.null_value, 0.0, np.exp(log_amplitude))
        return [1 + amplitude[:, iz, None] * (0.018 + 1 / (15000 * np.asarray(k_kms[iz])[None, :] - 8.9)) for iz in range(len(z))]

    def get_parameters(self):
        """Return likelihood parameters for the HCD model"""
        return self.params

    def get_A_damp_coeffs(self, like_params=None):
        """Return list of mean flux coefficients"""

        if like_params:
            ln_A_damp_coeff = self.ln_A_damp_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par_name, par_value in like_params.items():
                if "ln_A_damp" in par_name:
                    Npar += 1
                    array_names.append(par_name)
                    array_values.append(par_value)
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
                _ = np.argwhere(list(self.params)[ip] == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + list(self.params)[ip]
                    )
                else:
                    ln_A_damp_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_A_damp_coeff = self.ln_A_damp_coeff

        return ln_A_damp_coeff

    def plot_contamination(
        self,
        z,
        k_kms,
        ln_A_damp_coeff=None,
        plot_every_iz=1,
        cmap=None,
        smooth_k=False,
    ):
        """Delegate to :func:`cup1d.postprocessing.contaminants.plot_hcd_contamination`."""
        from cup1d.postprocessing.contaminants import plot_hcd_contamination as _plot

        return _plot(self, z, k_kms, ln_A_damp_coeff, plot_every_iz, cmap, smooth_k)

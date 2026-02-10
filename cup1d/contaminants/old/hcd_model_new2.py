import numpy as np
import copy, os
from matplotlib import pyplot as plt
from cup1d.utils.utils import get_discrete_cmap
from cup1d.likelihood import likelihood_parameter


class HCD_Model_new2(object):
    """New model for HCD contamination"""

    def __init__(
        self,
        coeffs=None,
        free_param_names=None,
        z_0=3.0,
        fid_vals=None,
        null_vals=None,
        flat_priors=None,
        Gauss_priors=None,
    ):
        self.list_coeffs = [
            "HCD_damp1",
            "HCD_scale1",
            "HCD_damp2",
            "HCD_scale2",
            "HCD_damp3",
            "HCD_scale3",
            "HCD_const",
        ]
        self.coeffs = {}

        # store input data
        self.z_0 = z_0
        self.Gauss_priors = Gauss_priors

        if flat_priors is None:
            self.flat_priors = {
                "HCD_damp": [[-0.5, 0.5], [-10, 5]],
                "HCD_scale": [[-1, 1], [1, 10]],
                "HCD_const": [[-1, 1], [-0.2, 1e-6]],
            }
        else:
            self.flat_priors = flat_priors

        if null_vals is None:
            self.null_vals = {"HCD_damp": -9.5}
        else:
            self.null_vals = null_vals

        if fid_vals is None:
            self.fid_vals = {
                "HCD_damp1": [0, -9.5],
                "HCD_scale1": [0, 5],
                "HCD_damp2": [0, -9.5],
                "HCD_scale2": [0, 5],
                "HCD_damp3": [0, -9.5],
                "HCD_scale3": [0, 5],
                "HCD_const": [0, 0],
            }
        else:
            self.fid_vals = fid_vals

        if coeffs is not None:
            if free_param_names is not None:
                raise ValueError(
                    "can not specify both coeffs and free_param_names"
                )
            for key in self.list_coeffs:
                if key in coeffs:
                    setattr(self, self.coeffs[key], coeffs[key])
                else:
                    raise ("HCD coeff not specified:", key)
        else:
            if free_param_names is None:
                raise ValueError(
                    "must specify either coeffs or free_param_names"
                )

            # figure out number of HCD free params
            self.n_pars = {}
            for key in self.list_coeffs:
                self.n_pars[key] = len(
                    [p for p in free_param_names if key + "_" in p]
                )
                if self.n_pars[key] == 0:
                    npar = 1
                else:
                    npar = self.n_pars[key]
                self.coeffs[key] = [0.0] * npar
                for ii in range(npar):
                    self.coeffs[key][-(ii + 1)] = self.fid_vals[key][-(ii + 1)]

        self.set_params()

    def set_params(self):
        """Setup likelihood parameters in the HCD model"""

        self.params = {}

        for key in self.list_coeffs:
            values = self.coeffs[key]
            for ii in range(len(values)):
                name = key + "_" + str(ii)
                set_prior = False
                for key2 in self.flat_priors:
                    if key2 in name:
                        xmin = self.flat_priors[key2][-(ii + 1)][0]
                        xmax = self.flat_priors[key2][-(ii + 1)][1]
                        set_prior = True
                        break

                if set_prior is False:
                    raise ValueError("Cannot find priors of:", key)

                # note non-trivial order in coefficients
                Gwidth = None
                if self.Gauss_priors is not None:
                    if name in self.Gauss_priors:
                        Gwidth = self.Gauss_priors[name][-(ii + 1)]

                par = likelihood_parameter.LikelihoodParameter(
                    name=name,
                    value=values[-(ii + 1)],
                    min_value=xmin,
                    max_value=xmax,
                    Gauss_priors_width=Gwidth,
                )
                self.params[name] = par

    def get_Nparam(self):
        """Number of parameters in the model"""
        n_params = len(self.params)
        n_coeffs = 0
        for coeff in self.coeffs:
            n_coeffs += len(coeff)
        if n_params != n_coeffs:
            raise ValueError("mismatch between number of params and coeffs")
        return n_params

    def get_value(self, name, z, like_params=[]):
        """Amplitude of HCD contamination around z_0"""

        coeff = self.get_coeff(name, like_params=like_params)
        if name in self.null_vals:
            if coeff[-1] <= self.null_vals[name]:
                return 0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(coeff)
        ln_out = ln_poly(xz)
        if name == "HCD_const":
            return ln_out
        else:
            return np.exp(ln_out)

    def get_param(self, name):
        """Return likelihood parameters for the HCD model"""
        return self.params[name]

    def get_coeff(self, name, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            coeff = self.coeffs[name].copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if name in par.name:
                    array_names.append(par.name)
                    array_values.append(par.value)
                    Npar += 1
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # return fiducial value
            if Npar == 0:
                return coeff
            elif Npar != self.n_pars[name]:
                print(Npar, self.n_pars[name])
                raise ValueError("number of params mismatch for: " + name)

            for ii in range(Npar):
                ind_arr = np.argwhere(name + "_" + str(ii) == array_names)[0, 0]
                coeff[-(ii + 1)] = array_values[ind_arr]
        else:
            coeff = self.coeffs[name]

        return coeff

    def get_contamination(self, z, k_kms, like_params=[]):
        """Multiplicative contamination caused by HCDs"""
        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z, like_params=like_params)
            )

        dla_corr = []
        for iz in range(len(z)):
            dla_corr.append(
                1
                + vals["HCD_const"][iz]
                + vals["HCD_damp1"][iz]
                / np.exp(k_kms[iz] * vals["HCD_scale1"][iz])
                + vals["HCD_damp2"][iz]
                / np.exp(k_kms[iz] * vals["HCD_scale2"][iz])
                + vals["HCD_damp3"][iz]
                / np.exp(k_kms[iz] * vals["HCD_scale3"][iz])
            )

        if len(z) == 1:
            dla_corr = dla_corr[0]

        return dla_corr

    def plot_contamination(
        self,
        z,
        k_kms,
        coeffs=None,
        plot_every_iz=1,
        cmap=None,
        dict_data=None,
        zrange=[0, 10],
        name=None,
    ):
        """Plot the contamination model"""

        # plot for fiducial value
        if coeffs is None:
            coeffs = self.coeffs
        else:
            for key in self.list_coeffs:
                if key not in coeffs:
                    coeffs[key] = self.coeffs[key]

        if cmap is None:
            cmap = get_discrete_cmap(len(z))

        hcd_model = HCD_Model_new2(coeffs=coeffs)

        cont = hcd_model.get_contamination(z, k_kms)

        yrange = [1, 1]
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig2, ax2 = plt.subplots(
            len(z), sharex=True, sharey=True, figsize=(8, len(z) * 4)
        )
        if len(z) == 1:
            ax2 = [ax2]

        for ii in range(0, len(z), plot_every_iz):
            if dict_data is not None:
                indz = np.argwhere(np.abs(dict_data["zs"] - z[ii]) < 1.0e-3)[
                    :, 0
                ]
                if len(indz) != 1:
                    continue
                else:
                    indz = indz[0]

            if (z[ii] > zrange[1]) | (z[ii] < zrange[0]):
                continue

            ax1.plot(
                k_kms[ii], cont[ii], color=cmap(ii), label="z=" + str(z[ii])
            )
            ax2[ii].plot(
                k_kms[ii], cont[ii], color=cmap(ii), label="z=" + str(z[ii])
            )

            yrange[0] = min(yrange[0], np.min(cont))
            yrange[1] = max(yrange[1], np.max(cont))

            if dict_data is not None:
                yy = (
                    dict_data["p1d_data"][indz]
                    / dict_data["p1d_model"][indz]
                    * cont[ii]
                )
                err_yy = (
                    dict_data["p1d_err"][indz]
                    / dict_data["p1d_model"][indz]
                    * cont[ii]
                )

                ax1.errorbar(
                    dict_data["k_kms"][indz],
                    yy,
                    err_yy,
                    marker="o",
                    linestyle=":",
                    color=cmap(ii),
                    alpha=0.5,
                )
                ax2[ii].errorbar(
                    dict_data["k_kms"][indz],
                    yy,
                    err_yy,
                    marker="o",
                    linestyle=":",
                    color=cmap(ii),
                    alpha=0.5,
                )

        ax1.axhline(1, color="k", linestyle=":")
        ax1.legend(ncol=4)
        ax1.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$k$ [1/Mpc]")
        ax1.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,HCD}$")
        for ax in ax2:
            ax.axhline(1, color="k", linestyle=":")
            ax.legend()
            ax.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
            ax.set_xlabel(r"$k$ [1/Mpc]")
            ax.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,HCD}$")
            ax.set_xscale("log")

        fig1.tight_layout()
        fig2.tight_layout()

        if name is None:
            fig1.show()
            fig2.show()
        else:
            if len(z) != 1:
                fig1.savefig(name + "_all.pdf")
                fig1.savefig(name + "_all.png")
            fig2.savefig(name + "_z.pdf")
            fig2.savefig(name + "_z.png")

        return

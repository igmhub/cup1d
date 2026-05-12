import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from cup1d.likelihood import likelihood_parameter
from cup1d.utils.utils import get_discrete_cmap, get_path_repo


class AGN_Model:
    """Multiplicative AGN feedback correction.

    This model follows the Chabanier et al. (2020) correction

    ``P1D(AGN) = (1 + beta) * P1D(noAGN)``

    where the redshift-dependent amplitude is represented as a polynomial in
    ``log((1 + z) / (1 + z_0))`` and the scale dependence is read from the
    tabulated AGN correction file.
    """

    def __init__(
        self,
        z_0=3.0,
        fid_value=None,
        null_value=-5.5,
        ln_AGN_coeff=None,
        free_param_names=None,
    ):
        """Build the AGN feedback model.

        Parameters
        ----------
        z_0 : float, optional
            Pivot redshift for the polynomial amplitude.
        fid_value : list[float] or None, optional
            Fiducial polynomial coefficients. The last entry is the amplitude
            at ``z_0``.
        null_value : float, optional
            Log-amplitude threshold below which the correction is disabled.
        ln_AGN_coeff : list[float] or None, optional
            Fixed polynomial coefficients. Mutually exclusive with
            ``free_param_names``.
        free_param_names : list[str] or None, optional
            Likelihood parameter names used to decide how many AGN coefficients
            are varied.
        """
        if fid_value is None:
            fid_value = [0, -5]
        self.z_0 = z_0
        if fid_value is None:
            fid_value = [0, -5]
        self.null_value = null_value

        if ln_AGN_coeff is not None:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.ln_AGN_coeff = ln_AGN_coeff
        else:
            if free_param_names:
                # figure out number of AGN free params
                n_AGN = len([p for p in free_param_names if "ln_AGN_" in p])
                if n_AGN == 0:
                    n_AGN = 1
            else:
                n_AGN = 1

            self.ln_AGN_coeff = [0.0] * n_AGN
            self.ln_AGN_coeff[-1] = fid_value[-1]
            if n_AGN == 2:
                self.ln_AGN_coeff[-2] = fid_value[-2]

        self.set_parameters()

        self.AGN_z, self.AGN_expansion = _load_agn_file()

    def set_parameters(self):
        """Create likelihood parameters for the AGN amplitude."""

        self.params = []
        Npar = len(self.ln_AGN_coeff)
        for i in range(Npar):
            name = "ln_AGN_" + str(i)
            # priors optimized so we do not get negative values
            if i == 0:
                xmin = -5
                xmax = 1
            else:
                xmin = -10
                xmax = 10
            # note non-trivial order in coefficients
            value = self.ln_AGN_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_Nparam(self):
        """Return the number of free AGN parameters."""
        assert len(self.ln_AGN_coeff) == len(self.params), "size mismatch"
        return len(self.ln_AGN_coeff)

    def get_AGN_damp(self, z, like_params=None):
        """Evaluate the AGN correction amplitude at redshift ``z``."""

        ln_AGN_coeff = self.get_AGN_coeffs(like_params=like_params)
        if ln_AGN_coeff[-1] <= self.null_value:
            return 0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(ln_AGN_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_contamination(self, z, k_kms, like_params=None):
        """Return the multiplicative AGN correction at ``z`` and ``k_kms``."""

        fAGN = self.get_AGN_damp(z, like_params=like_params)
        if fAGN == 0:
            return 1

        if z <= np.max(self.AGN_z):
            yy = self.AGN_expansion[:, 0][None, :] + self.AGN_expansion[:, 1][
                None, :
            ] * np.exp(-self.AGN_expansion[:, 2][None, :] * k_kms[:, None])
            delta = interp1d(self.AGN_z, yy)(z)
        else:
            AGN_upper = self.AGN_expansion[0, 0] + self.AGN_expansion[
                0, 1
            ] * np.exp(-self.AGN_expansion[0, 2] * k_kms)
            AGN_lower = self.AGN_expansion[1, 0] + self.AGN_expansion[
                1, 1
            ] * np.exp(-self.AGN_expansion[1, 2] * k_kms)
            z_upper = self.AGN_z[0]
            z_lower = self.AGN_z[1]
            delta = (AGN_upper - AGN_lower) / (z_upper - z_lower) * (
                z - z_upper
            ) + AGN_upper

        beta = delta * fAGN

        return 1 + beta

    def get_parameters(self):
        """Return the AGN likelihood parameters."""
        return self.params

    def get_AGN_coeffs(self, like_params=None):
        """Return AGN coefficients, updated from likelihood parameters."""

        if like_params:
            ln_AGN_coeff = self.ln_AGN_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_AGN" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_AGN_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError("number of params mismatch in get_AGN_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_AGN_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_AGN_coeff = self.ln_AGN_coeff

        return ln_AGN_coeff

    def plot_contamination(
        self,
        z,
        k_kms,
        ln_AGN_coeff=None,
        plot_every_iz=1,
        cmap=None,
        smooth_k=False,
        dict_data=None,
        zrange=None,
        name=None,
    ):
        """Plot the AGN correction for a set of redshifts and wavenumbers."""

        # plot for fiducial value
        if zrange is None:
            zrange = [0, 10]
        if ln_AGN_coeff is None:
            ln_AGN_coeff = self.ln_AGN_coeff

        if cmap is None:
            cmap = get_discrete_cmap(len(z))

        agn_model = AGN_Model(ln_AGN_coeff=ln_AGN_coeff)

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

            if smooth_k:
                k_use = np.logspace(
                    np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
                )
            else:
                k_use = k_kms[ii]
            cont = agn_model.get_contamination(z[ii], k_use)
            if isinstance(cont, int):
                cont = np.ones_like(k_use)

            ax1.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))
            ax2[ii].plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

            yrange[0] = min(yrange[0], np.min(cont))
            yrange[1] = max(yrange[1], np.max(cont))

            if dict_data is not None:
                yy = (
                    dict_data["p1d_data"][indz]
                    / dict_data["p1d_model"][indz]
                    * cont
                )
                err_yy = (
                    dict_data["p1d_err"][indz]
                    / dict_data["p1d_model"][indz]
                    * cont
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
        ax1.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,AGN}$")
        for ax in ax2:
            ax.axhline(1, color="k", linestyle=":")
            ax.legend()
            ax.set_ylim(yrange[0] * 0.95, yrange[1] * 1.05)
            ax.set_xlabel(r"$k$ [1/Mpc]")
            ax.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^\mathrm{no\,AGN}$")
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


def _load_agn_file():
    """Read the tabulated AGN scale-dependence coefficients."""
    agn_corr_filename = os.path.join(
        get_path_repo("cup1d"), "data", "nuisance", "AGN_corr.dat"
    )
    NzAGN = 9
    AGN_z = np.ndarray(NzAGN, "float")
    AGN_expansion = np.ndarray((NzAGN, 3), "float")
    with open(agn_corr_filename) as datafile:
        for i in range(NzAGN):
            line = datafile.readline()
            values = [float(valstring) for valstring in line.split()]
            AGN_z[i] = values[0]
            AGN_expansion[i] = values[1:]
    return AGN_z, AGN_expansion

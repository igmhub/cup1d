"""Multiplicative AGN feedback correction.

References
----------
.. [1] Chabanier et al. (2020) - Lyman-alpha forest P1D constraints
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
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

    Parameters
    ----------
    z_0 : float, optional
        Pivot redshift for the polynomial amplitude. Default is 3.0.
    fid_value : list[float] | None, optional
        Fiducial polynomial coefficients. The last entry is the amplitude
        at ``z_0``. Default is None, which sets to [0, -5].
    null_value : float, optional
        Log-amplitude threshold below which the correction is disabled.
        Default is -5.5.
    ln_AGN_coeff : list[float] | None, optional
        Fixed polynomial coefficients. Mutually exclusive with
        ``free_param_names``. Default is None.
    free_param_names : list[str] | None, optional
        Likelihood parameter names used to decide how many AGN coefficients
        are varied. Default is None.

    Attributes
    ----------
    z_0 : float
        Pivot redshift for the polynomial amplitude.
    null_value : float
        Log-amplitude threshold below which the correction is disabled.
    ln_AGN_coeff : list[float]
        Polynomial coefficients for the AGN correction amplitude.
    params : list[likelihood_parameter.LikelihoodParameter]
        Likelihood parameters for the AGN model.
    AGN_z : npt.NDArray[np.float64]
        Redshifts where the AGN correction is tabulated.
    AGN_expansion : npt.NDArray[np.float64]
        Tabulated AGN correction coefficients.
    """

    def __init__(
        self,
        z_0: float = 3.0,
        fid_value: list[float] | None = None,
        null_value: float = -5.5,
        ln_AGN_coeff: list[float] | None = None,
        free_param_names: list[str] | None = None,
    ):
        """Initialize the AGN feedback model."""
        if fid_value is None:
            fid_value = [0, -5]
        self.z_0 = z_0
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

    def set_parameters(self) -> None:
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

    def get_Nparam(self) -> int:
        """Return the number of free AGN parameters.

        Returns
        -------
        int
            Number of free AGN parameters.
        """
        assert len(self.ln_AGN_coeff) == len(self.params), "size mismatch"
        return len(self.ln_AGN_coeff)

    def get_AGN_damp(
        self,
        z: float,
        like_params: list | None = None,
        name_par: str = "ln_AGN",
    ) -> float:
        """Evaluate the AGN correction amplitude at redshift ``z``.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name prefix. Default is "ln_AGN".

        Returns
        -------
        float
            AGN damping amplitude.
        """
        ln_AGN_coeff = self.get_AGN_coeffs(like_params=like_params)
        if ln_AGN_coeff[-1] <= self.null_value:
            return 0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(ln_AGN_coeff)
        ln_out = ln_poly(xz)
        return float(np.exp(ln_out))

    def get_contamination(
        self,
        z: float,
        k_kms: npt.NDArray[np.float64],
        like_params: list | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the multiplicative AGN correction at ``z`` and ``k_kms``.

        Parameters
        ----------
        z : float
            Redshift.
        k_kms : npt.NDArray[np.float64]
            Wavenumber in s/km.
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        npt.NDArray[np.float64]
            AGN correction.
        """
        fAGN = self.get_AGN_damp(z, like_params=like_params)
        if fAGN == 0:
            return np.ones_like(k_kms)

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

    def get_parameters(self) -> list[likelihood_parameter.LikelihoodParameter]:
        """Return the AGN likelihood parameters.

        Returns
        -------
        list[likelihood_parameter.LikelihoodParameter]
            List of AGN likelihood parameters.
        """
        return self.params

    def get_AGN_coeffs(
        self, like_params: list | None = None
    ) -> list[float] | npt.NDArray[np.float64]:
        """Return AGN coefficients, updated from likelihood parameters.

        Parameters
        ----------
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        list[float] | npt.NDArray[np.float64]
            AGN coefficients.
        """
        if like_params:
            ln_AGN_coeff = list(self.ln_AGN_coeff)
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_AGN" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names_np = np.array(array_names)
            array_values_np = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_AGN_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError("number of params mismatch in get_AGN_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names_np)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_AGN_coeff[Npar - ip - 1] = array_values_np[_[0]]
        else:
            ln_AGN_coeff = self.ln_AGN_coeff

        return ln_AGN_coeff

    def plot_contamination(
        self,
        z: npt.NDArray[np.float64],
        k_kms: list[npt.NDArray[np.float64]],
        ln_AGN_coeff: list[float] | None = None,
        plot_every_iz: int = 1,
        cmap: plt.Colormap | None = None,
        smooth_k: bool = False,
        dict_data: dict | None = None,
        zrange: list[float] | None = None,
        name: str | None = None,
    ) -> None:
        """Plot the AGN correction for a set of redshifts and wavenumbers.

        Parameters
        ----------
        z : npt.NDArray[np.float64]
            Redshifts.
        k_kms : list[npt.NDArray[np.float64]]
            Wavenumbers for each redshift.
        ln_AGN_coeff : list[float] | None, optional
            AGN coefficients. Default is None.
        plot_every_iz : int, optional
            Plot every N-th redshift. Default is 1.
        cmap : plt.Colormap | None, optional
            Colormap to use. Default is None.
        smooth_k : bool, optional
            Whether to smooth the k axis. Default is False.
        dict_data : dict | None, optional
            Dictionary with data to plot. Default is None.
        zrange : list[float] | None, optional
            Redshift range to plot. Default is None.
        name : str | None, optional
            Name for the output plots. Default is None.
        """
        # plot for fiducial value
        if zrange is None:
            zrange = [0, 10]
        if ln_AGN_coeff is None:
            ln_AGN_coeff = self.ln_AGN_coeff

        if cmap is None:
            cmap = get_discrete_cmap(len(z))

        agn_model = AGN_Model(ln_AGN_coeff=ln_AGN_coeff)

        yrange = [1.0, 1.0]
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
            if isinstance(cont, (int, float)):
                cont = np.ones_like(k_use) * cont

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


def _load_agn_file() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read the tabulated AGN scale-dependence coefficients.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Redshifts and tabulated AGN correction coefficients.
    """
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

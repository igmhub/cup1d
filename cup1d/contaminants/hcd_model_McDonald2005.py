"""High-column-density absorber model from McDonald et al. (2005).

References
----------
.. [1] McDonald et al. (2005) - HCD modeling
.. [2] Palanque-Delabrouille et al. (2015) - SDSS Lyman-alpha forest
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from cup1d.likelihood import likelihood_parameter


class HCD_Model_McDonald2005:
    """Multiplicative HCD correction following McDonald et al. (2005).

    This model provides a multiplicative correction to the 1D power spectrum:
    ``P1D(HCD) = (1 + A_damp * f_HCD(k)) * P1D(noHCD)``.

    Parameters
    ----------
    z_0 : float, optional
        Pivot redshift for the polynomial amplitude. Default is 3.0.
    fid_A_damp : list[float] | None, optional
        Fiducial polynomial coefficients for the damping amplitude.
        Default is None, which sets to [0, -6].
    null_value : float, optional
        Log-amplitude threshold below which the correction is disabled.
        Default is -6.
    ln_A_damp_coeff : list[float] | None, optional
        Fixed polynomial coefficients. Mutually exclusive with
        ``free_param_names``. Default is None.
    free_param_names : list[str] | None, optional
        Likelihood parameter names used to decide how many HCD amplitude
        coefficients are varied. Default is None.

    Attributes
    ----------
    z_0 : float
        Pivot redshift for the polynomial amplitude.
    null_value : float
        Log-amplitude threshold below which the correction is disabled.
    ln_A_damp_coeff : list[float]
        Polynomial coefficients for the HCD amplitude.
    params : list[likelihood_parameter.LikelihoodParameter]
        Likelihood parameters for the HCD model.
    """

    def __init__(
        self,
        z_0: float = 3.0,
        fid_A_damp: list[float] | None = None,
        null_value: float = -6.0,
        ln_A_damp_coeff: list[float] | None = None,
        free_param_names: list[str] | None = None,
    ) -> None:
        """Build the McDonald et al. HCD correction model."""
        if fid_A_damp is None:
            fid_A_damp = [0, -6]
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
            self.ln_A_damp_coeff[-1] = fid_A_damp[-1]
            if n_hcd == 2:
                self.ln_A_damp_coeff[-2] = fid_A_damp[-2]

        self.set_parameters()

    def set_parameters(self) -> None:
        """Create likelihood parameters for the HCD amplitude."""
        self.params = []
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
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

    def get_Nparam(self) -> int:
        """Return the number of free HCD parameters.

        Returns
        -------
        int
            Number of free HCD parameters.
        """
        assert len(self.ln_A_damp_coeff) == len(self.params), "size mismatch"
        return len(self.ln_A_damp_coeff)

    def get_A_damp(
        self,
        z: float,
        like_params: list | None = None,
        name_par: str = "ln_A_damp",
    ) -> float:
        """Evaluate the HCD damping amplitude at redshift ``z``.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name prefix. Default is "ln_A_damp".

        Returns
        -------
        float
            HCD damping amplitude.
        """
        ln_A_damp_coeff = self.get_A_damp_coeffs(like_params=like_params)
        if ln_A_damp_coeff[-1] <= self.null_value:
            return 0.0

        xz = np.log((1 + z) / (1 + self.z_0))
        ln_poly = np.poly1d(ln_A_damp_coeff)
        ln_out = ln_poly(xz)
        return float(np.exp(ln_out))

    def get_contamination(
        self,
        z: float,
        k_kms: npt.NDArray[np.float64],
        like_params: list | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the multiplicative HCD correction at ``z`` and ``k_kms``.

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
            HCD correction.
        """
        A_damp = self.get_A_damp(z, like_params=like_params)
        if A_damp == 0:
            return np.ones_like(k_kms)

        # fitting function from Palanque-Delabrouille et al. (2015)
        # that qualitatively describes Fig 2 of McDonald et al. (2005)
        f_HCD = 0.018 + 1 / (15000 * k_kms - 8.9)
        return 1 + A_damp * f_HCD

    def get_parameters(self) -> list[likelihood_parameter.LikelihoodParameter]:
        """Return the HCD likelihood parameters.

        Returns
        -------
        list[likelihood_parameter.LikelihoodParameter]
            List of HCD likelihood parameters.
        """
        return self.params

    def get_A_damp_coeffs(
        self, like_params: list | None = None
    ) -> list[float] | npt.NDArray[np.float64]:
        """Return HCD coefficients, updated from likelihood parameters.

        Parameters
        ----------
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        list[float] | npt.NDArray[np.float64]
            HCD coefficients.
        """
        if like_params:
            ln_A_damp_coeff = list(self.ln_A_damp_coeff)
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_A_damp" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names_np = np.array(array_names)
            array_values_np = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_A_damp_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError("number of params mismatch in get_A_damp_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names_np)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_A_damp_coeff[Npar - ip - 1] = array_values_np[_[0]]
        else:
            ln_A_damp_coeff = self.ln_A_damp_coeff

        return ln_A_damp_coeff

    def plot_contamination(
        self,
        z: npt.NDArray[np.float64],
        k_kms: list[npt.NDArray[np.float64]],
        ln_A_damp_coeff: list[float] | None = None,
        plot_every_iz: int = 1,
        cmap: Any = None,
        smooth_k: bool = False,
    ) -> None:
        """Plot the HCD correction for a set of redshifts and wavenumbers.

        Parameters
        ----------
        z : npt.NDArray[np.float64]
            Redshifts.
        k_kms : list[npt.NDArray[np.float64]]
            Wavenumbers for each redshift.
        ln_A_damp_coeff : list[float] | None, optional
            HCD coefficients. Default is None.
        plot_every_iz : int, optional
            Plot every N-th redshift. Default is 1.
        cmap : Any, optional
            Colormap to use. Default is None.
        smooth_k : bool, optional
            Whether to smooth the k axis. Default is False.
        """
        from matplotlib import pyplot as plt

        # plot for fiducial value
        if ln_A_damp_coeff is None:
            ln_A_damp_coeff = self.ln_A_damp_coeff

        hcd_model = HCD_Model_McDonald2005(ln_A_damp_coeff=ln_A_damp_coeff)

        for ii in range(0, len(z), plot_every_iz):
            if smooth_k:
                k_use = np.logspace(
                    np.log10(k_kms[ii][0]), np.log10(k_kms[ii][-1]), 200
                )
            else:
                k_use = k_kms[ii]

            cont = hcd_model.get_contamination(z[ii], k_use)
            if isinstance(cont, (int, float)):
                cont = np.ones_like(k_use) * cont
            if cmap is None:
                plt.plot(k_use, cont, label="z=" + str(z[ii]))
            else:
                plt.plot(k_use, cont, color=cmap(ii), label="z=" + str(z[ii]))

        plt.axhline(1, color="k", linestyle=":")

        plt.legend()
        plt.xscale("log")
        plt.xlabel(r"$k$ [1/Mpc]")
        plt.ylabel("HCD contamination")
        plt.tight_layout()

"""Supernova feedback correction for the 1D Lyman-alpha power spectrum."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cup1d.likelihood import likelihood_parameter


class SN_Model:
    """Multiplicative supernova feedback model following Viel et al. (2013).

    Parameters
    ----------
    z_0 : float, optional
        Pivot redshift for the polynomial amplitude. Default is 3.0.
    fid_value : list[float] | None, optional
        Fiducial polynomial coefficients. The last entry is the amplitude
        at ``z_0``. Default is None, which sets to [0, -4].
    null_value : float, optional
        Log-amplitude threshold below which the correction is disabled.
        Default is -4.
    ln_SN_coeff : list[float] | None, optional
        Fixed polynomial coefficients. Mutually exclusive with
        ``free_param_names``. Default is None.
    free_param_names : list[str] | None, optional
        Likelihood parameter names used to decide how many SN coefficients
        are varied. Default is None.

    Attributes
    ----------
    z_0 : float
        Pivot redshift for the polynomial amplitude.
    null_value : float
        Log-amplitude threshold below which the correction is disabled.
    ln_SN_coeff : list[float]
        Polynomial coefficients for the SN correction amplitude.
    params : list[likelihood_parameter.LikelihoodParameter]
        Likelihood parameters for the SN model.
    """

    def __init__(
        self,
        z_0: float = 3.0,
        fid_value: list[float] | None = None,
        null_value: float = -4,
        ln_SN_coeff: list[float] | None = None,
        free_param_names: list[str] | None = None,
    ) -> None:
        """Initialize the supernova feedback model."""
        if fid_value is None:
            fid_value = [0, -4]
        self.z_0 = z_0
        self.null_value = null_value

        if ln_SN_coeff:
            if free_param_names is not None:
                raise ValueError("can not specify coeff and free_param_names")
            self.ln_SN_coeff = ln_SN_coeff
        else:
            if free_param_names:
                # figure out number of SN free params
                n_SN = len([p for p in free_param_names if "ln_SN_" in p])
                if n_SN == 0:
                    n_SN = 1
            else:
                n_SN = 1

            self.ln_SN_coeff = [0.0] * n_SN
            self.ln_SN_coeff[-1] = fid_value[-1]
            if n_SN == 2:
                self.ln_SN_coeff[-2] = fid_value[-2]

        self.set_parameters()

    def set_parameters(self) -> None:
        """Create likelihood parameters for the SN amplitude."""
        self.params = []
        Npar = len(self.ln_SN_coeff)
        for i in range(Npar):
            name = "ln_SN_" + str(i)
            if i == 0:
                xmin = -5
                xmax = 2
            else:
                # not optimized
                xmin = -1
                xmax = 1
            # note non-trivial order in coefficients
            value = self.ln_SN_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

    def get_Nparam(self) -> int:
        """Return the number of free SN parameters.

        Returns
        -------
        int
            Number of free SN parameters.
        """
        assert len(self.ln_SN_coeff) == len(self.params), "size mismatch"
        return len(self.ln_SN_coeff)

    def get_SN_damp(
        self,
        z: float,
        like_params: list | None = None,
        name_par: str = "ln_SN",
    ) -> float:
        """Evaluate the SN correction amplitude at redshift ``z``.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name prefix. Default is "ln_SN".

        Returns
        -------
        float
            SN damping amplitude.
        """
        ln_SN_coeff = self.get_SN_coeffs(like_params=like_params)
        if ln_SN_coeff[-1] <= self.null_value:
            return 0
        else:
            xz = np.log((1 + z) / (1 + self.z_0))
            ln_poly = np.poly1d(ln_SN_coeff)
            ln_out = ln_poly(xz)
            return float(np.exp(ln_out))

    def get_contamination(
        self,
        z: float,
        k_Mpc: npt.NDArray[np.float64],
        like_params: list | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the multiplicative SN correction at ``z`` and ``k_Mpc``.

        Parameters
        ----------
        z : float
            Redshift.
        k_Mpc : npt.NDArray[np.float64]
            Wavenumber in 1/Mpc.
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        npt.NDArray[np.float64]
            SN correction.
        """
        SN_damp = self.get_SN_damp(z, like_params=like_params)
        if SN_damp == 0:
            return np.ones_like(k_Mpc)
        else:
            # Viel+13 Fig 8 panel b

            k0_Mpc = 0.07  # Mpc^-1
            k1_Mpc = 1.4  # Mpc^-1
            # Supernovae SN
            tmpLowk = [-0.06, -0.04, -0.02]
            tmpHighk = [-0.01, -0.01, -0.01]
            if z < 2.5:
                d0 = tmpLowk[0]
                d1 = tmpHighk[0]
            elif z < 3.5:
                d0 = tmpLowk[1]
                d1 = tmpHighk[1]
            else:
                d0 = tmpLowk[2]
                d1 = tmpHighk[2]
            delta = d0 + (d1 - d0) * (k_Mpc - k0_Mpc) / (k1_Mpc - k0_Mpc)
            corSN = 1 / (1.0 + delta * SN_damp)

            return corSN

    def get_parameters(self) -> list[likelihood_parameter.LikelihoodParameter]:
        """Return the SN likelihood parameters.

        Returns
        -------
        list[likelihood_parameter.LikelihoodParameter]
            List of SN likelihood parameters.
        """
        return self.params

    def get_SN_coeffs(
        self, like_params: list | None = None
    ) -> list[float] | npt.NDArray[np.float64]:
        """Return SN coefficients, updated from likelihood parameters.

        Parameters
        ----------
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        list[float] | npt.NDArray[np.float64]
            SN coefficients.
        """
        if like_params:
            ln_SN_coeff = list(self.ln_SN_coeff)
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_SN" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names_np = np.array(array_names)
            array_values_np = np.array(array_values)

            # use fiducial value (no contamination)
            if Npar == 0:
                return self.ln_SN_coeff
            elif Npar != len(self.params):
                print(Npar, len(self.params))
                raise ValueError("number of params mismatch in get_SN_coeffs")

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names_np)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_SN_coeff[Npar - ip - 1] = array_values_np[_[0]]
        else:
            ln_SN_coeff = self.ln_SN_coeff

        return ln_SN_coeff

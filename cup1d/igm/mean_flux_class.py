"""Mean flux modeling for the IGM.

This module provides the MeanFlux class for modeling the mean
transmitted flux fraction in the intergalactic medium.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from cup1d.igm.base_igm import IGMModel

# Type aliases
Array1D = npt.NDArray[np.float64]


class MeanFlux(IGMModel):
    """Mean flux model for the IGM.

    Parameters
    ----------
    coeffs : dict[str, float] | None, optional
        Coefficient dictionary. Default is None.
    prop_coeffs : dict[str, Any] | None, optional
        Coefficient properties. Default is None.
    free_param_names : list[str] | None, optional
        List of free parameter names. Default is None.
    z_0 : float, optional
        Pivot redshift. Default is 3.0.
    fid_igm : dict[str, Array1D] | None, optional
        Fiducial IGM parameters. Default is None.
    fid_vals : dict[str, Array1D] | None, optional
        Fiducial values. Default is None.
    flat_priors : dict[str, list[list[float]]] | None, optional
        Flat prior bounds. Default is None.
    Gauss_priors : dict[str, list[float]] | None, optional
        Gaussian prior widths. Default is None.
    """

    def __init__(
        self,
        coeffs: dict[str, float] | None = None,
        prop_coeffs: dict[str, Any] | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        fid_igm: dict[str, Array1D] | None = None,
        fid_vals: dict[str, Array1D] | None = None,
        flat_priors: dict[str, list[list[float]]] | None = None,
        Gauss_priors: dict[str, list[float]] | None = None,
    ) -> None:
        """Initialize the mean flux model."""
        list_coeffs = ["tau_eff"]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "interp_spl"
                prop_coeffs[coeff + "_otype"] = "exp"

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-0.5, 0.5], [-0.2, 0.2]]

        if fid_vals is None:
            fid_vals = {}

        for coeff in list_coeffs:
            if coeff not in fid_vals:
                if prop_coeffs[coeff + "_ztype"] == "pivot":
                    fid_vals[coeff] = [0, 0]
                else:
                    fid_vals[coeff] = np.zeros(len(prop_coeffs[coeff + "_znodes"]))

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
            fid_igm=fid_igm,
        )

    def get_tau_eff(
        self,
        z: float,
        like_params: list | None = None,
        name_par: str = "tau_eff",
    ) -> float:
        """Effective optical depth at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name. Default is "tau_eff".

        Returns
        -------
        float
            Effective optical depth.
        """
        tau_eff = self.get_value(name_par, z, like_params=like_params)
        tau_eff *= self.fid_interp[name_par](z)
        return float(tau_eff)

    def get_mean_flux(self, z: float, like_params: list | None = None) -> float:
        """Mean transmitted flux fraction at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        float
            Mean flux fraction.
        """
        tau = self.get_tau_eff(z, like_params=like_params)
        return float(np.exp(-tau))

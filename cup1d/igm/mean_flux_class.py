"""Mean flux modeling for the IGM.

This module provides the MeanFlux class for modeling the mean
transmitted flux fraction in the intergalactic medium.

"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt

from cup1d.igm.base_igm import IGM_model

# Type aliases
Array1D = npt.NDArray[np.float64]
Float = Union[float, int]


class MeanFlux(IGM_model):
    """Mean flux model for the IGM.

    Parameters
    ----------
    coeffs : Optional[Dict[str, float]], optional
        Coefficient dictionary.
    prop_coeffs : Optional[Dict[str, Any]], optional
        Coefficient properties.
    free_param_names : Optional[List[str]], optional
        List of free parameter names.
    z_0 : float, optional
        Pivot redshift.
    fid_igm : Optional[Dict[str, Array1D]], optional
        Fiducial IGM parameters.
    fid_vals : Optional[Dict[str, Array1D]], optional
        Fiducial values.
    flat_priors : Optional[Dict[str, Tuple[float, float]]], optional
        Flat prior bounds.
    Gauss_priors : Optional[Dict[str, float]], optional
        Gaussian prior widths.
    """

    def __init__(
        self,
        coeffs: dict[str, float] | None = None,
        prop_coeffs: dict[str, Any] | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        fid_igm: dict[str, Array1D] | None = None,
        fid_vals: dict[str, Array1D] | None = None,
        flat_priors: dict[str, tuple[float, float]] | None = None,
        Gauss_priors: dict[str, float] | None = None,
    ) -> None:
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
        like_params: list = None,
        name_par: str = "tau_eff",
    ) -> float:
        """Effective optical depth at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : List, optional
            Likelihood parameters.
        name_par : str, optional
            Parameter name.

        Returns
        -------
        float
            Effective optical depth.
        """
        tau_eff = self.get_value(name_par, z, like_params=like_params)
        tau_eff *= self.fid_interp[name_par](z)
        return tau_eff

    def get_mean_flux(self, z: float, like_params: list = None) -> float:
        """Mean transmitted flux fraction at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : List, optional
            Likelihood parameters.

        Returns
        -------
        float
            Mean flux fraction.
        """
        tau = self.get_tau_eff(z, like_params=like_params)
        return np.exp(-tau)

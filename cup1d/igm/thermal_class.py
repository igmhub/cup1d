"""Thermal modeling for the IGM.

This module provides the Thermal class for modeling temperature
and thermal broadening in the intergalactic medium.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from lace.cosmo import thermal_broadening

from cup1d.igm.base_igm import IGM_model

# Type aliases
Array1D = npt.NDArray[np.float64]
Float = float | int


class Thermal(IGM_model):
    """Thermal model for the IGM.

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
        list_coeffs = ["sigT_kms", "gamma"]

        if prop_coeffs is None:
            prop_coeffs = {}
            for coeff in list_coeffs:
                prop_coeffs[coeff + "_ztype"] = "interp_spl"
                prop_coeffs[coeff + "_otype"] = "const"

        if flat_priors is None:
            flat_priors = {}
            for coeff in list_coeffs:
                flat_priors[coeff] = [[-1, 1], [-1.25, 1.25]]

        for coeff in list_coeffs:
            if coeff not in fid_vals:
                if prop_coeffs[coeff + "_ztype"] == "pivot":
                    fid_vals[coeff] = [0, 1]
                else:
                    fid_vals[coeff] = np.ones(len(prop_coeffs[coeff + "_znodes"]))

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

    def get_sigT_kms(
        self,
        z: float,
        like_params: list = None,
        name_par: str = "sigT_kms",
    ) -> float:
        """sigT_kms at the input redshift.

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
            Thermal broadening in km/s.
        """
        sigT_kms = self.get_value(name_par, z, like_params=like_params)
        sigT_kms *= self.fid_interp[name_par](z)
        return sigT_kms

    def get_T0(
        self,
        z: float,
        like_params: list = None,
        name_par: str = "sigT_kms",
    ) -> float:
        """T_0 at the input redshift.

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
            Temperature in Kelvin.
        """
        sigT_kms = self.get_sigT_kms(z, like_params=like_params, name_par=name_par)
        T0 = thermal_broadening.T0_from_broadening_kms(sigT_kms)
        return T0

    def get_gamma(
        self,
        z: float,
        like_params: list = None,
        name_par: str = "gamma",
    ) -> float:
        """gamma at the input redshift.

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
            Thermal gamma parameter.
        """
        gamma = self.get_value(name_par, z, like_params=like_params)
        gamma *= self.fid_interp[name_par](z)
        return gamma

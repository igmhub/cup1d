"""Thermal modeling for the IGM.

This module provides the Thermal class for modeling temperature
and thermal broadening in the intergalactic medium.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from lace.cosmo import thermal_broadening

from cup1d.igm.base_igm import IGMModel

# Type aliases
Array1D = npt.NDArray[np.float64]


class Thermal(IGMModel):
    """Thermal model for the IGM.

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
        """Initialize the thermal model."""
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

        if fid_vals is None:
            fid_vals = {}

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
        like_params: list | None = None,
        name_par: str = "sigT_kms",
    ) -> float:
        """sigT_kms at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name. Default is "sigT_kms".

        Returns
        -------
        float
            Thermal broadening in km/s.
        """
        sigT_kms = self.get_value(name_par, z, like_params=like_params)
        sigT_kms *= self.fid_interp[name_par](z)
        return float(sigT_kms)

    def get_T0(
        self,
        z: float,
        like_params: list | None = None,
        name_par: str = "sigT_kms",
    ) -> float:
        """T_0 at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name. Default is "sigT_kms".

        Returns
        -------
        float
            Temperature in Kelvin.
        """
        sigT_kms = self.get_sigT_kms(z, like_params=like_params, name_par=name_par)
        T0 = thermal_broadening.T0_from_broadening_kms(sigT_kms)
        return float(T0)

    def get_gamma(
        self,
        z: float,
        like_params: list | None = None,
        name_par: str = "gamma",
    ) -> float:
        """gamma at the input redshift.

        Parameters
        ----------
        z : float
            Redshift.
        like_params : list | None, optional
            Likelihood parameters. Default is None.
        name_par : str, optional
            Parameter name. Default is "gamma".

        Returns
        -------
        float
            Thermal gamma parameter.
        """
        gamma = self.get_value(name_par, z, like_params=like_params)
        gamma *= self.fid_interp[name_par](z)
        return float(gamma)

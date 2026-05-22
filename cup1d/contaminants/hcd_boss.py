"""High-column-density absorber model calibrated on BOSS measurements.

References
----------
.. [1] Walther et al. (2024) - DESI Y1 HCD modeling
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from cup1d.contaminants.base_contaminants import Contaminant

# Type aliases
Array1D = npt.NDArray[np.float64]


def fun_cont(damp: float, k: float | Array1D) -> float | Array1D:
    """Evaluate the Walther et al. (2024) HCD correction shape.

    Based on Walther+24, their equation is weird.

    Parameters
    ----------
    damp : float
        Damping amplitude.
    k : float | Array1D
        Wavenumber in s/km.

    Returns
    -------
    float | Array1D
        HCD correction shape.
    """
    return 1 + 1 / (1 - (1 / (15000 * k - 8.9))) * damp


class HCD_BOSS(Contaminant):
    """HCD contamination model based on Eq. 5.2 of Walther et al. (2024).

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
    fid_vals : dict[str, Array1D] | None, optional
        Fiducial values. Default is None.
    flat_priors : dict[str, list[list[float]]] | None, optional
        Flat prior bounds. Default is None.
    null_vals : dict[str, float] | None, optional
        Null values for baseline. Default is None.
    Gauss_priors : dict[str, list[float]] | None, optional
        Gaussian prior widths. Default is None.
    """

    def __init__(
        self,
        coeffs: dict[str, float] | None = None,
        prop_coeffs: dict[str, Any] | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        fid_vals: dict[str, Array1D] | None = None,
        flat_priors: dict[str, list[list[float]]] | None = None,
        null_vals: dict[str, float] | None = None,
        Gauss_priors: dict[str, list[float]] | None = None,
    ) -> None:
        """Build the BOSS HCD correction model."""
        # list of all coefficients
        list_coeffs = [
            "HCD_damp1",
        ]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {
                "HCD_damp1": [[-0.5, 0.5], [-10.0, -1.0]],
            }

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "HCD_damp1_ztype": "pivot",
                "HCD_damp1_otype": "exp",
            }

        # fiducial values
        if fid_vals is None:
            fid_vals = {
                "HCD_damp1": np.array([0, -20.0]),
            }

        # null values
        if null_vals is None:
            null_vals = {
                "HCD_damp1": -21.5,
            }

        super().__init__(
            coeffs=coeffs,
            list_coeffs=list_coeffs,
            prop_coeffs=prop_coeffs,
            free_param_names=free_param_names,
            z_0=z_0,
            fid_vals=fid_vals,
            null_vals=null_vals,
            flat_priors=flat_priors,
            Gauss_priors=Gauss_priors,
        )

    def get_contamination(
        self,
        z: Array1D | float,
        k_kms: list[Array1D] | Array1D,
        like_params: list | None = None,
    ) -> list[Array1D] | Array1D:
        """Return the multiplicative HCD correction for each redshift bin.

        Parameters
        ----------
        z : Array1D | float
            Redshift(s).
        k_kms : list[Array1D] | Array1D
            Wavenumber(s) in s/km.
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        list[Array1D] | Array1D
            HCD correction.
        """
        z_array = np.atleast_1d(z)
        if isinstance(k_kms, np.ndarray) and k_kms.ndim == 1:
            k_kms_list = [k_kms] * len(z_array)
        else:
            k_kms_list = list(k_kms)

        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z_array, like_params=like_params)
            )
            if self.null_vals is not None and key in self.null_vals:
                if self.prop_coeffs[key + "_otype"] == "const":
                    null = self.null_vals[key]
                else:
                    null = np.exp(self.null_vals[key])
                mask = vals[key] <= null
                vals[key][mask] = 0

        dla_corr = []
        for iz in range(len(z_array)):
            cont = fun_cont(vals["HCD_damp1"][iz], k_kms_list[iz])
            dla_corr.append(cont)

        if isinstance(z, (float, int)) or len(z_array) == 1:
            return dla_corr[0]

        return dla_corr

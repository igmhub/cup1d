"""High-column-density absorber model following Rogers et al. (2018).

References
----------
.. [1] Rogers et al. (2018) - HCD modeling
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from cup1d.contaminants.base_contaminants import Contaminant

# Type aliases
Array1D = npt.NDArray[np.float64]


def fun_damping(k_kms: float | Array1D, a: float, b: float) -> float | Array1D:
    """Evaluate one Rogers et al. damping template.

    Parameters
    ----------
    k_kms : float | Array1D
        Wavenumber in s/km.
    a : float
        Template parameter a.
    b : float
        Template parameter b.

    Returns
    -------
    float | Array1D
        Damping template value.
    """
    return 1 / (a * np.exp(k_kms * b) - 1) ** 2


class HCD_Model_Rogers(Contaminant):
    """HCD contamination model with four Rogers et al. damping templates.

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

    Attributes
    ----------
    a_0 : Array1D
        Base template parameter a.
    a_1 : Array1D
        Evolution parameter for a.
    b_0 : Array1D
        Base template parameter b.
    b_1 : Array1D
        Evolution parameter for b.
    z_0_rogers : float
        Pivot redshift for Rogers templates.
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
        """Build the Rogers HCD correction model."""
        # list of all coefficients
        list_coeffs = [
            "HCD_damp1",
            "HCD_damp2",
            "HCD_damp3",
            "HCD_damp4",
            "HCD_const",
        ]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {
                "HCD_damp1": [[-0.5, 0.5], [-10.0, -1.0]],
                "HCD_damp2": [[-0.5, 0.5], [-10.0, -1.0]],
                "HCD_damp3": [[-0.5, 0.5], [-10.0, -1.0]],
                "HCD_damp4": [[-0.5, 0.5], [-10.0, -1.0]],
                "HCD_const": [[-1, 1], [-0.2, 1e-6]],
            }

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "HCD_damp1_ztype": "pivot",
                "HCD_damp2_ztype": "pivot",
                "HCD_damp3_ztype": "pivot",
                "HCD_damp4_ztype": "pivot",
                "HCD_const_ztype": "pivot",
                "HCD_damp1_otype": "exp",
                "HCD_damp2_otype": "exp",
                "HCD_damp3_otype": "exp",
                "HCD_damp4_otype": "exp",
                "HCD_const_otype": "const",
            }

        # fiducial values
        if fid_vals is None:
            fid_vals = {
                "HCD_damp1": np.array([0, -20.0]),
                "HCD_damp2": np.array([0, -20.0]),
                "HCD_damp3": np.array([0, -20.0]),
                "HCD_damp4": np.array([0, -20.0]),
                "HCD_const": np.array([0, 0]),
            }

        # null values
        if null_vals is None:
            null_vals = {
                "HCD_damp1": -21.5,
                "HCD_damp2": -21.5,
                "HCD_damp3": -21.5,
                "HCD_damp4": -21.5,
            }

        self.a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
        self.a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
        self.b_0 = np.array([36.449, 81.388, 162.95, 429.58])
        self.b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
        self.z_0_rogers = 2.0

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
            cont = 1 + vals["HCD_const"][iz] + np.zeros_like(k_kms_list[iz])
            for it in range(4):
                # compute the z-dependent correction terms
                a_z = (
                    self.a_0[it]
                    * ((1 + z_array[iz]) / (1 + self.z_0_rogers))
                    ** self.a_1[it]
                )
                b_z = (
                    self.b_0[it]
                    * ((1 + z_array[iz]) / (1 + self.z_0_rogers))
                    ** self.b_1[it]
                )
                cont += vals[f"HCD_damp{it+1}"][iz] * fun_damping(
                    k_kms_list[iz], a_z, b_z
                )
            dla_corr.append(cont)

        if isinstance(z, (float, int)) or len(z_array) == 1:
            return dla_corr[0]

        return dla_corr

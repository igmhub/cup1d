"""Spectral-resolution nuisance correction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cup1d.contaminants.base_contaminants import Contaminant


def get_Rz(z: float, k_kms: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Estimate the DESI resolution in km/s from wavelength-dependent fits.

    Parameters
    ----------
    z : float
        Redshift.
    k_kms : npt.NDArray[np.float64]
        Wavenumber in s/km.

    Returns
    -------
    npt.NDArray[np.float64]
        Spectral resolution in km/s.

    References
    ----------
    .. [1] DESI Collaboration (2024) - DESI Y1 results
    """
    # fig 32 https://arxiv.org/abs/2205.10939
    # lambda_AA = np.arange([3523.626, 3993.217, 4413.652, 4752.203, 5019.740, 5243.594, 5522.035, 5767.681, 5996.975, 6226.294, 6471.940, 6783.036])
    # resolution = np.array([2012.821, 2272.247, 2513.575, 2694.570, 2857.466, 2996.229, 3177.225, 3364.253, 3521.116, 3659.879, 3846.908, 4124.434])
    # rfit = np.polyfit(lambda_AA, resolution, 2)
    # plt.plot(lambda_AA, np.poly1d(rfit)(lambda_AA))

    c_kms = 2.99792458e5
    lya_AA = 1215.67
    rfit = np.array([4.53087663e-05, 1.70716005e-01, 8.60679006e02])
    R_coeff_lambda = np.poly1d(rfit)
    kms2AA = lya_AA * (1 + z) / c_kms
    # lambda_kms = lambda_AA * AA2kms
    k_AA = k_kms / kms2AA
    lambda_AA = 2 * np.pi / k_AA

    Rz = c_kms / (2.355 * R_coeff_lambda(lambda_AA))

    return Rz


def get_Rz_Naim(z: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """Estimate the DESI resolution in km/s using the Naim et al. convention.

    Parameters
    ----------
    z : float or npt.NDArray[np.float64]
        Redshift(s).

    Returns
    -------
    float or npt.NDArray[np.float64]
        Spectral resolution in km/s.

    References
    ----------
    .. [1] Naim et al. (2023) - DESI Y1 results (https://arxiv.org/abs/2306.06316)
    """
    # 4.1 https://arxiv.org/abs/2306.06316
    c_kms = 2.99792458e5
    lya_AA = 1215.67  # angstroms
    Delta_lambda_AA = 0.8  # angstroms
    # kms2AA = lya_AA * (1 + z) / c_kms
    # k_A = k_kms / kms2AA
    Rz = c_kms * Delta_lambda_AA / (1 + z) / lya_AA
    return Rz


class Resolution(Contaminant):
    """Multiplicative correction for uncertainty in spectral resolution.

    The default model exposes a single pivot-evolving coefficient that
    rescales the quadratic ``k`` dependence induced by resolution errors.

    Parameters
    ----------
    coeffs : dict | None
        Coefficients for the resolution correction.
    prop_coeffs : dict | None
        Properties of the coefficients.
    free_param_names : list[str] | None
        Names of the free parameters.
    z_0 : float, optional
        Pivot redshift. Default is 3.0.
    z_max_res : float, optional
        Maximum redshift for resolution. Default is 10.
    fid_vals : dict | None
        Fiducial values for the coefficients.
    flat_priors : dict | None
        Flat priors for the coefficients.
    null_vals : dict | None
        Null values for the coefficients.
    Gauss_priors : dict | None
        Gaussian priors for the coefficients.

    Attributes
    ----------
    list_coeffs : list[str]
        List of coefficient names.
    """

    def __init__(
        self,
        coeffs: dict | None = None,
        prop_coeffs: dict | None = None,
        free_param_names: list[str] | None = None,
        z_0: float = 3.0,
        z_max_res: float = 10,
        fid_vals: dict | None = None,
        flat_priors: dict | None = None,
        null_vals: dict | None = None,
        Gauss_priors: dict | None = None,
    ):
        """Initialize the resolution correction model."""

        list_coeffs = ["R_coeff"]

        # priors for all coefficients
        if flat_priors is None:
            flat_priors = {"R_coeff": [[-0.5, 0.5], [-0.1, 0.1]]}

        # z dependence and output type of coefficients
        if prop_coeffs is None:
            prop_coeffs = {
                "R_coeff_ztype": "pivot",
                "R_coeff_otype": "const",
            }

        # fiducial values
        if (fid_vals is None) or (len(fid_vals["R_coeff"]) == 0):
            fid_vals = {
                "R_coeff": [0, 0],
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
        z: npt.NDArray[np.float64] | float,
        k_kms: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
        like_params: list | None = None,
    ) -> list[npt.NDArray[np.float64]] | npt.NDArray[np.float64]:
        """Return the multiplicative resolution correction for each redshift.

        Parameters
        ----------
        z : npt.NDArray[np.float64] or float
            Redshift(s).
        k_kms : list[npt.NDArray[np.float64]] or npt.NDArray[np.float64]
            Wavenumber(s) in s/km.
        like_params : list | None, optional
            Likelihood parameters. Default is None.

        Returns
        -------
        list[npt.NDArray[np.float64]] or npt.NDArray[np.float64]
            Resolution correction.
        """
        z_array = np.atleast_1d(z)
        vals = {}
        for key in self.list_coeffs:
            vals[key] = np.atleast_1d(
                self.get_value(key, z_array, like_params=like_params)
            )

        cont = []
        for iz in range(len(z_array)):
            res = (
                1
                + 2
                * vals["R_coeff"][iz]
                * get_Rz_Naim(z_array[iz]) ** 2
                * k_kms[iz] ** 2
            )
            cont.append(res)

        if isinstance(z, (float, int)) or len(z_array) == 1:
            return cont[0]

        return cont

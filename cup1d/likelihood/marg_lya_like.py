"""Gaussian marginalized Lyman-alpha constraints in star-parameter space."""

from __future__ import annotations

import numpy as np


def gaussian_chi2(
    neff: float | np.ndarray,
    DL2: float | np.ndarray,
    neff_val: float,
    DL2_val: float,
    neff_err: float,
    DL2_err: float,
    r: float,
) -> float | np.ndarray:
    """Compute Gaussian delta chi-square for correlated ``n_eff`` and ``DL2``.

    Parameters
    ----------
    neff : float or np.ndarray
        Effective slope.
    DL2 : float or np.ndarray
        Linear power amplitude.
    neff_val : float
        Central value for neff.
    DL2_val : float
        Central value for DL2.
    neff_err : float
        Error for neff.
    DL2_err : float
        Error for DL2.
    r : float
        Correlation coefficient.

    Returns
    -------
    float or np.ndarray
        Computed chi-square value(s).
    """
    chi2 = (
        (DL2 - DL2_val) ** 2 / DL2_err**2
        + (neff - neff_val) ** 2 / neff_err**2
        - 2 * r * (neff - neff_val) * (DL2 - DL2_val) / DL2_err / neff_err
    ) / (1 - r * r)
    return chi2


def gaussian_chi2_McDonald2005(
    neff: float | np.ndarray, DL2: float | np.ndarray
) -> dict:
    """Compute Gaussian Delta chi^2 using measurement from McDonald et al. (2005).

    Parameters
    ----------
    neff : float or np.ndarray
        Effective slope at kp = 0.009 s/km.
    DL2 : float or np.ndarray
        k^3 P(k) / (2 pi^2) at z=3.

    Returns
    -------
    dict
        Dictionary containing central values, errors, correlation, and chi2.

    References
    ----------
    .. [3] McDonald et al. (2006) - SDSS Lyman-alpha forest
    """
    # DL2 = k^3 P(k) / (2 pi^2) , at z=3
    DL2_val = 0.47
    DL2_err = 0.06
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.3
    neff_err = 0.055
    # correlation coefficient
    r = 0.6
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_Chabanier2019(
    neff: float | np.ndarray, DL2: float | np.ndarray
) -> dict:
    """Compute Gaussian Delta chi^2 using measurement from Chabanier et al. (2019).

    Actual values from Table I of Goldstein+23 (https://arxiv.org/abs/2303.00746).

    Parameters
    ----------
    neff : float or np.ndarray
        Effective slope at kp = 0.009 s/km.
    DL2 : float or np.ndarray
        k^3 P(k) / (2 pi^2) at z=3.

    Returns
    -------
    dict
        Dictionary containing central values, errors, correlation, and chi2.

    References
    ----------
    .. [1] Chabanier et al. (2019) - Lyman-alpha forest P1D constraints
    """
    # DL2 = k^3 P(k) / (2 pi^2), at z=3
    DL2_val = 0.310
    DL2_err = 0.020
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.340
    neff_err = 0.0060
    # correlation coefficient
    r = 0.512
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_PalanqueDelabrouille2015(
    neff: float | np.ndarray, DL2: float | np.ndarray
) -> dict:
    """Compute Gaussian Delta chi^2 using measurement from Palanque-Delabrouille+2015.

    Parameters
    ----------
    neff : float or np.ndarray
        Effective slope at kp = 0.009 s/km.
    DL2 : float or np.ndarray
        k^3 P(k) / (2 pi^2) at z=3.

    Returns
    -------
    dict
        Dictionary containing central values, errors, correlation, and chi2.
    """
    # DL2 = k^3 P(k) / (2 pi^2), at z=3
    DL2_val = 0.32
    DL2_err = 0.03
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val = -2.36
    neff_err = 0.01
    # correlation coefficient (no idea from where)
    r = 0.55
    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results


def gaussian_chi2_Walther2024(
    neff: float | np.ndarray, DL2: float | np.ndarray, ana_type: str = "priors"
) -> dict:
    """Compute Gaussian Delta chi^2 using measurement from Walther+2024 (Table 3).

    Parameters
    ----------
    neff : float or np.ndarray
        Effective slope at kp = 0.009 s/km.
    DL2 : float or np.ndarray
        k^3 P(k) / (2 pi^2) at z=3.
    ana_type : str, optional
        Analysis type ('priors' or 'standard'). Default is 'priors'.

    Returns
    -------
    dict
        Dictionary containing central values, errors, correlation, and chi2.
    """

    if ana_type == "priors":
        # DL2 = k^3 P(k) / (2 pi^2) , at z=3
        DL2_val = 0.388
        DL2_err = 0.045
        # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
        neff_val = -2.316
        neff_err = 0.014
        # correlation coefficient (Table 3 of Walther2024)
        r = 0.58
    elif ana_type == "standard":
        # DL2 = k^3 P(k) / (2 pi^2) , at z=3
        DL2_val = 0.380
        DL2_err = 0.039
        # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
        neff_val = -2.312
        neff_err = 0.012
        # correlation coefficient (Table 3 of Walther2024)
        r = 0.56
    else:
        raise ValueError("ana_type not found")

    results = {
        "Delta2_star": DL2_val,
        "Delta2_star_err": DL2_err,
        "n_star": neff_val,
        "n_star_err": neff_err,
        "r": r,
        "chi2": gaussian_chi2(
            neff, DL2, neff_val, DL2_val, neff_err, DL2_err, r
        ),
    }
    return results

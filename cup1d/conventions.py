"""Scientific naming and array contracts used by cup1d."""

from __future__ import annotations

from typing import Any

import numpy as np

LEGACY_UNIT_KEYS = {
    "k_Mpc": "k_iMpc",
    "k_kms": "k_ikms",
    "p1d_Mpc": "P1D_Mpc",
    "p3d_Mpc": "P3D_Mpc",
    "Pk_kms": "P1D_kms",
    "dkms_dMpc": "dkms_diMpc",
}


def canonicalize_unit_keys(values: dict[str, Any]) -> dict[str, Any]:
    """Add canonical unit-key aliases to a shallow mapping copy.

    Parameters
    ----------
    values : dict
        Mapping that may use legacy names such as ``k_kms`` or ``Pk_kms``.

    Returns
    -------
    dict
        Shallow copy retaining every original entry and adding canonical names
        only when the canonical key was absent. Referenced array objects are
        not copied.
    """
    result = dict(values)
    for old, new in LEGACY_UNIT_KEYS.items():
        if new not in result and old in result:
            result[new] = result[old]
    return result


def validate_p1d_contract(k_ikms: Any, P1D_kms: Any, cov_P1D_kms: Any) -> None:
    """Validate one redshift bin of the public velocity-space P1D contract.

    Parameters
    ----------
    k_ikms : array_like
        Finite, strictly positive one-dimensional wavenumber grid in s/km.
    P1D_kms : array_like
        P1D values in km/s with shape ``(Nk,)``.
    cov_P1D_kms : array_like
        P1D covariance in (km/s)^2 with shape ``(Nk, Nk)``.

    Raises
    ------
    ValueError
        If the k grid is not finite positive one-dimensional, the power has
        a different shape, or the covariance is not ``(Nk, Nk)``.
    """
    k = np.asarray(k_ikms)
    power = np.asarray(P1D_kms)
    covariance = np.asarray(cov_P1D_kms)
    if k.ndim != 1 or not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("k_ikms must be a finite, positive 1D array")
    if power.shape != k.shape:
        raise ValueError("P1D_kms must have the same shape as k_ikms")
    if covariance.shape != (len(k), len(k)):
        raise ValueError("cov_P1D_kms must have shape (Nk, Nk)")

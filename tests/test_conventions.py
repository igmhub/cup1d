import numpy as np
import pytest

from cup1d.conventions import canonicalize_unit_keys, validate_p1d_contract
from cup1d.p1ds.base_p1d_data import BaseDataP1D


def test_legacy_keys_are_canonicalized():
    values = canonicalize_unit_keys({"k_kms": 1, "Pk_kms": 2})
    assert values["k_ikms"] == 1
    assert values["P1D_kms"] == 2


def test_p1d_array_contract():
    validate_p1d_contract([0.001, 0.002], [1.0, 2.0], np.eye(2))
    with pytest.raises(ValueError, match="same shape"):
        validate_p1d_contract([0.001], [1.0, 2.0], np.eye(1))
    with pytest.raises(ValueError, match="Nk"):
        validate_p1d_contract([0.001, 0.002], [1.0, 2.0], np.eye(3))


def test_data_uses_canonical_names_with_shared_legacy_aliases():
    k_ikms = np.array([0.001, 0.002])
    P1D_kms = np.array([1.0, 2.0])
    covariance = np.eye(2)
    data = BaseDataP1D([3.0], [k_ikms], [P1D_kms], [covariance])
    assert data.k_kms is data.k_ikms
    assert data.Pk_kms is data.P1D_kms
    assert data.cov_Pk_kms is data.cov_P1D_kms
    assert data.get_Pk_iz(0) is data.get_P1D_iz(0)

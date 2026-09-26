import numpy as np
import pytest

from cup1d.conventions import canonicalize_unit_keys, validate_p1d_contract
from cup1d.p1ds.base_p1d_data import BaseDataP1D
from cup1d.inference.fitter import Fitter
from cup1d.likelihood import parameter
from cup1d.utils.rebinning import Rebinning
from cup1d.utils.blinding import apply_blinding, apply_unblinding
from cup1d.utils.various_dicts import get_blob_value, get_blob_values


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


def test_dense_blob_contract_and_legacy_compatibility():
    dense = np.arange(12.0).reshape(2, 6)
    names = ["Delta2_star", "n_star"]
    np.testing.assert_array_equal(get_blob_values(dense, names), dense[:, :2])
    np.testing.assert_array_equal(get_blob_value(dense, "H0"), dense[:, 5])

    legacy = np.zeros(2, dtype=[(name, float) for name in (
        "Delta2_star", "n_star", "alpha_star", "f_star", "g_star", "H0"
    )])
    legacy["Delta2_star"] = [1.0, 2.0]
    np.testing.assert_array_equal(
        get_blob_value(legacy, "Delta2_star"), [1.0, 2.0]
    )


def test_dense_blob_blinding_roundtrip():
    blobs = np.zeros((2, 3, 6))
    blind = {"Delta2_star": 0.1, "n_star": -0.02, "alpha_star": 0.003}
    apply_blinding(blind, blobs)
    expected = np.broadcast_to([0.1, -0.02, 0.003], blobs[..., :3].shape)
    np.testing.assert_allclose(blobs[..., :3], expected)
    apply_unblinding(blind, blobs)
    np.testing.assert_allclose(blobs, 0.0, atol=1.0e-16)


def test_dense_blob_axis_is_preserved_when_chain_is_collapsed():
    fitter = object.__new__(Fitter)
    fitter.explore = True
    fitter.lnprob = np.zeros((2, 3))
    fitter.chain = np.zeros((2, 3, 4))
    fitter.blobs = np.zeros((2, 3, 6))

    chain, log_probability, blobs = fitter.get_chain(collapse=True)
    assert chain.shape == (6, 4)
    assert log_probability.shape == (6,)
    assert blobs.shape == (6, 6)


def test_values_from_cube_batch_is_columnar_and_validated():
    parameters = {
        "a": {"min_value": 2.0, "max_value": 4.0},
        "b": {"min_value": -1.0, "max_value": 3.0},
    }
    values = parameter.values_from_cube_batch(
        parameters, np.array([[0.0, 0.25], [1.0, 0.75]])
    )
    np.testing.assert_allclose(values["a"], [2.0, 4.0])
    np.testing.assert_allclose(values["b"], [0.0, 2.0])
    with pytest.raises(ValueError, match="n_batch, 2"):
        parameter.values_from_cube_batch(parameters, np.zeros(2))


def test_rebinning_batch_matches_scalar_for_ragged_k_grids():
    rebin = object.__new__(Rebinning)
    rebin.zs = {"data": np.array([3.0, 4.0])}
    rebin.cover = {
        "data": [
            np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
            np.array([[0.25, 0.75]]),
        ]
    }
    rebin.sum_cover = {"data": [np.array([1.0, 1.0]), np.array([1.0])]}
    scalar = [
        [np.array([1.0, 2.0, 4.0]), np.array([3.0, 3.0])],
        [np.array([4.0, 5.0, 7.0]), np.array([6.0, 6.0])],
    ]
    batched_input = [
        np.array([[1.0, 2.0, 4.0], [4.0, 5.0, 7.0]]),
        np.array([[3.0, 3.0], [6.0, 6.0]]),
    ]
    batched = rebin.rebinning_batch("data", batched_input)
    expected = [
        np.asarray([rebin.rebinning("data", [row[0], row[1]])[0] for row in scalar]),
        np.asarray([rebin.rebinning("data", [row[0], row[1]])[1] for row in scalar]),
    ]
    for result, reference in zip(batched, expected):
        np.testing.assert_allclose(result, reference)

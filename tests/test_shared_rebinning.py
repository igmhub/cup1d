"""One fine k grid shared by data sets at the same redshifts."""
import types

import numpy as np
import pytest

from cup1d.utils.rebinning import Rebinning, shared_fine_grid


def dataset(z, edges):
    """Data stub with (possibly overlapping) bins given as (lower, upper) per redshift."""
    lower = [np.asarray(e[0], dtype=float) for e in edges]
    upper = [np.asarray(e[1], dtype=float) for e in edges]
    return types.SimpleNamespace(z=np.asarray(z, dtype=float), k_kms=[(a + b) / 2 for a, b in zip(lower, upper)],
                                 k_kms_min=lower, k_kms_max=upper)


DESI_EDGES = np.linspace(0.001, 0.027, 53)
DESI = dataset([2.2, 2.4], [(DESI_EDGES[:-1], DESI_EDGES[1:])] * 2)
# log-spaced, slightly overlapping bins, as Karacayli2022
CENTRES = np.geomspace(0.0055, 0.063, 11)
HIRES = dataset([2.2, 2.4], [(CENTRES / 1.2, CENTRES * 1.1)] * 2)


def test_grid_extends_primary_to_next_bin_then_separate_spacing():
    k, lower, upper = shared_fine_grid([DESI, HIRES], 0, 32)
    separate = np.linspace(DESI_EDGES[0] * 0.95, DESI_EDGES[-1] * 1.05, 52 * 32)
    np.testing.assert_array_equal(k[:len(separate)], separate)  # DESI points unchanged
    step = separate[1] - separate[0]
    boundary = HIRES.k_kms_min[0][HIRES.k_kms_min[0] >= separate[-1] + step / 2].min()
    assert np.any(np.isclose(upper, boundary, rtol=0, atol=1e-15))  # a cell edge at the next hires bin
    below = upper <= boundary + 1e-15
    assert np.allclose(np.diff(lower[below])[:-1], step)  # primary spacing up to it
    assert upper[-1] == pytest.approx(HIRES.k_kms_max[0][-1] * 1.05)
    np.testing.assert_allclose(lower[1:], upper[:-1], rtol=0, atol=1e-15)  # contiguous
    assert np.all(np.diff(k) > 0)


def test_rebinning_shares_grid_and_averages_exactly():
    rebin = Rebinning({"desi": DESI, "hires": HIRES}, k_rebin_factor=32, n_k_emu=64)
    assert rebin.shared and rebin.groups == [["desi", "hires"]]
    assert rebin.k_kms["desi"][0] is rebin.k_kms["hires"][0]
    np.testing.assert_array_equal(rebin.k_kms_emu["desi"][1], rebin.k_kms_emu["hires"][1])
    separate = Rebinning({"desi": DESI, "hires": HIRES}, k_rebin_factor=32, shared_k_grid=False)
    smooth = lambda k: 2.0 + 3.0 * k + 50.0 * k**2
    # the primary data set is averaged exactly as on its own grid
    np.testing.assert_allclose(rebin.rebinning("desi", [smooth(k) for k in rebin.k_kms["desi"]])[0],
                               separate.rebinning("desi", [smooth(k) for k in separate.k_kms["desi"]])[0],
                               rtol=1e-12)
    # the other one is averaged by overlap length, also across cells of
    # different widths (cells cut by bin edges contribute their centre value)
    lower, upper = HIRES.k_kms_min[0], HIRES.k_kms_max[0]
    exact = 2.0 + 1.5 * (upper + lower) + 50.0 * (upper**2 + upper * lower + lower**2) / 3
    np.testing.assert_allclose(rebin.rebinning("hires", [smooth(k) for k in rebin.k_kms["hires"]])[0],
                               exact, rtol=1e-5)

def test_separate_grids_when_requested_or_redshifts_differ():
    separate = Rebinning({"desi": DESI, "hires": HIRES}, k_rebin_factor=32, shared_k_grid=False)
    assert not separate.shared and separate.groups == [["desi"], ["hires"]]
    other_z = dataset([2.2, 2.6], [(CENTRES / 1.2, CENTRES * 1.1)] * 2)
    assert not Rebinning({"desi": DESI, "hires": other_z}, k_rebin_factor=32).shared
    assert not Rebinning({"desi": DESI, "hires": HIRES}, k_rebin_factor=1).shared

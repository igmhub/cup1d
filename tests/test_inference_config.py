import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cup1d.configuration.args import Args
from cup1d.inference.fitter import Fitter
from cup1d.theory.theory import interpolate_p1d

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_interpolate_p1d_returns_nan_only_for_non_finite_rows():
    coarse = np.geomspace(0.002, 0.03, 16)
    fine = np.geomspace(0.002, 0.03, 40)
    good = 50 * (1 + (coarse / 0.01) ** 2) ** -0.6
    bad = good.copy()
    bad[3] = np.inf
    out = interpolate_p1d(coarse, np.stack([good, bad, -good]), fine)
    assert out.shape == (3, fine.size)
    assert np.all(np.isfinite(out[0])) and np.all(np.isfinite(out[2]))
    assert np.all(np.isnan(out[1]))
    np.testing.assert_allclose(out[2], -out[0], rtol=1e-4)


def test_gauss_newton_hessian_scales_exponential_parameters_to_the_cube():
    # p is exponential (a = exp(x)), q is linear; model = (a, x_q), unit icov
    free_params = {
        "p": {"min_value": -10.0, "max_value": -0.03, "hessian_transform": "exp"},
        "q": {"min_value": 0.0, "max_value": 2.0},
    }
    widths = np.array([9.97, 2.0])

    def physical(cube):
        return np.array([-10.0, 0.0]) + np.asarray(cube) * widths

    def prediction(cube, zmask=None):
        x = physical(cube)
        return np.array([np.exp(x[0]), x[1]]), np.eye(2)

    fitter = object.__new__(Fitter)
    fitter.like = SimpleNamespace(free_params=free_params, Gauss_priors=None)
    fitter.ndim = 2
    fitter._prediction_vector_and_icov = prediction
    for cube_p in (0.1, 0.5, 0.9):
        fitter.mle_cube = np.array([cube_p, 0.4])
        amplitude = np.exp(physical(fitter.mle_cube)[0])
        expected = np.diag([(amplitude * widths[0]) ** 2, widths[1] ** 2])
        hessian = fitter._gauss_newton_hessian(1e-4)
        np.testing.assert_allclose(hessian, expected, rtol=1e-4, atol=1e-12)


def test_initial_walkers_are_not_stacked_at_the_cube_edge():
    fitter = object.__new__(Fitter)
    fitter.ndim, fitter.nwalkers = 2, 50
    fitter.rng = np.random.default_rng(0)
    fitter.print = lambda *args, **kwargs: None
    p0 = fitter.get_initial_walkers(pini=np.array([0.999, 0.5]), rms=0.01)
    assert np.all((p0 > 0) & (p0 < 1))
    assert np.unique(p0[:, 0]).size == fitter.nwalkers


def test_per_redshift_cov_factor_is_kept_and_validated():
    values = np.linspace(1.0, 2.0, 11)
    args = Args(cov_factor={"val_stat": values.tolist()})
    np.testing.assert_allclose(args.cov_factor["val_stat"], values)
    assert len(args.cov_factor["z"]) == 11
    np.testing.assert_allclose(Args(cov_factor={"val_stat": 1.0}).cov_factor["val_stat"], 1.0)
    with pytest.raises(ValueError, match="val_stat"):
        Args(cov_factor={"val_stat": [1.0, 2.0, 3.0]})


def test_unknown_nested_option_warns_with_suggestion():
    with pytest.warns(UserWarning, match="cov_factor.val_stats.*val_stat"):
        Args(cov_factor={"val_stats": 1.0})
    with pytest.warns(UserWarning, match="mcmc.n_step"):
        Args(mcmc={"n_step": 5})


@pytest.mark.parametrize(
    "config",
    [
        "cm2026/cm2026_base.yaml",
        "cm2026/variations/HCD0.yaml",
        "cm2026/variations/Metals_Ma2025.yaml",
        "desi_dr1_tests/cm2026_base_forestmpggp_no_inflate.yaml",
    ],
)
def test_known_configurations_do_not_warn_about_unknown_options(config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Args.from_yaml(CONFIGS / config, verbose=False)
    assert not [w for w in caught if "Unknown option" in str(w.message)]


def test_emulator_weights_interpolate_between_bracketing_redshifts():
    from cup1d.likelihood.likelihood import emulator_interpolation_weights

    emu_z = np.repeat([2.0, 2.25, 2.5, 2.75], 3)
    emu_k = np.tile([0.1, 1.0, 3.0], 4)
    weights = emulator_interpolation_weights(
        [2.4, 2.6, 2.25, 1.8, 3.0], [1.1, 1.1, 0.12, 3.0, 0.1], emu_z, emu_k
    )
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    # z=2.4: 40% at 2.25 and 60% at 2.5, nearest k = 1.0
    np.testing.assert_allclose(weights[0, [4, 7]], [0.4, 0.6])
    # z=2.6: 60% at 2.5 and 40% at 2.75, so 2.4 and 2.6 no longer coincide
    np.testing.assert_allclose(weights[1, [7, 10]], [0.6, 0.4])
    assert not np.allclose(weights[0], weights[1])
    # on a grid redshift, or outside the grid, the nearest snapshot only
    assert weights[2, 3] == 1.0
    assert weights[3, 2] == 1.0 and weights[4, 9] == 1.0
    # the projected covariance stays positive semidefinite
    rng = np.random.default_rng(1)
    factor = rng.normal(size=(emu_z.size, emu_z.size))
    projected = weights @ (factor @ factor.T) @ weights.T
    assert np.linalg.eigvalsh(projected).min() > -1e-10


def test_null_out_folder_is_derived_below_path_out(tmp_path):
    args = Args(out_folder=None, path_out=str(tmp_path), emulator_label="forest_mpg_gp")
    assert Path(args.out_folder) == tmp_path / "DESIY1_QMLE3" / "global_opt" / "forest_mpg_gp"


def test_null_out_folder_uses_the_variation_name(tmp_path):
    args = Args(
        out_folder=None,
        path_out=str(tmp_path),
        emulator_label="forest_mpg_gp",
        name_variation="no_inflate",
    )
    assert Path(args.out_folder) == tmp_path / "DESIY1_QMLE3" / "no_inflate" / "forest_mpg_gp"


def test_ic_filenames_include_the_variation_only_when_given():
    from cup1d.utils.ic_files import get_ic_filename

    assert (
        get_ic_filename(["DESIY1_QMLE3"], "forest_mpg_gp", "global")
        == "DESIY1_QMLE3_forest_mpg_gp_ic_global.npy"
    )
    assert (
        get_ic_filename("DESIY1_QMLE3", "forest_mpg_gp", "at_a_time", "fixcosmo")
        == "DESIY1_QMLE3_forest_mpg_gp_fixcosmo_ic_at_a_time.npy"
    )
    assert get_ic_filename("A", "e", "global", "None") == "A_e_ic_global.npy"
    with pytest.raises(ValueError):
        get_ic_filename("A", "e", "other")


def test_ic_lookup_falls_back_from_variation_to_baseline_to_file_ic(tmp_path):
    from cup1d.utils.ic_files import ic_candidates, resolve_ic_file

    args = SimpleNamespace(
        data_label=["DESIY1_QMLE3"],
        emulator_label="forest_mpg_gp",
        name_variation="fixcosmo",
        path_ic=str(tmp_path),
        file_ic=str(tmp_path / "DESIY1_QMLE3_lace_mpg_ic_global.npy"),
    )
    variation = tmp_path / "DESIY1_QMLE3_forest_mpg_gp_fixcosmo_ic_global.npy"
    baseline = tmp_path / "DESIY1_QMLE3_forest_mpg_gp_ic_global.npy"
    configured = tmp_path / "DESIY1_QMLE3_lace_mpg_ic_global.npy"
    assert ic_candidates(args) == [variation, baseline, configured]
    assert resolve_ic_file(args) is None
    configured.touch()
    assert resolve_ic_file(args) == configured
    baseline.touch()
    assert resolve_ic_file(args) == baseline
    variation.touch()
    assert resolve_ic_file(args) == variation
    args.name_variation = None
    assert ic_candidates(args) == [baseline, configured]
    args.file_ic = None
    assert ic_candidates(args) == []


"""Regression test for the baseline DR1 tutorial analysis."""

from pathlib import Path

import numpy as np
import pytest

from cup1d import Analysis, Args
from lace.configuration import get_data_path


# Reference after adopting the LaCE cosmology interface in ``Theory``.
EXPECTED_CHI_SQUARED = np.float64(655.2820094791965)


@pytest.mark.external_model
def test_dr1_baseline_chi_squared(tmp_path):
    """The initial DR1 likelihood must reproduce the reference chi-squared."""

    model_directory = get_data_path() / "GPmodels" / "CH24_mpgcen_gpr"
    if not model_directory.is_dir():
        pytest.skip(
            "DR1 baseline regression requires the external LaCE model bundle at "
            f"{model_directory}"
        )

    arguments = Args.from_baseline(verbose=False)
    analysis = Analysis(arguments, out_folder=str(tmp_path))
    initial_point = analysis.fitter.sampling_point_from_parameters().copy()

    assert isinstance(analysis.like.free_params, dict)
    assert list(analysis.like.free_params) == analysis.like.free_param_names
    assert all(
        isinstance(parameter, dict)
        for parameter in analysis.like.free_params.values()
    )
    first_name = analysis.like.free_param_names[0]
    assert set(analysis.like.free_params[first_name]) == {
        "name", "value", "min_value", "max_value",
        "Gauss_priors_width", "fixed", "hessian_transform",
    }
    physical = analysis.fitter.parameters_from_sampling_point(initial_point)
    assert isinstance(physical, dict)
    assert list(physical) == analysis.like.free_param_names
    np.testing.assert_allclose(
        analysis.fitter.sampling_point_from_parameters(physical), initial_point
    )

    assert not hasattr(analysis.like, "sampling_point_from_parameters")
    chi_squared = analysis.fitter.get_chi2(initial_point)
    np.testing.assert_allclose(analysis.like.get_chi2(physical), chi_squared)

    assert isinstance(chi_squared, np.float64)
    np.testing.assert_allclose(chi_squared, EXPECTED_CHI_SQUARED, rtol=1.0e-6)

    # Result files contain only the YAML reference, fit state, and sampler paths.
    analysis.fitter.set_mle(initial_point, chi_squared, force=True)
    minimizer_path = analysis.fitter.save_minimizer_results()
    minimizer_payload = np.load(minimizer_path, allow_pickle=True).item()
    assert minimizer_path.name == "minimizer_results.npy"
    assert set(minimizer_payload) == {
        "format_version", "result_type", "config_path", "config_loader",
        "synthetic", "parameter_names", "fit",
    }
    assert {
        "mle_cube", "mle_chi2", "mle_cosmo_errors",
        "mle_cosmo_covariance", "mle_cosmo_correlation",
    } <= set(minimizer_payload["fit"])
    assert minimizer_payload["fit"]["mle_error_method"] == "gauss_newton"

    analysis.fitter.chain = np.zeros((2, 2, analysis.fitter.ndim))
    analysis.fitter.lnprob = np.zeros((2, 2))
    analysis.fitter.blobs = np.zeros((2, 2), dtype=analysis.fitter.blobs_dtype)
    sampler_path = analysis.fitter.save_sampler_results()
    sampler_payload = np.load(sampler_path, allow_pickle=True).item()
    assert sampler_path.name == "sampler_results.npy"
    for key in ("chain_path", "blobs_path", "lnprob_path"):
        assert key in sampler_payload
        assert Path(sampler_payload[key]).exists()

    directories_before_load = {
        path.name for path in tmp_path.iterdir() if path.is_dir()
    }
    restored = Analysis.from_results(minimizer_path)
    directories_after_load = {
        path.name for path in tmp_path.iterdir() if path.is_dir()
    }
    assert directories_after_load == directories_before_load
    np.testing.assert_allclose(restored.fitter.mle_cube, initial_point)
    np.testing.assert_allclose(restored.fitter.mle_chi2, chi_squared)
    assert restored.fitter.mle == analysis.fitter.mle

    # Plotting must use the same model arrays and leave the fit unchanged.
    import matplotlib.pyplot as plt

    prediction = analysis.like.get_p1d_kms(physical)[0]
    primary = next(iter(analysis.like.data.values()))
    selected_z = primary.z[1:3]
    output = analysis.like.plot_p1d(
        physical, residuals=True, plot_panels=True,
        zmask=selected_z, return_all=True, show=False,
    )
    for key, plotted in output.items():
        for z, model in zip(plotted['zs'], plotted['p1d_model']):
            index = np.flatnonzero(np.isclose(analysis.like.data[key].z, z))[0]
            np.testing.assert_allclose(model, np.asarray(prediction[key][index]).reshape(-1))
    np.testing.assert_allclose(analysis.like.get_chi2(physical), chi_squared, rtol=1e-12)
    plt.close('all')

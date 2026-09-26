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

    assert analysis.fitter.blobs_dtype is float
    assert analysis.fitter.blob_names == [
        "Delta2_star", "n_star", "alpha_star", "f_star", "g_star", "H0"
    ]

    batch_points = np.vstack(
        [initial_point, initial_point + 1.0e-4, initial_point - 1.0e-4]
    )
    batch_results = analysis.fitter.log_prob_and_blobs_batch(batch_points)
    scalar_results = [
        analysis.fitter.log_prob_and_blobs(point) for point in batch_points
    ]
    for batched, scalar in zip(batch_results, scalar_results):
        np.testing.assert_allclose(batched[0], scalar[0], rtol=1.0e-12)
        assert batched[1:] == scalar[1:]

    invalid_point = initial_point.copy()
    invalid_point[0] = -0.1
    mixed_results = analysis.fitter.log_prob_and_blobs_batch(
        np.vstack([batch_points[0], invalid_point, batch_points[1]])
    )
    np.testing.assert_allclose(mixed_results[0][0], scalar_results[0][0])
    assert mixed_results[1][0] == analysis.like.min_log_like
    np.testing.assert_allclose(mixed_results[2][0], scalar_results[1][0])

    # Batch theory inputs retain explicit batch and redshift axes and agree
    # with the established scalar construction point by point.
    parameter_columns = analysis.fitter.parameters_from_sampling_points(batch_points)
    zs = next(iter(analysis.like.Rebin_data.zs.values()))
    emu_batch, M_batch, blobs_batch = analysis.like.theory.get_emulator_calls(
        zs, parameter_columns, return_M_of_z=True, return_blob=True
    )
    assert M_batch.shape == (len(batch_points), len(zs))
    assert blobs_batch.shape == (len(batch_points), 6)
    for index, point in enumerate(batch_points):
        scalar_parameters = analysis.fitter.parameters_from_sampling_point(point)
        emu_scalar, M_scalar, blob_scalar = analysis.like.theory.get_emulator_calls(
            zs, scalar_parameters, return_M_of_z=True, return_blob=True
        )
        np.testing.assert_allclose(M_batch[index], M_scalar)
        np.testing.assert_allclose(blobs_batch[index], blob_scalar)
        for name in emu_scalar:
            if name == "mF_fid":
                continue
            np.testing.assert_allclose(emu_batch[name][index], emu_scalar[name])

    k_kms = next(iter(analysis.like.Rebin_data.k_kms.values()))
    hcd_batch = analysis.like.theory.model_cont.hcd_model.get_contamination_batch(
        zs, k_kms, parameter_columns
    )
    resolution_batch = analysis.like.theory.model_syst.resolution_model.get_contamination_batch(
        zs, k_kms, parameter_columns
    )
    for index, point in enumerate(batch_points):
        scalar_parameters = analysis.fitter.parameters_from_sampling_point(point)
        hcd_scalar = analysis.like.theory.model_cont.hcd_model.get_contamination(
            zs, k_kms, scalar_parameters
        )
        resolution_scalar = analysis.like.theory.model_syst.resolution_model.get_contamination(
            zs, k_kms, scalar_parameters
        )
        for iz in range(len(zs)):
            np.testing.assert_allclose(hcd_batch[iz][index], hcd_scalar[iz])
            np.testing.assert_allclose(resolution_batch[iz][index], resolution_scalar[iz])

    si_add = analysis.like.theory.model_cont.metal_models["Si_add"]
    si_add_batch = si_add.get_contamination_batch(
        zs, k_kms, emu_batch["mF"], parameter_columns
    )
    for index, point in enumerate(batch_points):
        scalar_parameters = analysis.fitter.parameters_from_sampling_point(point)
        scalar_emu = analysis.like.theory.get_emulator_calls(
            zs, scalar_parameters, return_M_of_z=False
        )
        si_add_scalar = si_add.get_contamination(
            zs, k_kms, scalar_emu["mF"], scalar_parameters
        )
        for iz in range(len(zs)):
            np.testing.assert_allclose(si_add_batch[iz][index], si_add_scalar[iz])

    si_mult = analysis.like.theory.model_cont.metal_models["Si_mult"]
    si_mult_batch = si_mult.get_contamination_batch(
        zs, k_kms, emu_batch["mF"], parameter_columns
    )
    for index, point in enumerate(batch_points):
        scalar_parameters = analysis.fitter.parameters_from_sampling_point(point)
        scalar_emu = analysis.like.theory.get_emulator_calls(
            zs, scalar_parameters, return_M_of_z=False
        )
        si_mult_scalar = si_mult.get_contamination(
            zs, k_kms, scalar_emu["mF"], scalar_parameters
        )
        for iz in range(len(zs)):
            np.testing.assert_allclose(si_mult_batch[iz][index], si_mult_scalar[iz])

    theory_batch = analysis.like.theory.get_p1d_kms(zs, k_kms, parameter_columns)
    for index, point in enumerate(batch_points):
        scalar_parameters = analysis.fitter.parameters_from_sampling_point(point)
        theory_scalar = analysis.like.theory.get_p1d_kms(
            zs, k_kms, scalar_parameters, return_blob=False
        )
        for iz in range(len(zs)):
            np.testing.assert_allclose(theory_batch[iz][index], theory_scalar[iz])

    import emcee

    sampler = emcee.EnsembleSampler(
        len(batch_points),
        analysis.fitter.ndim,
        analysis.fitter.log_prob_and_blobs_batch,
        vectorize=True,
        blobs_dtype=analysis.fitter.blobs_dtype,
    )
    _, dense_blobs = sampler.compute_log_prob(batch_points)
    assert dense_blobs.shape == (len(batch_points), 6)
    assert dense_blobs.dtype == float

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
    analysis.fitter.blobs = np.zeros(
        (2, 2, len(analysis.fitter.blob_names)),
        dtype=analysis.fitter.blobs_dtype,
    )
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

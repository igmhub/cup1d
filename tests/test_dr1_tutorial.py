"""Regression test for the baseline DR1 tutorial analysis."""

import numpy as np

from cup1d import Analysis, Args


# Reference after adopting the LaCE cosmology interface in ``Theory``.
EXPECTED_CHI_SQUARED = np.float64(655.2820094791965)


def test_dr1_baseline_chi_squared(tmp_path):
    """The initial DR1 likelihood must reproduce the reference chi-squared."""

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

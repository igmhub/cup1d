"""Rendering regressions using small, deterministic likelihood fixtures."""

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import block_diag

from cup1d.likelihood.likelihood import Likelihood
from cup1d.postprocessing.p1d import P1DPlotter, plot_p1d, plot_p1d_errors


class SmallLikelihood:
    """Same prediction interface as Likelihood, with known spectra."""
    plot_p1d = Likelihood.plot_p1d
    old_plot_p1d = Likelihood.old_plot_p1d
    plot_p1d_errors = Likelihood.plot_p1d_errors

    def __init__(self, nz=4, datasets=1):
        self.data = {}
        self.cov_Pk_kms = {}
        self.full_cov_Pk_kms = {}
        self.free_params = [object()]
        self.rank = 1
        for j in range(datasets):
            key = f'data{j}'
            k = [np.linspace(.002, .02, 5) for _ in range(nz)]
            power = [np.full(5, 10. + i + j) for i in range(nz)]
            self.data[key] = SimpleNamespace(z=np.arange(nz) * .2 + 2,
                                            k_kms=k, Pk_kms=power)
            self.cov_Pk_kms[key] = [np.eye(5) * .25 for _ in range(nz)]
            self.full_cov_Pk_kms[key] = block_diag(*self.cov_Pk_kms[key])

    def get_p1d_kms(self, values=None):
        offset = .5 if values is None else np.asarray(values)[0]
        return ({key: [p - offset for p in data.Pk_kms] for key, data in self.data.items()}, {})

    def get_chi2(self, values=None, return_all=False, zmask=None):
        prediction = self.get_p1d_kms(values)[0]
        values_by_key = {}
        for key, data in self.data.items():
            values_by_key[key] = np.array([
                np.sum((p - prediction[key][i]) ** 2 / .25)
                if zmask is None or np.any(np.isclose(z, zmask)) else 0.
                for i, (z, p) in enumerate(zip(data.z, data.Pk_kms))
            ])
        total = sum(np.sum(v) for v in values_by_key.values())
        return (total, values_by_key) if return_all else total


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('residuals', [False, True])
@pytest.mark.parametrize('panels,nz,datasets', [(False, 4, 1), (True, 1, 1), (True, 5, 1), (True, 4, 2)])
def test_rendering_and_native_outputs(residuals, panels, nz, datasets):
    like = SmallLikelihood(nz, datasets)
    result = like.plot_p1d(residuals=residuals, plot_panels=panels, return_all=True, show=False)
    axes = [ax for ax in plt.gcf().axes if ax.get_visible()]
    assert len(axes) == (nz * datasets if panels else datasets)
    for j, (key, data) in enumerate(like.data.items()):
        np.testing.assert_allclose(result[key]['p1d_model'], np.asarray(data.Pk_kms) - .5)
        np.testing.assert_allclose(result[key]['p1d_err'], .5)
        np.testing.assert_allclose(result[key]['chi2'], 5.)
        ax = axes[j * nz] if panels else axes[j]
        plotted_data = ax.containers[0].lines[0].get_ydata()
        expected = data.Pk_kms[0] / (data.Pk_kms[0] - .5) if residuals else data.k_kms[0] * data.Pk_kms[0] / np.pi
        np.testing.assert_allclose(plotted_data, expected)


def test_mask_uses_original_prediction_indices_and_correct_dof():
    like = SmallLikelihood()
    result = like.plot_p1d(zmask=[2.4], residuals=True, plot_panels=True, return_all=True, show=False)
    np.testing.assert_allclose(result['data0']['p1d_model'][0], 11.5)
    assert result['data0']['zs'] == [2.4]
    assert 'n_\\mathrm{deg}=4' in plt.gcf()._suptitle.get_text()


def test_posterior_bands_and_realizations():
    like = SmallLikelihood(datasets=2)
    bins, _, _ = P1DPlotter(like).prepare(rand_posterior=[[.2], [.8]], n_perturb=3, zmask=[2.2, 2.6])
    for item in bins:
        np.testing.assert_allclose(item.model_error, .3)
        assert item.realizations.shape == (3, 5)
    plot_p1d(like, residuals=True, rand_posterior=[[.2], [.8]], n_perturb=2, show=False)
    assert all(len(ax.collections) >= 8 for ax in plt.gcf().axes)


def test_per_redshift_points():
    like = SmallLikelihood()
    output = plot_p1d(like, values=[[.2], [.7]], zmask=[2.2, 2.6],
                      z_at_time=True, return_all=True, show=False)
    np.testing.assert_allclose(output['data0']['p1d_model'][0], 10.8)
    np.testing.assert_allclose(output['data0']['p1d_model'][1], 12.3)


def test_store_data_save_and_row_limits(tmp_path):
    like = SmallLikelihood(nz=4)
    output = plot_p1d(like, residuals=True, plot_panels=True, store_data=True,
                      plot_fname=tmp_path / 'residuals', ylims=[[.5, 1.5], [.8, 1.2]])
    assert (tmp_path / 'residuals.pdf').stat().st_size > 0
    assert (tmp_path / 'residuals.png').stat().st_size > 0
    np.testing.assert_allclose(output['y0'], 10 / 9.5)
    for i, ax in enumerate(plt.gcf().axes[:4]):
        np.testing.assert_allclose(ax.get_ylim(), [.5, 1.5] if i < 3 else [.8, 1.2])


def test_store_data_multiple_datasets_does_not_overwrite():
    output = plot_p1d(SmallLikelihood(datasets=2), store_data=True, show=False)
    assert set(output) == {'data0', 'data1'}
    assert not np.array_equal(output['data0']['y0'], output['data1']['y0'])


def test_old_alias_and_error_histograms():
    like = SmallLikelihood(nz=1)
    with pytest.warns(DeprecationWarning, match='old_plot_p1d'):
        old = like.old_plot_p1d(return_all=True, show=False)
    new = like.plot_p1d(return_all=True, show=False)
    np.testing.assert_array_equal(old['data0']['p1d_model'], new['data0']['p1d_model'])
    errors = like.plot_p1d_errors(show=False)
    np.testing.assert_allclose(errors['(d-m)/err'][0], 1.)
    assert len(plt.gcf().axes) == 2


@pytest.mark.parametrize('options,error,match', [
    ({'return_covar': True}, NotImplementedError, 'rebinned likelihood'),
    ({'zmask': [8]}, ValueError, 'No data redshifts'),
    ({'plot_every_iz': 0}, ValueError, 'positive integer'),
    ({'rand_posterior': [.1, .2]}, ValueError, 'shape'),
    ({'z_at_time': True, 'values': [.2]}, ValueError, 'one sampling point'),
    ({'n_perturb': -1}, ValueError, 'nonnegative'),
])
def test_actionable_invalid_options(options, error, match):
    with pytest.raises(error, match=match):
        plot_p1d(SmallLikelihood(), show=False, **options)


def test_core_modules_do_not_import_rendering_libraries():
    root = Path(__file__).resolve().parents[1] / 'cup1d'
    for folder in ('emulator', 'inference', 'likelihood', 'models', 'p1ds', 'theory', 'utils'):
        for path in (root / folder).rglob('*.py'):
            if '.ipynb_checkpoints' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                assert not any(m.startswith(('matplotlib', 'seaborn', 'corner')) for m in modules), path


def test_model_and_simulation_compatibility_wrappers():
    from cup1d.models.contaminants.HCD.hcd_model_McDonald2005 import HCD_Model_McDonald2005
    from cup1d.p1ds.simulations.data_accel2 import Accel2_P1D
    from cup1d.utils.utils import get_discrete_cmap
    from cup1d.utils.fit_ellipse import plot_ellipse

    HCD_Model_McDonald2005().plot_contamination([3.], [np.linspace(.001, .02, 5)])
    assert plt.gca().lines
    assert get_discrete_cmap(3).N == 3
    fig, ax = plt.subplots()
    plot_ellipse(ax=ax)
    assert len(ax.patches) == 1
    simulation = object.__new__(Accel2_P1D)
    simulation.plot_p1d_z({'k1d_Mpc': np.array([.1, .2]), 'p1d_Mpc': np.ones((1, 2)), 'z': [3.]})
    assert plt.gca().lines


def test_completed_fit_plotter_calls_new_renderer(tmp_path):
    from cup1d.postprocessing import Plotter

    plotter = object.__new__(Plotter)
    plotter.fitter = SimpleNamespace(like=SmallLikelihood(nz=1))
    plotter.mle_values = np.array([.5])
    plotter.save_directory = str(tmp_path)
    plotter.plot_p1d(residuals=True, plot_panels=True)
    plotter.plot_p1d_errors(zmask=[2.])
    assert (tmp_path / 'P1D_mle_residuals_panels.png').exists()
    assert (tmp_path / 'P1D_mle_errors.pdf').exists()


def test_stride_and_disabled_realizations():
    like = SmallLikelihood(nz=5)
    result = plot_p1d(like, plot_every_iz=2, n_perturb=4,
                      plot_realizations=False, return_all=True, show=False)
    np.testing.assert_allclose(result['data0']['zs'], [2., 2.4, 2.8])
    # Three data curves and three model curves, no realization curves.
    assert len(plt.gca().lines) == 6


def test_mask_with_no_bins_in_second_dataset():
    like = SmallLikelihood(datasets=2)
    like.data['data1'].z = like.data['data1'].z + 3
    bins, _, _ = P1DPlotter(like).prepare(zmask=[2.], n_perturb=2)
    assert len(bins) == 1
    assert bins[0].key == 'data0'

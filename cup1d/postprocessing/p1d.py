"""Likelihood P1D plots: prepare once, then render spectra or residuals.

The likelihood owns predictions and chi-squared calculations. This module
only selects their outputs and converts them into figures.
"""

from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution, norm


@dataclass
class P1DBin:
    """One selected dataset/redshift bin in native velocity units."""

    key: str
    index: int
    z: float
    k: np.ndarray
    data: np.ndarray
    model: np.ndarray
    error: np.ndarray
    chi2: float
    dof: int
    color: object
    shift: float
    model_error: np.ndarray | None = None
    realizations: np.ndarray | None = None

    @property
    def probability(self):
        return chi2_distribution.sf(self.chi2, self.dof)


class P1DPlotter:
    """Prepare likelihood predictions independently of figure layout.

    ``values`` is a sampling point, or one point per selected redshift when
    ``z_at_time=True``. Redshifts are ordered as in the input datasets, with
    duplicates removed. ``rand_posterior`` is a two-dimensional array of
    sampling points used to estimate pointwise model uncertainty.
    """

    def __init__(self, likelihood):
        self.like = likelihood

    def _predict(self, values):
        result = self.like.get_p1d_kms(values=values)
        if result is None:
            raise ValueError("Cannot plot P1D: the sampling point is outside the model domain.")
        return result[0]

    def prepare(self, values=None, *, zmask=None, plot_every_iz=1,
                z_at_time=False, rand_posterior=None, n_perturb=0,
                collapse=False, plot_panels=False, glob_full=False,
                fix_cosmo=False, n_param_glob_full=16, chi2_nozcov=False):
        if not isinstance(plot_every_iz, (int, np.integer)) or plot_every_iz < 1:
            raise ValueError("plot_every_iz must be a positive integer.")
        if not isinstance(n_perturb, (int, np.integer)) or n_perturb < 0:
            raise ValueError("n_perturb must be a nonnegative integer.")
        mask = None if zmask is None else np.atleast_1d(zmask)
        selected = {
            key: [i for i, z in enumerate(data.z)
                  if mask is None or np.any(np.isclose(z, mask, atol=1e-3, rtol=0))]
            for key, data in self.like.data.items()
        }
        if not any(selected.values()):
            raise ValueError("No data redshifts match zmask.")
        points = {}
        if z_at_time:
            redshifts = list(dict.fromkeys(
                float(self.like.data[key].z[i]) for key, indices in selected.items() for i in indices
            ))
            array = np.asarray(values)
            if array.shape != (len(redshifts), len(self.like.free_params)):
                raise ValueError("z_at_time requires one sampling point per selected redshift.")
            points = dict(zip(redshifts, array))
            predictions = {z: self._predict(point) for z, point in points.items()}
            per_z = {z: self.like.get_chi2(values=point, return_all=True, zmask=np.array([z]))[1]
                     for z, point in points.items()}
            chi2_bins = {key: {i: per_z[float(self.like.data[key].z[i])][key][i] for i in indices}
                         for key, indices in selected.items()}
            total_chi2 = sum(sum(v.values()) for v in chi2_bins.values())
        else:
            prediction = self._predict(values)
            total_chi2, chi2_bins = self.like.get_chi2(values=values, return_all=True, zmask=mask)
            predictions = None
            if chi2_nozcov:
                total_chi2 = sum(chi2_bins[key][i] for key, indices in selected.items() for i in indices)

        posterior = None
        if rand_posterior is not None:
            samples = np.asarray(rand_posterior)
            if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != len(self.like.free_params):
                raise ValueError("rand_posterior must have shape (n_samples, n_free_parameters).")
            posterior = [self._predict(sample) for sample in samples]

        bins = []
        total_ndata = 0
        per_redshift_dof = 0
        n_free = len(self.like.free_params)
        for key, indices in selected.items():
            data = self.like.data[key]
            models = {i: np.asarray((predictions[float(data.z[i])] if z_at_time else prediction)[key][i]).reshape(-1)
                      for i in indices}
            noise = self._realizations(key, models, indices, n_perturb)
            for i in indices:
                ndata = np.count_nonzero(data.Pk_kms[i])
                total_ndata += ndata
                nparameters = np.count_nonzero(points[float(data.z[i])]) if z_at_time else n_free
                dof = ndata - nparameters
                per_redshift_dof += dof
                if glob_full:
                    dof = ndata - n_param_glob_full
                if i % plot_every_iz:
                    continue
                k = np.asarray(data.k_kms[i])
                model = models[i]
                if model.shape != k.shape:
                    raise ValueError(f"P1D prediction shape mismatch for dataset {key}, z={data.z[i]}.")
                model_error = None
                if posterior is not None:
                    model_error = np.std([np.asarray(p[key][i]).reshape(-1) for p in posterior], axis=0)
                nz = len(data.z)
                color = plt.cm.jet(i / (nz - 1)) if nz > 1 and not plot_panels else 'C0'
                shift = 4 * i / (nz - 1) if nz > 1 and not collapse and not plot_panels else 0
                bins.append(P1DBin(
                    key, i, float(data.z[i]), k, np.asarray(data.Pk_kms[i]), model,
                    np.sqrt(np.diag(self.like.cov_Pk_kms[key][i])),
                    float(chi2_bins[key][i]), dof, color, shift, model_error, noise.get(i),
                ))
        if not bins:
            raise ValueError("No redshift bins remain after applying plot_every_iz and zmask.")
        dof = per_redshift_dof if z_at_time else total_ndata - n_free - (2 if fix_cosmo else 0)
        return bins, float(total_chi2), dof

    def _realizations(self, key, models, indices, count):
        if not count or not indices:
            return {}
        data = self.like.data[key]
        lengths = [len(k) for k in data.k_kms]
        offsets = np.cumsum([0] + lengths)
        chosen = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])
        covariance = self.like.full_cov_Pk_kms[key][np.ix_(chosen, chosen)]
        draws = np.random.multivariate_normal(np.concatenate([models[i] for i in indices]), covariance, count)
        split = np.cumsum([len(models[i]) for i in indices])[:-1]
        return dict(zip(indices, np.split(draws, split, axis=1)))


def _axes(bins, panels):
    if panels:
        rows = (len(bins) + 2) // 3
        fig, axes = plt.subplots(rows, 3, squeeze=False, figsize=(12, rows * 2.5), sharex=True)
        axes = axes.ravel()
        for ax in axes[len(bins):]:
            ax.set_visible(False)
        return fig, list(axes[:len(bins)])
    keys = list(dict.fromkeys(b.key for b in bins))
    fig, axes = plt.subplots(len(keys), 1, squeeze=False, figsize=(16, 14), sharex=True)
    lookup = dict(zip(keys, axes.ravel()))
    return fig, [lookup[b.key] for b in bins]


def _bin_label(item, print_chi2):
    if print_chi2:
        return (rf"$\chi^2={item.chi2:.2f}$, $n_\mathrm{{deg}}={item.dof}$, "
                f"prob={100 * item.probability:.2f}%")
    return rf"$z={item.z:g}$, $\chi^2={item.chi2:.2f}$, $n_\mathrm{{data}}={np.count_nonzero(item.data)}$"


def plot_p1d_spectra(bins, *, panels=False, fontsize=20, print_chi2=True):
    """Render prepared bins as dimensionless spectra; return figure and axes."""
    fig, axes = _axes(bins, panels)
    for item, ax in zip(bins, axes):
        factor = item.k / np.pi
        ax.errorbar(item.k, item.data * factor, yerr=item.error * factor,
                    color=item.color, fmt='o', ms=4, label=f'z={item.z:g}')
        ax.plot(item.k, item.model * factor, color=item.color, linestyle='dashed')
        if item.model_error is not None:
            ax.fill_between(item.k, (item.model - item.model_error) * factor,
                            (item.model + item.model_error) * factor, color=item.color, alpha=.35)
        if item.realizations is not None:
            ax.plot(item.k, (item.realizations * factor).T, color=item.color, alpha=.05)
        if panels:
            ax.text(.05, .95, _bin_label(item, print_chi2), transform=ax.transAxes,
                    va='top', fontsize=fontsize - 6, wrap=True)
        else:
            ax.text(item.k[-1] + .001, (item.model * factor)[-1],
                    _bin_label(item, print_chi2), fontsize=fontsize - 4)
        ax.set_yscale('log')
        ax.set_ylabel(r'$k_\parallel P_{\rm 1D}/\pi$', fontsize=fontsize)
        ax.legend(loc='lower right', ncol=1 if panels else 4, fontsize=fontsize - 4)
    return fig, axes


def plot_p1d_residuals(bins, *, panels=False, fontsize=20, print_chi2=True):
    """Render data/model ratios; return figure and axes."""
    fig, axes = _axes(bins, panels)
    for item, ax in zip(bins, axes):
        if np.any(item.model == 0):
            raise ValueError(f"Cannot plot residuals for {item.key}, z={item.z}: model contains zeros.")
        ratio = item.data / item.model + item.shift
        error = item.error / np.abs(item.model)
        ax.errorbar(item.k, ratio, yerr=error, color=item.color, fmt='o', ms=4, label=f'z={item.z:g}')
        ax.axhline(1 + item.shift, color=item.color, linestyle=':', alpha=.5)
        if item.model_error is not None:
            uncertainty = item.model_error / np.abs(item.model)
            ax.fill_between(item.k, 1 + item.shift - uncertainty, 1 + item.shift + uncertainty,
                            color=item.color, alpha=.35)
        if item.realizations is not None:
            ax.plot(item.k, (item.realizations / item.model + item.shift).T, color=item.color, alpha=.025)
        if panels:
            extent = 1.05 * np.max(np.abs(np.r_[ratio - error - 1, ratio + error - 1]))
            ax.set_ylim(1 - max(extent, .01), 1 + max(extent, .01))
            ax.text(.05, .05, _bin_label(item, print_chi2), transform=ax.transAxes, fontsize=fontsize - 4)
        elif print_chi2:
            ax.text(item.k[0], .75 + item.shift, _bin_label(item, True), fontsize=fontsize - 4)
        ax.set_ylabel(r'$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 4)
    return fig, axes


def _finish(fig, axes, chi2, dof, *, fontsize, ylims, plot_fname, show):
    probability = chi2_distribution.sf(chi2, dof)
    fig.suptitle(rf'$\chi^2={chi2:.2f}$, $n_\mathrm{{deg}}={dof}$, prob={probability * 100:.4g}%', fontsize=fontsize)
    unique_axes = list(dict.fromkeys(axes))
    if ylims is not None:
        limits = np.asarray(ylims)
        if limits.shape == (2,):
            limits = np.tile(limits, (len(unique_axes), 1))
        elif limits.shape == ((len(unique_axes) + 2) // 3, 2) and len(unique_axes) > 1:
            limits = np.repeat(limits, 3, axis=0)[:len(unique_axes)]
        if limits.shape != (len(unique_axes), 2):
            raise ValueError('ylims must be a pair, one pair per axis, or one pair per panel row.')
        for ax, limit in zip(unique_axes, limits):
            ax.set_ylim(*limit)
    for ax in unique_axes:
        ax.tick_params(axis='both', labelsize=fontsize)
    fig.supxlabel(r'$k_\parallel\,[\mathrm{km}^{-1}\mathrm{s}]$', fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.15, hspace=.15)
    if plot_fname is not None:
        fig.savefig(str(plot_fname) + '.pdf')
        fig.savefig(str(plot_fname) + '.png')
    elif show:
        plt.show()


def _output(bins):
    result = {}
    for item in bins:
        values = dict(zs=item.z, k_kms=item.k, p1d_data=item.data, p1d_model=item.model,
                      p1d_err=item.error, chi2=item.chi2, prob=item.probability)
        entry = result.setdefault(item.key, {key: [] for key in values})
        for key, value in values.items():
            entry[key].append(value)
    return result


def plot_p1d(
    likelihood, values=None, plot_every_iz=1, residuals=False, plot_fname=None,
    rand_posterior=None, show=True, return_covar=False, print_ratio=False,
    print_chi2=True, return_all=False, collapse=False, plot_realizations=True,
    zmask=None, n_perturb=0, plot_panels=False, z_at_time=False, fontsize=20,
    glob_full=False, fix_cosmo=False, n_param_glob_full=16, chi2_nozcov=False,
    ylims=None, store_data=False,
):
    """Dispatch to spectra or residual rendering with the historical API.

    ``return_all`` returns native spectra and diagnostics grouped by dataset.
    ``store_data`` returns plotted coordinates (flat for one dataset, keyed
    by dataset for multiple datasets). Otherwise the return value is ``None``.
    Figures remain accessible through Matplotlib. Emulator covariance is not
    exposed by the current rebinned likelihood; use posterior draws for bands.
    """
    if return_covar:
        raise NotImplementedError(
            'return_covar=True is unsupported by the current rebinned likelihood. '
            'Use rand_posterior for model uncertainty bands, or return_covar=False.'
        )
    if z_at_time and rand_posterior is not None:
        raise ValueError('rand_posterior describes a joint fit; use it with z_at_time=False.')
    bins, chi2, dof = P1DPlotter(likelihood).prepare(
        values, zmask=zmask, plot_every_iz=plot_every_iz, z_at_time=z_at_time,
        rand_posterior=rand_posterior, n_perturb=n_perturb if plot_realizations else 0,
        collapse=collapse, plot_panels=plot_panels, glob_full=glob_full,
        fix_cosmo=fix_cosmo, n_param_glob_full=n_param_glob_full, chi2_nozcov=chi2_nozcov,
    )
    renderer = plot_p1d_residuals if residuals else plot_p1d_spectra
    with plt.rc_context({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'}):
        fig, axes = renderer(bins, panels=plot_panels, fontsize=fontsize, print_chi2=print_chi2)
        _finish(fig, axes, chi2, dof, fontsize=fontsize, ylims=ylims, plot_fname=plot_fname, show=show)
    if print_ratio and getattr(likelihood, 'rank', 0) == 0:
        for item in bins:
            print(item.data / item.model)
    if return_all:
        return _output(bins)
    if store_data:
        result = {}
        for item in bins:
            factor = 1 / item.model if residuals else item.k / np.pi
            entry = result.setdefault(item.key, {})
            entry[f'x{item.index}'] = item.k
            entry[f'y{item.index}'] = item.data * factor + (item.shift if residuals else 0)
            entry[f'yerr{item.index}'] = item.error * np.abs(factor)
        return next(iter(result.values())) if len(result) == 1 else result


def old_plot_p1d(likelihood, *args, **kwargs):
    """Deprecated alias using the maintained P1D renderer and data interface."""
    warnings.warn('old_plot_p1d is deprecated; use plot_p1d.', DeprecationWarning, stacklevel=2)
    return plot_p1d(likelihood, *args, **kwargs)


def plot_p1d_errors(likelihood, values=None, plot_fname=None, show=True,
                    zmask=None, z_at_time=False, fontsize=16):
    """Plot standardized-residual histograms against a unit Gaussian.

    Return ``bins``, ``zs``, and ``(d-m)/err`` as in the historical routine.
    For multiple datasets, these dictionaries are keyed by dataset name.
    """
    bins, chi2, dof = P1DPlotter(likelihood).prepare(values, zmask=zmask, z_at_time=z_at_time)
    rows = (len(bins) + 2) // 2
    fig, grid = plt.subplots(rows, 2, squeeze=False, figsize=(12, 3 * rows))
    axes = grid.ravel()
    edges = np.linspace(-5, 5, 50)
    x = np.linspace(-5, 5, 100)
    result = {}
    for item, ax in zip(bins, axes):
        residual = (item.data - item.model) / item.error
        ax.hist(residual, bins=edges, density=True, label=f'{item.key}, z={item.z:g}')
        ax.plot(x, norm.pdf(x), color='C1')
        ax.legend()
        entry = result.setdefault(item.key, {'bins': edges, 'zs': [], '(d-m)/err': []})
        entry['zs'].append(item.z)
        entry['(d-m)/err'].append(residual)
    combined = np.concatenate([r for entry in result.values() for r in entry['(d-m)/err']])
    axes[len(bins)].hist(combined, bins=edges, density=True, label='All')
    axes[len(bins)].plot(x, norm.pdf(x), color='C1')
    axes[len(bins)].legend()
    for ax in axes[len(bins) + 1:]:
        ax.set_visible(False)
    fig.supxlabel('(data - model)/error', fontsize=fontsize)
    fig.supylabel('PDF', fontsize=fontsize)
    fig.tight_layout()
    if plot_fname is not None:
        fig.savefig(str(plot_fname) + '.pdf')
        fig.savefig(str(plot_fname) + '.png')
    elif show:
        plt.show()
    return next(iter(result.values())) if len(result) == 1 else result

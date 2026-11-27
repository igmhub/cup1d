"""Renderers of prepared P1D bins (:class:`cup1d.postprocessing.p1d.bins.P1DBin`).

Every renderer takes bins (from :class:`P1DPlotter` or built by another
sampler, e.g. lya_interface) and returns the figure; nothing is shown or
printed.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution, norm

from cup1d.postprocessing.labels import K_KMS_LABEL, KP1D_LABEL
from cup1d.postprocessing.p1d.bins import P1DPlotter
from cup1d.postprocessing.style import apply_house_style, styled


def _axes(bins, panels):
    """Create grouped or per-bin axes for prepared P1D bins.

    Parameters
    ----------
    bins : sequence of P1DBin
        Prepared bins defining dataset grouping and panel count.
    panels : bool
        Create a three-column per-bin layout when true; otherwise share one
        axis for every dataset key.

    Returns
    -------
    tuple
        Matplotlib figure and a list of axes aligned with ``bins``.
    """
    if panels:
        rows = (len(bins) + 2) // 3
        fig, axes = plt.subplots(
            rows, 3, squeeze=False, figsize=(12, rows * 2.5),
            sharex=True, sharey="row",
        )
        axes = axes.ravel()
        for ax in axes[len(bins):]:
            ax.set_visible(False)
        return fig, list(axes[:len(bins)])
    keys = list(dict.fromkeys(b.key for b in bins))
    fig, axes = plt.subplots(
        len(keys), 1, squeeze=False, figsize=(16, max(14, 10 * len(keys))), sharex=True
    )
    lookup = dict(zip(keys, axes.ravel()))
    return fig, [lookup[b.key] for b in bins]


def _bin_label(item, print_chi2):
    """Format a redshift-bin annotation.

    Parameters
    ----------
    item : P1DBin
        Prepared bin whose diagnostics are displayed.
    print_chi2 : bool
        Include degrees of freedom and survival probability instead of data
        count.

    Returns
    -------
    str
        LaTeX-formatted Matplotlib annotation.
    """
    if print_chi2:
        return (rf"$\chi^2={item.chi2:.2f}$, "
                rf"$n_\mathrm{{data}}={np.count_nonzero(item.data)}$")
    return rf"$z={item.z:g}$, $\chi^2={item.chi2:.2f}$, $n_\mathrm{{data}}={np.count_nonzero(item.data)}$"


def _overlay_reference(bins, axes, *, alpha=0.5):
    """Draw the first dataset's measurements behind the other datasets' axes.

    Each reference redshift bin is drawn as a dotted line in the color of the
    matching redshift (tolerance ``1e-3``) on the target axis; reference bins
    without a matching redshift are skipped.

    Parameters
    ----------
    bins : sequence of P1DBin
        Prepared bins aligned with ``axes``.
    axes : sequence of matplotlib.axes.Axes
        Axes aligned with ``bins`` (one axis per dataset key).
    alpha : float, default: 0.5
        Line transparency of the reference measurements.
    """
    keys = list(dict.fromkeys(b.key for b in bins))
    if len(keys) < 2:
        return
    reference = [b for b in bins if b.key == keys[0]]
    for key in keys[1:]:
        targets = [(b, ax) for b, ax in zip(bins, axes) if b.key == key]
        ax = targets[0][1]
        for ref in reference:
            match = [b for b, _ in targets if np.isclose(b.z, ref.z, atol=1e-3, rtol=0)]
            if not match:
                continue
            ax.plot(ref.k, ref.data * ref.k / np.pi, color=match[0].color,
                    linestyle=':', lw=2, alpha=alpha, zorder=0)
        ax.plot([], [], color='k', linestyle=':', lw=2, alpha=alpha,
                label=f'{keys[0]} data')


def _dataset_titles(axes, keys, stats, fontsize):
    """Title each per-dataset axis with its summed chi-squared and data count.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Axes aligned with prepared bins.
    keys : sequence of str
        Dataset key of each entry in ``axes``.
    stats : mapping[str, sequence]
        ``(sum of per-redshift chi2, ndata)`` for every dataset key, summed
        over all selected redshifts.
    fontsize : float
        Base figure font size.
    """
    for key, ax in dict(zip(keys, axes)).items():
        chi2_key, ndata_key = stats[key]
        ax.set_title(rf'{key}: $\sum_z\chi^2_z={chi2_key:.2f}$, $n_\mathrm{{data}}={ndata_key}$',
                     fontsize=fontsize - 2)


def plot_p1d_spectra(bins, *, panels=False, fontsize=20, print_chi2=True):
    """Render prepared bins as dimensionless P1D spectra.

    Parameters
    ----------
    bins : sequence of P1DBin
        Prepared P1D measurements in native velocity units.
    panels : bool, default: False
        Use one panel per selected bin instead of one axis per dataset.
    fontsize : float, default: 20
        Base figure font size.
    print_chi2 : bool, default: True
        Annotate bins with chi-squared probabilities.

    Returns
    -------
    tuple
        Matplotlib figure and axes.  Ordinate values are ``k P1D / pi``.
    """
    fig, axes = _axes(bins, panels)
    for index, (item, ax) in enumerate(zip(bins, axes)):
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
        if not panels or index % 3 == 0:
            ax.set_ylabel(KP1D_LABEL, fontsize=fontsize)
    if not panels:
        _overlay_reference(bins, axes)
    for ax in dict.fromkeys(axes):
        ax.legend(loc='lower right', ncol=1 if panels else 4, fontsize=fontsize - 4)
    return fig, axes


def plot_p1d_residuals(bins, *, panels=False, fontsize=20, print_chi2=True):
    """Render data-to-model P1D ratios and uncertainty bands.

    Parameters
    ----------
    bins : sequence of P1DBin
        Prepared P1D measurements in native velocity units.
    panels : bool, default: False
        Use one panel per selected bin instead of one axis per dataset.
    fontsize : float, default: 20
        Base figure font size.
    print_chi2 : bool, default: True
        Annotate bins with chi-squared probabilities.

    Returns
    -------
    tuple
        Matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any model P1D value is zero.
    """
    fig, axes = _axes(bins, panels)
    for index, (item, ax) in enumerate(zip(bins, axes)):
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
        if not panels or index % 3 == 0:
            ax.set_ylabel(r'$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 4)
    return fig, axes


def _finish(fig, axes, chi2, dof, *, fontsize, ylims, plot_fname, show=None, ndata=None):
    """Finalize labels, limits, title, display, and optional figure files.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalize.
    axes : sequence of matplotlib.axes.Axes
        Axes created for prepared bins, possibly with repeated entries.
    chi2 : float
        Total chi-squared displayed in the title.
    dof : int
        Degrees of freedom used for the chi-squared survival probability.
    fontsize : float
        Base figure font size.
    ylims : array_like, optional
        One pair, one pair per distinct axis, or one pair per panel row.
    plot_fname : str or path-like, optional
        File root for PDF and PNG output.
    show : object
        Ignored (figures are returned, never shown).
    ndata : int, optional
        Total number of data points, shown in the title when given.

    Raises
    ------
    ValueError
        If ``ylims`` cannot be aligned with the distinct plotted axes.
    """
    probability = chi2_distribution.sf(chi2, dof)
    ndata_str = '' if ndata is None else rf'$n_\mathrm{{data}}={ndata}$, '
    fig.suptitle(rf'$\chi^2={chi2:.2f}$, {ndata_str}$n_\mathrm{{deg}}={dof}$, '
                 f'prob={probability * 100:.4g}%', fontsize=fontsize)
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
    fig.supxlabel(K_KMS_LABEL, fontsize=fontsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=.15, hspace=.15)
    if axes and axes[0].get_title():
        # extra 0.5 inch between the global title and the first panel title
        fig.subplots_adjust(top=fig.subplotpars.top - 0.5 / fig.get_figheight())
    if plot_fname is not None:
        fig.savefig(str(plot_fname) + '.pdf')
        fig.savefig(str(plot_fname) + '.png')


def _output(bins):
    """Collect native P1D diagnostics grouped by dataset key.

    Parameters
    ----------
    bins : sequence of P1DBin
        Prepared bins to serialize.

    Returns
    -------
    dict[str, dict[str, list]]
        Redshifts, ``k_kms`` in ``s / km``, P1D data/model/error in ``km / s``,
        chi-squared values, and survival probabilities grouped by dataset.
    """
    result = {}
    for item in bins:
        values = dict(zs=item.z, k_kms=item.k, p1d_data=item.data, p1d_model=item.model,
                      p1d_err=item.error, chi2=item.chi2, prob=item.probability)
        entry = result.setdefault(item.key, {key: [] for key in values})
        for key, value in values.items():
            entry[key].append(value)
    return result


def plot_p1d_bins(bins, chi2, dof, *, residuals=False, panels=False, ndata=None, fontsize=20,
                  print_chi2=True, ylims=None, dataset_stats=None, plot_fname=None):
    """Data against model of prepared bins, with the chi-squared title.

    Parameters
    ----------
    bins, chi2, dof
        Output of :meth:`P1DPlotter.prepare`.
    residuals : bool, default=False
        Data/model instead of ``k P1D / pi``.
    panels : bool, default=False
        One panel per bin instead of one axis per data set.
    ndata : int, optional
        Number of data points (title).
    fontsize : float, default=20
        Base font size.
    print_chi2 : bool, default=True
        Annotate the bins with their chi-squared.
    ylims : array_like, optional
        One pair, one per axis, or one per panel row.
    dataset_stats : mapping, optional
        ``P1DPlotter.dataset_stats``: titles of the data-set axes.
    plot_fname : str or path-like, optional
        File root for PDF and PNG output.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_house_style()
    renderer = plot_p1d_residuals if residuals else plot_p1d_spectra
    fig, axes = renderer(bins, panels=panels, fontsize=fontsize, print_chi2=print_chi2)
    if dataset_stats is not None and not panels and print_chi2:
        _dataset_titles(axes, [b.key for b in bins], dataset_stats, fontsize)
    _finish(fig, axes, chi2, dof, ndata=ndata, fontsize=fontsize, ylims=ylims, plot_fname=plot_fname)
    return fig


def p1d_figure_data(bins, residuals=False):
    """Arrays drawn by :func:`plot_p1d_spectra` / :func:`plot_p1d_residuals` (Zenodo files).

    Returns
    -------
    dict
        ``x<i>``, ``y<i>``, ``yerr<i>`` per redshift index (``k P1D / pi``, or
        data/model plus the residual offset); one such dict per data set when
        there are several.
    """
    result = {}
    for item in bins:
        factor = 1 / item.model if residuals else item.k / np.pi
        entry = result.setdefault(item.key, {})
        entry[f'x{item.index}'] = item.k
        entry[f'y{item.index}'] = item.data * factor + (item.shift if residuals else 0)
        entry[f'yerr{item.index}'] = item.error * np.abs(factor)
    return next(iter(result.values())) if len(result) == 1 else result


def p1d_native_output(bins):
    """Redshifts, k, data, model, errors, chi2 and probabilities of prepared bins, per data set."""
    return _output(bins)


def plot_p1d(
    likelihood, values=None, plot_every_iz=1, residuals=False, plot_fname=None,
    rand_posterior=None, show=None, return_covar=False, print_ratio=False,
    print_chi2=True, return_all=False, collapse=False, plot_realizations=True,
    zmask=None, n_perturb=0, plot_panels=False, z_at_time=False, fontsize=20,
    glob_full=False, fix_cosmo=False, n_param_glob_full=16, chi2_nozcov=False,
    ylims=None, store_data=False,
):
    """Plot the P1D data against the model of a likelihood (``Likelihood.plot_p1d``).

    Parameters
    ----------
    likelihood : object
        Re-binned likelihood exposing P1D predictions and covariance.
    values, zmask, plot_every_iz, z_at_time, rand_posterior, n_perturb
        Sampling, redshift selection, posterior-band, and realization options
        forwarded to :meth:`P1DPlotter.prepare`.
    residuals : bool, default: False
        Plot data/model ratios rather than dimensionless spectra.
    plot_fname : str or path-like, optional
        File root for PDF and PNG output.
    print_chi2 : bool, default: True
        Annotate the bins with their chi-squared.
    collapse, plot_realizations, plot_panels, glob_full, fix_cosmo,
    n_param_glob_full, chi2_nozcov, ylims, fontsize
        Rendering and degrees-of-freedom options forwarded to preparation or
        finalization.
    show, print_ratio : object
        Deprecated and ignored: the figure is returned, never shown or printed.
    return_all, store_data : bool
        Deprecated: return :func:`p1d_native_output` or
        :func:`p1d_figure_data` of the bins instead of the figure. Use
        :class:`P1DPlotter` and those functions.
    return_covar : bool
        Unsupported (raises).

    Returns
    -------
    matplotlib.figure.Figure
        Unless ``return_all`` or ``store_data``.

    Raises
    ------
    NotImplementedError
        If ``return_covar`` is requested.
    ValueError
        If posterior samples are combined with redshift-at-a-time fitting or
        any selection/preparation contract is invalid.
    """
    if return_covar:
        raise NotImplementedError(
            'return_covar=True is unsupported by the current rebinned likelihood. '
            'Use rand_posterior for model uncertainty bands, or return_covar=False.'
        )
    if print_ratio:
        warnings.warn('print_ratio is ignored; compute data/model from P1DPlotter bins.',
                      DeprecationWarning, stacklevel=2)
    if z_at_time and rand_posterior is not None:
        raise ValueError('rand_posterior describes a joint fit; use it with z_at_time=False.')
    plotter = P1DPlotter(likelihood)
    bins, chi2, dof = plotter.prepare(
        values, zmask=zmask, plot_every_iz=plot_every_iz, z_at_time=z_at_time,
        rand_posterior=rand_posterior, n_perturb=n_perturb if plot_realizations else 0,
        collapse=collapse, plot_panels=plot_panels, glob_full=glob_full,
        fix_cosmo=fix_cosmo, n_param_glob_full=n_param_glob_full, chi2_nozcov=chi2_nozcov,
    )
    fig = plot_p1d_bins(bins, chi2, dof, residuals=residuals, panels=plot_panels, ndata=plotter.ndata,
                        fontsize=fontsize, print_chi2=print_chi2, ylims=ylims, plot_fname=plot_fname,
                        dataset_stats=None if z_at_time else plotter.dataset_stats)
    if return_all:
        return _output(bins)
    if store_data:
        return p1d_figure_data(bins, residuals=residuals)
    return fig


def p1d_standardized_residuals(bins):
    """``(data - model) / error`` of prepared bins, per data set.

    Returns
    -------
    dict
        Per data set (the dict itself when there is one): ``zs`` and
        ``(d-m)/err`` (one array per redshift).
    """
    result = {}
    for item in bins:
        entry = result.setdefault(item.key, {'zs': [], '(d-m)/err': []})
        entry['zs'].append(item.z)
        entry['(d-m)/err'].append((item.data - item.model) / item.error)
    return next(iter(result.values())) if len(result) == 1 else result


@styled
def plot_p1d_errors(likelihood, values=None, plot_fname=None, zmask=None, z_at_time=False,
                    fontsize=16):
    """Histograms of the standardized P1D residuals against a unit Gaussian.

    Parameters
    ----------
    likelihood : object
        Re-binned likelihood exposing P1D predictions and covariance.
    values : array_like or mapping, optional
        Sampling point forwarded to :meth:`P1DPlotter.prepare`.
    plot_fname : str or path-like, optional
        File root for PDF and PNG output.
    zmask : array_like, optional
        Selected redshifts.
    z_at_time : bool, default: False
        Interpret ``values`` as one point per selected redshift.
    fontsize : float, default: 16
        Base axis-label font size.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per bin and one with all of them
        (:func:`p1d_standardized_residuals` gives the arrays).
    """
    bins, _, _ = P1DPlotter(likelihood).prepare(values, zmask=zmask, z_at_time=z_at_time)
    rows = (len(bins) + 2) // 2
    fig, grid = plt.subplots(rows, 2, squeeze=False, figsize=(12, 3 * rows))
    axes = grid.ravel()
    edges = np.linspace(-5, 5, 50)
    x = np.linspace(-5, 5, 100)
    residuals = []
    for item, ax in zip(bins, axes):
        residual = (item.data - item.model) / item.error
        residuals.append(residual)
        ax.hist(residual, bins=edges, density=True, label=f'{item.key}, z={item.z:g}')
        ax.plot(x, norm.pdf(x), color='C1')
        ax.legend()
    axes[len(bins)].hist(np.concatenate(residuals), bins=edges, density=True, label='All')
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
    return fig


@styled
def plot_p1d_with_residuals(bins, ncols=4, title=None, fontsize=10, residual_range=0.1):
    """``k P1D / pi`` above the relative residual, one panel pair per bin.

    Parameters
    ----------
    bins : sequence of P1DBin
        Bins of every data set and redshift (``key``, ``z``, ``k``, ``data``,
        ``model``, ``error`` and ``chi2``; the other fields are not used).
        Panels are grouped by data set (in the order of their first bin; named
        in the titles when there are several), then ordered in redshift.
    ncols : int, default=4
        Panel columns.
    title : str, optional
        Figure title (e.g. the correlated joint chi-squared).
    fontsize : float, default=10
        Font size of the panel titles.
    residual_range : float, default=0.1
        Y range of the residual panels (+-).

    Returns
    -------
    fig : matplotlib.figure.Figure
    spectra : numpy.ndarray of matplotlib.axes.Axes
        The spectrum axes, one per bin.

    Notes
    -----
    Residuals are ``data / model - 1`` with marginal (diagonal) errors
    ``error / model``; the chi-squared of every panel is that of its bin.
    """
    from matplotlib.ticker import MaxNLocator

    keys = list(dict.fromkeys(item.key for item in bins))
    several = len(keys) > 1
    entries = sorted(bins, key=lambda item: (keys.index(item.key), item.z))
    nrows = -(-len(entries) // ncols)
    fig = plt.figure(figsize=(4.2 * ncols, 4.2 * nrows))
    outer = fig.add_gridspec(nrows, ncols, left=0.06, right=0.99, bottom=0.05,
                             top=0.88, wspace=0.28, hspace=0.3)
    spectra = []
    for n, item in enumerate(entries):
        k, pk, mod, err = (np.asarray(v) for v in (item.k, item.data, item.model, item.error))
        inner = outer[n // ncols, n % ncols].subgridspec(2, 1, height_ratios=(3, 1), hspace=0)
        top = fig.add_subplot(inner[0])
        bottom = fig.add_subplot(inner[1], sharex=top)
        top.errorbar(k, k * pk / np.pi, k * err / np.pi, fmt="o", ms=3, capsize=2, color="C0",
                     label="data")
        top.plot(k, k * mod / np.pi, color="C1", label="model")
        label = f"{item.key} " if several else ""
        top.set_title(f"{label}z={item.z:.1f}  $\\chi^2$={item.chi2:.1f}/{len(k)}", fontsize=fontsize)
        top.tick_params(labelbottom=False)
        bottom.errorbar(k, pk / mod - 1, err / mod, fmt="o", ms=3, capsize=2, color="C0")
        bottom.axhline(0, color="C1", lw=1)
        bottom.axhspan(-0.01, 0.01, color="gray", alpha=0.2, lw=0)
        bottom.set_ylim(-residual_range, residual_range)
        bottom.set_yticks([-residual_range / 2, 0, residual_range / 2])
        top.yaxis.set_major_locator(MaxNLocator(prune="lower"))
        if n % ncols == 0:
            top.set_ylabel(KP1D_LABEL)
            bottom.set_ylabel("data/model - 1")
        if n >= len(entries) - ncols:
            bottom.set_xlabel(K_KMS_LABEL)
        spectra.append(top)
    spectra[0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    return fig, np.array(spectra)

"""Redshift histories: IGM parameters and systematics against redshift."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cup1d.postprocessing.labels import IGM_LABELS, RESOLUTION_LABEL, Z_LABEL
from cup1d.postprocessing.posterior.stats import weighted_quantile

# IGM history family -> sub-model of cup1d's IGM class
_IGM_MODELS = {"tau_eff": "F_model", "sigT_kms": "T_model", "gamma": "T_model", "kF_kms": "P_model"}

def _family_history(igm, family, z, columns):
    """Physical history of one family for columnar parameter samples, shape (n, nz)."""
    model = igm.models[_IGM_MODELS[family]]
    return model.get_value(family, z, columns) * model.fid_interp[family](z)[None, :]


def igm_histories(igm, columns, z):
    """Evaluate IGM histories for many parameter points at once.

    Parameters
    ----------
    igm : object
        cup1d IGM model (``cup1d.models.igm.IGM`` or lya_interface's copy),
        with ``F_model``, ``T_model`` and ``P_model`` histories.
    columns : mapping
        Native coefficient name to an array of shape ``(n,)``, e.g.
        ``samples.columns()`` or ``{name: [value]}`` for a single point.
        Coefficients absent from it keep the model's stored values.
    z : array_like
        Redshifts.

    Returns
    -------
    dict
        ``tau_eff``, ``mF``, ``sigT_kms`` (km/s), ``T0`` (in 10^4 K),
        ``gamma`` and ``kF_kms`` (s/km), each with shape ``(n, nz)``.
    """
    from lace.cosmo import thermal_broadening

    z = np.atleast_1d(np.asarray(z, dtype=float))
    columns = {name: np.atleast_1d(np.asarray(value, dtype=float)) for name, value in columns.items()}
    out = {family: _family_history(igm, family, z, columns) for family in _IGM_MODELS}
    out["mF"] = np.exp(-out["tau_eff"])
    out["T0"] = thermal_broadening.T0_from_broadening_kms(out["sigT_kms"]) / 1e4
    return out


def load_dr1_igm(M_of_z=None):
    """Return the DESI DR1 P1D IGM posterior packaged with ForestFlow.

    Parameters
    ----------
    M_of_z : callable, optional
        ``H(z)/(1+z)`` in km/s/Mpc, used to convert the stored comoving
        ``sigT_Mpc`` and ``kF_Mpc`` to velocity units. Without it only the
        mean flux and gamma are returned.

    Returns
    -------
    dict
        ``z`` (shape ``(nz,)``) and samples of shape ``(1000, nz)`` of ``mF``,
        ``gamma`` and, with ``M_of_z``, ``sigT_kms``, ``T0`` (10^4 K) and
        ``kF_kms``.

    Notes
    -----
    These are the Chaves-Montero+26 (DESI DR1 P1D) samples behind ForestFlow's
    IGM priors (``forestflow.priors.get_IGM_priors``). The conversion uses the
    supplied ``M_of_z`` (e.g. the fiducial cosmology of the analysis), not the
    cosmology of every DR1 sample, an approximation well below the errors.
    """
    from forestflow.priors import get_IGM_priors
    from lace.cosmo import thermal_broadening

    _, raw = get_IGM_priors(3.0, return_all=True)
    out = {"z": np.asarray(raw["zs"], dtype=float), "mF": raw["mF"], "gamma": raw["gamma"]}
    if M_of_z is not None:
        M = np.array([M_of_z(z) for z in out["z"]])
        out["sigT_kms"] = raw["sigT_Mpc"] * M[None, :]
        out["T0"] = thermal_broadening.T0_from_broadening_kms(out["sigT_kms"]) / 1e4
        out["kF_kms"] = raw["kF_Mpc"] / M[None, :]
    return out


LITERATURE_STYLES = {"Gaikwad+21": ("C1", "--"), "Turner+24": ("C2", "-.")}


def draw_literature_igm(ax, quantity, literature=None, marker=None):
    """Draw the literature IGM measurements of one quantity (Gaikwad+21, Turner+24)."""
    if literature is None:
        from cup1d.postprocessing.literature import literature_igm

        literature = literature_igm()
    for name, data in literature.items():
        if quantity not in data:
            continue
        color, ls = LITERATURE_STYLES.get(name, ("C4", ":"))
        ax.errorbar(data["z"], data[quantity], yerr=data[quantity + "_err"], fmt=ls, marker=marker,
                    ms=4, color=color, alpha=0.75, lw=2, label=name)


def draw_training_cloud(ax, quantity, emulator_igm, label="Training data"):
    """Scatter the IGM histories of the emulator training simulations.

    ``emulator_igm``: ``{simulation: {"z": ..., quantity: ...}}`` (e.g.
    ``theory.emu_igm_all``); entries whose label does not end in a digit
    (averaged or special simulations) and zero values are skipped.
    """
    first = True
    for sim_label, history in emulator_igm.items():
        if not sim_label[-1].isdigit() or quantity not in history:
            continue
        keep = np.asarray(history[quantity]) != 0
        if keep.any():
            ax.scatter(np.asarray(history["z"])[keep], np.asarray(history[quantity])[keep], marker=".",
                       color="C1", alpha=0.75 if first else 0.1, s=10, label=label if first else None)
            first = False


def draw_prior_envelope(ax, quantity, envelope, label="Fiducial"):
    """Fiducial IGM history and the histories at the prior edges (red lines).

    ``envelope``: output of :func:`cup1d.postprocessing.fit.igm.igm_prior_envelope`.
    """
    z = envelope["z"]
    for key, alpha, lw, lab in (("fiducial", 0.5, 1.5, label), ("low", 0.3, 1, None), ("high", 0.3, 1, None)):
        values = np.asarray(envelope[key][quantity]).reshape(-1)
        keep = values != 0
        ax.plot(z[keep], values[keep], "C3-", alpha=alpha, lw=lw, label=lab)


def draw_truth(ax, quantity, truth, label="Truth"):
    """IGM history of a mock's truth (``{"z": ..., quantity: ...}``)."""
    if quantity in truth:
        values = np.asarray(truth[quantity])
        keep = values != 0
        ax.plot(np.asarray(truth["z"])[keep], values[keep], "C3:o", alpha=0.8, label=label)


def plot_igm_histories(histories, z, weights=None, best_fit=None, dr1=None,
                       quantities=("mF", "T0", "gamma", "kF_kms"), literature=True,
                       label="This work", fontsize=16, layout="grid",
                       levels=((0.16, 0.84), (0.025, 0.975)), best_fit_label="Best fit",
                       best_fit_nodes=None, prior_envelope=None, prior_label="Fiducial",
                       training=None, truth=None, extra=None, ylims=None, title=None):
    """Plot IGM histories: posterior bands and any of the overlays.

    Parameters
    ----------
    histories : mapping, optional
        Output of :func:`igm_histories` for posterior samples, shape ``(n, nz)``
        (``None``: no bands, e.g. for a minimizer result).
    z : array_like
        Redshifts of ``histories`` and ``best_fit``.
    weights : array_like, shape (n,), optional
        Sample weights.
    best_fit : mapping, optional
        Output of :func:`igm_histories` for one point (dotted line).
    dr1 : mapping, optional
        Output of :func:`load_dr1_igm` (median and 68 per cent interval).
    quantities : sequence of str, default=("mF", "T0", "gamma", "kF_kms")
        Panels, keys of :data:`cup1d.postprocessing.labels.IGM_LABELS`.
    literature : bool or mapping, default=True
        Overlay Gaikwad+21 and Turner+24 (or the given
        :func:`cup1d.postprocessing.literature.literature_igm` mapping).
    label : str, default="This work"
        Legend label of the posterior bands.
    fontsize : float, default=16
        Base font size.
    layout : {"grid", "column"}, default="grid"
        Two columns, or one column with a taller first panel (the DR1 paper
        figure of mean flux, temperature and gamma).
    levels : sequence of (float, float)
        Quantile pairs of the bands, innermost first.
    best_fit_label : str, default="Best fit"
        Legend label of ``best_fit``.
    best_fit_nodes : mapping, optional
        ``{quantity: (z_nodes, values)}`` drawn as dots on the best fit.
    prior_envelope : mapping, optional
        Fiducial and prior-edge histories (red lines), from
        :func:`cup1d.postprocessing.fit.igm.igm_prior_envelope`.
    prior_label : str, default="Fiducial"
        Legend label of the fiducial history.
    training : mapping, optional
        Emulator training IGM histories (:func:`draw_training_cloud`).
    truth : mapping, optional
        Truth of a mock (:func:`draw_truth`).
    extra : mapping, optional
        ``{label: {"z": ..., quantity: values}}`` further histories (dashed).
    ylims : mapping, optional
        ``{quantity: (low, high)}``.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from cup1d.postprocessing.primitives import draw_band
    from cup1d.postprocessing.style import apply_house_style

    apply_house_style()
    z = np.asarray(z)
    if layout == "column":
        ratios = [3] + [1] * (len(quantities) - 1)
        fig, axes = plt.subplots(len(quantities), 1, figsize=(8, 10 * len(quantities) / 3), sharex=True,
                                 squeeze=False, gridspec_kw={"height_ratios": ratios})
        ncols = 1
    else:
        ncols = 2 if len(quantities) > 1 else 1
        nrows = -(-len(quantities) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), sharex=True,
                                 squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(quantities):]:
        ax.set_visible(False)
    others = None
    if literature is not False and literature is not None:
        if literature is True:
            from cup1d.postprocessing.literature import literature_igm

            others = literature_igm()
        else:
            others = literature
    for ax, quantity in zip(axes, quantities):
        if truth is not None:
            draw_truth(ax, quantity, truth)
        if training is not None:
            draw_training_cloud(ax, quantity, training)
        if histories is not None:
            draw_band(ax, z, histories[quantity], weights=weights, color="C0", levels=levels,
                      alphas=(0.5, 0.2), label=label)
        if best_fit is not None:
            ax.plot(z, np.asarray(best_fit[quantity])[0], "C0:", lw=3, label=best_fit_label)
            if best_fit_nodes is not None and quantity in best_fit_nodes:
                ax.scatter(*best_fit_nodes[quantity], marker="o", color="C0")
        if dr1 is not None and quantity in dr1:
            low, mid, high = np.percentile(dr1[quantity], [16, 50, 84], axis=0)
            ax.errorbar(dr1["z"], mid, yerr=[mid - low, high - mid], fmt="s", ms=5, color="k",
                        capsize=2, label="DESI DR1 (Chaves-Montero+26)")
        if others is not None:
            draw_literature_igm(ax, quantity, others)
        for i, (name, history) in enumerate((extra or {}).items()):
            if quantity in history:
                ax.plot(history["z"], history[quantity], f"C{5 + i}--", alpha=0.5, lw=3, label=name)
        if prior_envelope is not None:
            draw_prior_envelope(ax, quantity, prior_envelope, label=prior_label)
        ax.set_ylabel(IGM_LABELS[quantity], fontsize=fontsize)
        if quantity == "tau_eff":
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4 if layout == "grid" else 3))
        ax.tick_params(labelsize=fontsize - 4)
        ax.set_xlim(z.min() - 0.05, z.max() + 0.05)
        if ylims is not None and quantity in ylims:
            ax.set_ylim(ylims[quantity])
    for ax in axes[max(len(quantities) - ncols, 0):len(quantities)]:
        ax.set_xlabel(Z_LABEL, fontsize=fontsize)
    handles = {}
    for ax in axes[:len(quantities)]:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(text, handle)
    axes[0].legend(handles.values(), handles.keys(), fontsize=fontsize - (5 if layout == "grid" else 2),
                   loc="best")
    if title:
        fig.suptitle(title, fontsize=fontsize + 2)
    fig.tight_layout()
    return fig


def plot_parameters_vs_z(samples, families, best_fit=None, fontsize=16, labels=None):
    """Plot the posterior of redshift-node coefficients against redshift.

    Parameters
    ----------
    samples : PosteriorSamples
        Posterior.
    families : mapping
        History family (e.g. ``"R_coeff"``) to its node redshifts; the
        coefficient ``<family>_<i>`` is plotted at the ``i``-th node.
    best_fit : mapping, optional
        ``{name: value}`` drawn as crosses.
    fontsize : float, default=16
        Base font size.
    labels : mapping, optional
        Family to y-axis label; by default the label of its first
        coefficient without the index (``f_\\mathrm{res}`` for ``R_coeff``).

    Returns
    -------
    matplotlib.figure.Figure
        One panel per family with the median and the 68 and 95 per cent
        intervals at every node, and the flat-prior bounds as dotted lines.
    """
    default_labels = {"R_coeff": RESOLUTION_LABEL}
    families = {family: np.asarray(z) for family, z in families.items() if samples.family(family)}
    if not families:
        raise ValueError("none of the requested families is in the samples")
    fig, axes = plt.subplots(len(families), 1, figsize=(8, 3.5 * len(families)), squeeze=False)
    for ax, (family, znodes) in zip(axes[:, 0], families.items()):
        names = samples.family(family)
        idx = [int(name.rsplit("_", 1)[1]) for name in names]
        z = znodes[idx]
        q = np.array([samples.quantiles(name, [0.025, 0.16, 0.5, 0.84, 0.975]) for name in names]).T
        ax.errorbar(z, q[2], yerr=[q[2] - q[0], q[4] - q[2]], fmt="none", color="C0", alpha=0.35,
                    lw=4, label="95%")
        ax.errorbar(z, q[2], yerr=[q[2] - q[1], q[3] - q[2]], fmt="o", color="C0", capsize=3,
                    label="median, 68%")
        if best_fit is not None:
            ax.plot(z, [best_fit[name] for name in names], "x", color="C3", ms=8, mew=2, label="Best fit")
        bounds = [samples.bounds[name] for name in names if name in samples.bounds]
        if bounds and all(b == bounds[0] for b in bounds):
            for edge in bounds[0]:
                ax.axhline(edge, color="k", ls=":", lw=1)
        ax.axhline(0, color="0.5", lw=0.8)
        ylabel = (labels or {}).get(family, default_labels.get(family, family.replace("_", r"\_")))
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(Z_LABEL, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 4)
        spread = q[4].max() - q[0].min()
        ax.set_ylim(q[0].min() - 0.3 * spread, q[4].max() + 0.3 * spread)
        ax.legend(fontsize=fontsize - 5)
    fig.tight_layout()
    return fig

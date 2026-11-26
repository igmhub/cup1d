"""Redshift histories: IGM parameters and systematics against redshift."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cup1d.postprocessing.posterior.stats import weighted_quantile

# IGM history family -> sub-model of cup1d's IGM class
_IGM_MODELS = {"tau_eff": "F_model", "sigT_kms": "T_model", "gamma": "T_model", "kF_kms": "P_model"}

IGM_LABELS = {
    "mF": r"$\bar{F}$",
    "tau_eff": r"$\tau_\mathrm{eff}$",
    "T0": r"$T_0\,[10^4\,\mathrm{K}]$",
    "sigT_kms": r"$\sigma_T\,[\mathrm{km\,s^{-1}}]$",
    "gamma": r"$\gamma$",
    "kF_kms": r"$k_F\,[\mathrm{s\,km^{-1}}]$",
}


def _family_history(igm, family, z, columns):
    """Physical history of one family for columnar parameter samples, shape (n, nz)."""
    model = igm.models[_IGM_MODELS[family]]
    return model.get_value_batch(family, z, columns) * model.fid_interp[family](z)[None, :]


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


def plot_igm_histories(histories, z, weights=None, best_fit=None, dr1=None,
                       quantities=("mF", "T0", "gamma", "kF_kms"), literature=True,
                       label="This work", fontsize=16):
    """Plot posterior IGM histories against DESI DR1 and literature measurements.

    Parameters
    ----------
    histories : mapping
        Output of :func:`igm_histories` for posterior samples, shape ``(n, nz)``.
    z : array_like
        Redshifts of ``histories``.
    weights : array_like, shape (n,), optional
        Sample weights.
    best_fit : mapping, optional
        Output of :func:`igm_histories` for one point (drawn as a line).
    dr1 : mapping, optional
        Output of :func:`load_dr1_igm` (median and 68 per cent interval).
    quantities : sequence of str, default=("mF", "T0", "gamma", "kF_kms")
        Panels, keys of :data:`IGM_LABELS`.
    literature : bool, default=True
        Overlay Gaikwad+21 (mean flux, temperature, gamma, broadening) and
        Turner+24 (mean flux, optical depth).
    label : str, default="This work"
        Legend label of the posterior.
    fontsize : float, default=16
        Base font size.

    Returns
    -------
    matplotlib.figure.Figure
        68 (dark) and 95 (light) per cent bands of the posterior.
    """
    z = np.asarray(z)
    ncols = 2 if len(quantities) > 1 else 1
    nrows = -(-len(quantities) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(quantities):]:
        ax.set_visible(False)
    others = None
    if literature:
        from cup1d.likelihood.likelihood import others_igm

        others = dict(zip(("Gaikwad+21", "Turner+24"), others_igm()))
    for ax, quantity in zip(axes, quantities):
        values = histories[quantity]
        bands = np.array([weighted_quantile(values[:, i], [0.025, 0.16, 0.5, 0.84, 0.975], weights)
                          for i in range(values.shape[1])]).T
        ax.fill_between(z, bands[0], bands[4], color="C0", alpha=0.2, lw=0)
        ax.fill_between(z, bands[1], bands[3], color="C0", alpha=0.5, lw=0, label=label)
        if best_fit is not None:
            ax.plot(z, np.asarray(best_fit[quantity])[0], "C0:", lw=2.5, label="Best fit")
        if dr1 is not None and quantity in dr1:
            low, mid, high = np.percentile(dr1[quantity], [16, 50, 84], axis=0)
            ax.errorbar(dr1["z"], mid, yerr=[mid - low, high - mid], fmt="s", ms=5, color="k",
                        capsize=2, label="DESI DR1 (Chaves-Montero+26)")
        if others is not None:
            for (name, data), color, fmt in zip(others.items(), ("C1", "C2"), ("o--", "D-.")):
                if quantity in data:
                    ax.errorbar(data["z"], data[quantity], yerr=data[quantity + "_err"], fmt=fmt,
                                ms=4, color=color, alpha=0.8, lw=1.5, label=name)
        ax.set_ylabel(IGM_LABELS[quantity], fontsize=fontsize)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=fontsize - 4)
        ax.set_xlim(z.min() - 0.05, z.max() + 0.05)
    for ax in axes[max(len(quantities) - ncols, 0):len(quantities)]:
        ax.set_xlabel(r"$z$", fontsize=fontsize)
    handles = {}
    for ax in axes[:len(quantities)]:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(text, handle)
    axes[0].legend(handles.values(), handles.keys(), fontsize=fontsize - 5, loc="best")
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
    default_labels = {"R_coeff": r"$f_\mathrm{res}$"}
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
        ax.set_xlabel(r"$z$", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 4)
        spread = q[4].max() - q[0].min()
        ax.set_ylim(q[0].min() - 0.3 * spread, q[4].max() + 0.3 * spread)
        ax.legend(fontsize=fontsize - 5)
    fig.tight_layout()
    return fig

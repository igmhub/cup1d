"""Metal and HCD contamination implied by a posterior, at one redshift."""

import matplotlib.pyplot as plt
import numpy as np

from cup1d.postprocessing.posterior.histories import igm_histories
from cup1d.postprocessing.posterior.stats import weighted_quantile

_PAIRS = ("SiIII_Lya", "SiIIa_Lya", "SiIIb_Lya", "SiIIc_Lya", "SiIII_SiIIa", "SiIII_SiIIb",
          "SiIII_SiIIc", "SiIIc_SiIIb", "SiIIc_SiIIa", "SiIIb_SiIIa")

# component label -> transition pairs switched on (as cup1d's figure 20a)
METAL_COMPONENTS = {
    r"Ly$\alpha$-SiIII": ("SiIII_Lya",),
    r"Ly$\alpha$-SiII": ("SiIIa_Lya", "SiIIb_Lya"),
    r"SiII-SiIII": ("SiIII_SiIIa", "SiIII_SiIIb"),
}

HCD_COMPONENTS = {"LLS": "HCD_damp1", "sub-DLA": "HCD_damp2", "small DLA": "HCD_damp3",
                  "large DLA": "HCD_damp4"}

K_LABEL = r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$"


def metal_contamination(contaminants, igm, columns, z, k_kms):
    """Evaluate the silicon contamination of many parameter points at one redshift.

    Parameters
    ----------
    contaminants : cup1d.models.contaminants.model_contaminants.Contaminants
        Contaminant model with the ``Si_mult`` (and optionally ``Si_add``)
        templates.
    igm : object
        cup1d IGM model, for the mean flux that normalizes the metal terms.
    columns : mapping
        Native coefficient name to an array of shape ``(n,)``.
    z : float
        Redshift.
    k_kms : array_like
        Wavenumbers in s/km.

    Returns
    -------
    dict
        Component label to an array of shape ``(n, nk)``: the multiplicative
        corrections of :data:`METAL_COMPONENTS` and their product ("All", all
        default transitions), and ``"SiII-SiII (additive)"`` in km/s when the
        model has the additive template.

    Raises
    ------
    NotImplementedError
        For metal models without a columnar implementation (e.g. Kaiser).
    """
    models = contaminants.metal_models
    if "Si_mult" not in models or not hasattr(models["Si_mult"], "get_contamination_batch"):
        raise NotImplementedError(f"no columnar multiplicative silicon model in {sorted(models)}")
    z = np.array([float(z)])
    k = [np.asarray(k_kms, dtype=float)]
    mF = igm_histories(igm, columns, z)["mF"]
    out = {}
    for label, pairs in METAL_COMPONENTS.items():
        flags = {pair: int(pair in pairs) for pair in _PAIRS}
        out[label] = models["Si_mult"].get_contamination_batch(z, k, mF, columns, remove=flags)[0]
    out["All"] = models["Si_mult"].get_contamination_batch(z, k, mF, columns)[0]
    if "Si_add" in models and hasattr(models["Si_add"], "get_contamination_batch"):
        out["SiII-SiII (additive)"] = models["Si_add"].get_contamination_batch(z, k, mF, columns)[0]
    return out


def hcd_contamination(contaminants, columns, z, k_kms):
    """Evaluate the HCD correction of many parameter points at one redshift.

    Parameters
    ----------
    contaminants : cup1d.models.contaminants.model_contaminants.Contaminants
        Contaminant model with a columnar HCD model (Rogers+18 templates).
    columns : mapping
        Native coefficient name to an array of shape ``(n,)``.
    z : float
        Redshift.
    k_kms : array_like
        Wavenumbers in s/km.

    Returns
    -------
    dict
        Component label (:data:`HCD_COMPONENTS`, then "All") to the
        multiplicative correction, shape ``(n, nk)``. Every component keeps
        only its own damping amplitude, the other free ones set to their
        null value (fixed amplitudes keep their configured values).
    """
    model = contaminants.hcd_model
    if not hasattr(model, "get_contamination_batch"):
        raise NotImplementedError(f"no columnar HCD model in {type(model).__name__}")
    z = np.array([float(z)])
    k = [np.asarray(k_kms, dtype=float)]
    n = len(next(iter(columns.values())))
    out = {}
    for label, family in HCD_COMPONENTS.items():
        if family not in model.null_vals:
            continue
        own = dict(columns)
        for other in HCD_COMPONENTS.values():
            if other != family and other in model.null_vals:
                for index in range(len(model.coeffs.get(other, []))):
                    own[f"{other}_{index}"] = np.full(n, model.null_vals[other], dtype=float)
        out[label] = model.get_contamination_batch(z, k, own)[0]
    out["All"] = model.get_contamination_batch(z, k, columns)[0]
    return out


def plot_contamination(components, k_kms, weights=None, ylabel=None, stacked=False,
                       reference=1.0, title=None, fontsize=16):
    """Plot the posterior median and 68 per cent band of contamination terms.

    Parameters
    ----------
    components : mapping
        Label to samples of shape ``(n, nk)`` (e.g. :func:`metal_contamination`).
    k_kms : array_like
        Wavenumbers in s/km.
    weights : array_like, shape (n,), optional
        Sample weights.
    ylabel : str or mapping, optional
        Y label, or label per component when ``stacked``.
    stacked : bool, default=False
        One panel per component instead of all in one panel.
    reference : float, default=1.0
        Value of no contamination (dotted line); ``0`` is used for additive
        components (labels containing "additive").
    title : str, optional
        Figure title.
    fontsize : float, default=16
        Base font size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    k = np.asarray(k_kms)
    labels = list(components)
    if stacked:
        fig, axes = plt.subplots(len(labels), 1, figsize=(8, 1.9 * len(labels) + 1), sharex=True,
                                 squeeze=False)
        axes = list(axes[:, 0])
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax] * len(labels)
    styles = ["-", "--", "-.", (0, (2, 2)), ":", (0, (5, 1))]
    for i, (label, ax) in enumerate(zip(labels, axes)):
        values = components[label]
        q = np.array([weighted_quantile(values[:, j], [0.16, 0.5, 0.84], weights)
                      for j in range(values.shape[1])]).T
        color = "k" if label == "All" else f"C{i}"
        ls = "-" if stacked or label == "All" else styles[i % len(styles)]
        ax.plot(k, q[1], color=color, ls=ls, lw=2.5, label=label)
        ax.fill_between(k, q[0], q[2], color=color, alpha=0.3, lw=0)
        if stacked or i == 0:
            ax.axhline(0 if "additive" in label else reference, color="0.4", ls=":", lw=1.5)
        if stacked:
            ax.legend(fontsize=fontsize - 4, loc="best")
            if isinstance(ylabel, dict) and label in ylabel:
                ax.set_ylabel(ylabel[label], fontsize=fontsize - 2)
        ax.tick_params(labelsize=fontsize - 4)
    if not stacked:
        axes[0].legend(fontsize=fontsize - 3)
        if ylabel is not None:
            axes[0].set_ylabel(ylabel, fontsize=fontsize)
    elif isinstance(ylabel, str):
        fig.supylabel(ylabel, fontsize=fontsize)
    axes[-1].set_xlabel(K_LABEL, fontsize=fontsize)
    if title:
        fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    return fig

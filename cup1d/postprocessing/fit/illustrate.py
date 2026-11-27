"""What each contaminant does to the fit: residuals after switching terms off.

:func:`residual_scenarios` evaluates the model at one point and redshift with
some contaminant terms switched off (the scenarios), and
:func:`plot_contaminant_residuals` draws the data residual of every scenario
together with the size of the removed term (Fig. 7 of the DR1 paper).
"""

import numpy as np

from cup1d.postprocessing.labels import K_KMS_LABEL
from cup1d.postprocessing.style import styled

# metal transition pairs of cup1d's silicon model
METAL_PAIRS = ("SiIII_Lya", "SiIIa_Lya", "SiIIb_Lya", "SiIIc_Lya", "SiIII_SiIIa", "SiIII_SiIIb",
               "SiIII_SiIIc", "SiIIc_SiIIb", "SiIIc_SiIIa", "SiIIb_SiIIa")
# terms of the baseline model (HCD and the pairs without SiIIc)
BASELINE_TERMS = ("HCD", "SiIII_Lya", "SiIIa_Lya", "SiIIb_Lya", "SiIII_SiIIa", "SiIII_SiIIb",
                  "SiIIb_SiIIa")

# scenario -> (terms switched off, legend of the residual, legend of the removed term)
FIG7_SCENARIOS = {
    "all": ((), "Residual w/ all terms", None),
    "SiII-SiIII": (("SiIII_SiIIa", "SiIII_SiIIb"),
                   r"Residual w/o $\mathrm{SiII}-\mathrm{SiIII}$ term",
                   r"$\mathrm{SiII}-\mathrm{SiIII}$ term (2 params)"),
    "SiII-SiII": (("SiIIb_SiIIa",),
                  r"Residual w/o $\mathrm{SiII}-\mathrm{SiII}$ term",
                  r"$\mathrm{SiII}-\mathrm{SiII}$ term (2 params)"),
    "Lya-SiII": (("SiIIa_Lya", "SiIIb_Lya"),
                 r"Residual w/o $\mathrm{Ly}\alpha-\mathrm{SiII}$ term",
                 r"$\mathrm{Ly}\alpha-\mathrm{SiII}$ term (2 params)"),
    "HCD": (("HCD",), "Residual w/o HCD term", "HCD term (4 params)"),
    "Lya-SiIII": (("SiIII_Lya",),
                  r"Residual w/o $\mathrm{Ly}\alpha-\mathrm{SiIII}$ term",
                  r"$\mathrm{Ly}\alpha-\mathrm{SiIII}$ term (2 params)"),
}


def _without_hcd(like, point):
    """Copy of a physical point with every HCD damping amplitude at its null value."""
    from cup1d.postprocessing.posterior.samples import family_of

    null_vals = like.theory.model_cont.hcd_model.null_vals
    out = dict(point)
    for name in like.free_params:
        family = family_of(name)
        if family.startswith("HCD_damp") and family in null_vals:
            out[name] = null_vals[family]
    return out


def residual_scenarios(like, point, z, scenarios=FIG7_SCENARIOS, key=None,
                       terms=BASELINE_TERMS):
    """Model P1D at one point and redshift with contaminant terms switched off.

    Parameters
    ----------
    like : cup1d.likelihood.likelihood.Likelihood
        Likelihood (data, covariance and forward model).
    point : mapping
        Physical value of every free parameter (e.g. a best fit).
    z : float
        Data redshift.
    scenarios : mapping, default: :data:`FIG7_SCENARIOS`
        Name to ``(terms switched off, residual label, term label)``. Terms
        are metal pairs of :data:`METAL_PAIRS` or ``"HCD"``.
    key : str, optional
        Data set (default: the first one).
    terms : sequence of str, default: :data:`BASELINE_TERMS`
        Terms switched on before a scenario switches some off.

    Returns
    -------
    dict
        ``z``, ``k``, ``data``, ``error`` (diagonal), ``models`` (scenario ->
        model P1D) and ``chi2`` (scenario -> chi2 of the redshift bin, full
        covariance of the bin).
    """
    key = next(iter(like.data)) if key is None else key
    data = like.data[key]
    iz = int(np.argmin(np.abs(np.asarray(data.z) - z)))
    if abs(data.z[iz] - z) > 1e-3:
        raise ValueError(f"no data at z={z} in {key}")
    pk = np.asarray(data.Pk_kms[iz])
    icov = np.asarray(like.icov_Pk_kms[key][iz])
    out = {"z": float(data.z[iz]), "k": np.asarray(data.k_kms[iz]), "data": pk,
           "error": np.sqrt(np.diag(like.cov_Pk_kms[key][iz])), "models": {}, "chi2": {}}
    for name, (off, _, _) in scenarios.items():
        on = [term for term in terms if term not in off]
        values = point if "HCD" in on else _without_hcd(like, point)
        remove = {pair: int(pair in on) for pair in METAL_PAIRS}
        columns = {name_: np.array([value], dtype=float) for name_, value in values.items()}
        predictions, valid, _ = like.predict_batch(columns, remove=remove)
        if not valid[0]:
            raise ValueError("the point is outside the model domain (star priors or training hull)")
        model = np.asarray(predictions[key][iz][0])
        out["models"][name] = model
        diff = pk - model
        out["chi2"][name] = float(diff @ icov @ diff)
    return out


def residual_scenarios_data(result, reference="all"):
    """Arrays of :func:`plot_contaminant_residuals` (as in the Zenodo files).

    ``x`` (k); per scenario ``i``: ``y<i>_blue`` (data/model - 1),
    ``yerr<i>_blue`` and, but for the reference, ``y<i>_orange`` (size of the
    removed term: reference model / scenario model - 1).
    """
    out = {"x": result["k"]}
    base = result["models"][reference]
    for i, (name, model) in enumerate(result["models"].items()):
        out[f"y{i}_blue"] = result["data"] / model - 1
        out[f"yerr{i}_blue"] = result["error"] / model
        if name != reference:
            out[f"y{i}_orange"] = base / model - 1
    return out


@styled
def plot_contaminant_residuals(result, scenarios=FIG7_SCENARIOS, reference="all", fontsize=18,
                               ylims=((-0.12, 0.18), (-0.12, 0.18), (-0.16, 0.28)), ncols=2):
    """Data residual of every scenario and the size of the removed term.

    Parameters
    ----------
    result : mapping
        Output of :func:`residual_scenarios`.
    scenarios : mapping
        The scenarios of ``result`` (for the legends).
    reference : str, default="all"
        Scenario with every term (the removed term is reference/scenario - 1).
    fontsize : float, default=18
        Base font size.
    ylims : sequence of (float, float), optional
        Y limits of the first row, of the third panel and of the last panel
        (those of the DR1 paper figure); ``None`` keeps the automatic ones.
    ncols : int, default=2
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = list(result["models"])
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey="row", figsize=(6 * ncols, 3 * nrows),
                             squeeze=False)
    axes = axes.reshape(-1)
    k = result["k"]
    base = result["models"][reference]
    for ax, name in zip(axes, names):
        model = result["models"][name]
        _, residual_label, term_label = scenarios[name]
        ax.errorbar(k, result["data"] / model - 1, result["error"] / model, color="C0", ls=":",
                    marker=".", label=residual_label)
        if name != reference:
            ax.plot(k, base / model - 1, "C1-", label=term_label)
        ax.text(0.75, 0.07, rf"$\chi^2=${result['chi2'][name]:.1f}", fontsize=fontsize - 2,
                transform=ax.transAxes)
        ax.axhline(0, color="k", ls=":", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.legend(loc="upper right", fontsize=fontsize - 4)
    for ax in axes[len(names):]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel(K_KMS_LABEL, fontsize=fontsize)
    if len(names) % ncols:
        axes[len(names) - 1].tick_params(labelbottom=True)
        axes[len(names) - 1].set_xlabel(K_KMS_LABEL, fontsize=fontsize)
    fig.supylabel("Residual", x=0.01, fontsize=fontsize)
    fig.tight_layout()
    if ylims is not None:
        axes[0].set_ylim(ylims[0])
        if len(names) > 2:
            axes[2].set_ylim(ylims[1])
        axes[len(names) - 1].set_ylim(ylims[2])
    return fig

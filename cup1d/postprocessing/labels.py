"""One registry of the LaTeX labels of the cup1d figures.

Parameter labels come from :data:`cup1d.utils.various_dicts.param_dict`
(shared with the chain products); this module adds the plot-only labels and
is the place every figure takes its labels from.
"""

from cup1d.utils.various_dicts import param_dict

STAR_PARAMETERS = ("Delta2_star", "n_star")

# compressed linear-power parameters (always with \star)
STAR_LABELS = {
    "Delta2_star": r"$\Delta^2_\star$",
    "n_star": r"$n_\star$",
    "alpha_star": r"$\alpha_\star$",
}

# derived IGM histories
IGM_LABELS = {
    "mF": r"$\bar{F}$",
    "tau_eff": r"$\tau_\mathrm{eff}$",
    "T0": r"$T_0\,[10^4\,\mathrm{K}]$",
    "sigT_kms": r"$\sigma_T\,[\mathrm{km\,s^{-1}}]$",
    "gamma": r"$\gamma$",
    "kF_kms": r"$k_F\,[\mathrm{s\,km^{-1}}]$",
}

K_KMS_LABEL = r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$"
Z_LABEL = r"$z$"
KP1D_LABEL = r"$k_\parallel P_{\rm 1D}/\pi$"
RESOLUTION_LABEL = r"$f_\mathrm{res}$"

BLINDED_SUFFIX = " (blinded)"


def parameter_label(name, blinded=False):
    """Return the LaTeX label of a native parameter.

    Parameters
    ----------
    name : str
        Native cup1d name, e.g. ``tau_eff_0`` or ``Delta2_star``.
    blinded : bool, default=False
        Append ``(blinded)`` to the star parameters.

    Returns
    -------
    str
        Label of :data:`STAR_LABELS`, :data:`IGM_LABELS` or
        :data:`cup1d.utils.various_dicts.param_dict`, else the escaped name.
    """
    label = STAR_LABELS.get(name) or param_dict.get(name) or IGM_LABELS.get(name)
    if label is None:
        label = name.replace("_", r"\_")
    if blinded and name in STAR_PARAMETERS:
        label += BLINDED_SUFFIX
    return label

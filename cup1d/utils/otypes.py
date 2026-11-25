"""Output types of coefficient histories and conversions between them.

Every IGM, contaminant and systematic family has an output type (``otype``):

``exp``
    The sampled coefficients are logarithms; the model uses ``exp(coeff)``.
``const``
    The sampled coefficients are used linearly.

Saved fits and initial-condition (IC) files store coefficient values in the
space of the otype they were produced with. They now also record that otype,
and loaders convert to the otype of the current configuration. Files written
before otypes were recorded are assumed to use the CM2026 baseline otypes,
which all configurations used at the time.
"""

from functools import lru_cache

import numpy as np

from cup1d.utils.utils import split_string

OTYPES = ("exp", "const")

# Value assigned when a non-positive linear coefficient is converted to a
# logarithmic one and the caller gives no better floor (e.g. a prior bound).
# Same convention as the "negligible contaminant" fiducial values (exp(-20)).
DEFAULT_LOG_FLOOR = -20.0


def _sections(args):
    """Yield the IGM, contaminant and systematic sections of ``args``."""
    if isinstance(args, dict):
        sections = (args.get("fid_igm"), args.get("fid_cont"), args.get("fid_syst"))
    else:
        sections = (
            getattr(args, "fid_igm", None),
            getattr(args, "fid_cont", None),
            getattr(args, "fid_syst", None),
        )
    for section in sections:
        if section is not None:
            yield section


def history_otype(args, family):
    """Return the otype of a coefficient family in a configuration.

    Parameters
    ----------
    args : cup1d.configuration.args.Args or dict
        Resolved configuration (or a mapping with ``fid_igm``, ``fid_cont``
        and ``fid_syst`` sections).
    family : str
        Family name without index, e.g. ``"HCD_damp1"``.

    Returns
    -------
    str or None
        ``"exp"`` or ``"const"``; ``None`` for names that are not coefficient
        histories (e.g. ``As``, ``ns``).
    """
    for section in _sections(args):
        key = family + "_otype"
        if key in section:
            return section[key]
    return None


def parameter_otypes(args, names):
    """Return ``{name: otype}`` for the coefficient histories among ``names``.

    Parameters
    ----------
    args : cup1d.configuration.args.Args or dict
        Resolved configuration.
    names : iterable of str
        Indexed parameter names, e.g. ``"HCD_damp1_0"``. Names without an
        otype (cosmology) are omitted.
    """
    out = {}
    for name in names:
        otype = history_otype(args, split_string(name)[0])
        if otype is not None:
            out[name] = otype
    return out


@lru_cache(maxsize=None)
def _baseline_otypes():
    """Return the otypes of the CM2026 baseline defaults, by family."""
    from cup1d.configuration import make_cm2026_defaults
    from cup1d.configuration.args import Args

    defaults = make_cm2026_defaults()
    out = dict(Args._KAISER_OTYPES)
    for section in _sections(defaults):
        for key, value in section.items():
            if key.endswith("_otype"):
                out[key[: -len("_otype")]] = value
    return out


def legacy_otype(family):
    """Return the otype assumed for files that do not record it.

    Files written before otypes were stored used the CM2026 baseline otypes.

    Parameters
    ----------
    family : str
        Family name without index.

    Returns
    -------
    str or None
        Baseline otype, or ``None`` if the family is unknown (no conversion).
    """
    return _baseline_otypes().get(family)


def convert_otype(values, from_otype, to_otype, floor=None):
    """Convert coefficient values between output types.

    Parameters
    ----------
    values : float or array_like
        Coefficients in the space of ``from_otype``.
    from_otype, to_otype : {"exp", "const"} or None
        Source and target output types. ``None`` on either side, or equal
        types, return the values unchanged.
    floor : float, optional
        Value used when a non-positive linear coefficient is converted to
        ``exp`` (its logarithm does not exist). Pass the lower prior bound
        when available; defaults to :data:`DEFAULT_LOG_FLOOR`.

    Returns
    -------
    float or numpy.ndarray
        Converted values, with the shape of ``values``.

    Raises
    ------
    ValueError
        If an otype is not ``"exp"`` or ``"const"``.
    """
    scalar = np.ndim(values) == 0
    values = np.array(values, dtype=float)
    if from_otype is None or to_otype is None or from_otype == to_otype:
        out = values
    else:
        for otype in (from_otype, to_otype):
            if otype not in OTYPES:
                raise ValueError(f"Unknown otype {otype!r}; expected one of {OTYPES}")
        if from_otype == "exp":
            out = np.exp(values)
        else:
            floor = DEFAULT_LOG_FLOOR if floor is None else float(floor)
            positive = values > 0
            out = np.full_like(values, floor)
            out[positive] = np.log(values[positive])
            out = np.maximum(out, floor)
    return float(out) if scalar else out


def interp_ic_history(entry, family, znode, otype, floor=None):
    """Interpolate a global-IC history at a node, in the requested otype.

    Parameters
    ----------
    entry : mapping
        Global IC entry ``{"z": nodes, "val": values[, "otype": otype]}``.
    family : str
        Family name, used for :func:`legacy_otype` when ``entry`` has no
        ``"otype"``.
    znode : float
        Redshift node of the coefficient.
    otype : str or None
        Otype of the coefficient in the current configuration.
    floor : float, optional
        See :func:`convert_otype`.

    Returns
    -------
    float
        Interpolated coefficient. The history is interpolated in the space it
        was saved in and then converted.
    """
    value = float(np.interp(znode, entry["z"], entry["val"]))
    saved = entry.get("otype") or legacy_otype(family)
    return convert_otype(value, saved, otype, floor=floor)

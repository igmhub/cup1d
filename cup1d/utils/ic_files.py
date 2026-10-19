"""Naming and lookup of initial-condition (IC) files with baseline fallback."""

import re
from pathlib import Path

from cup1d.utils.utils import get_path_repo

_TYPES = ("global", "at_a_time")


def get_ic_filename(data_label, emulator_label, ic_type, variation=None):
    """Return the standard initial-condition filename.

    Parameters
    ----------
    data_label : str or sequence of str
        P1D data label(s) the initial conditions were fitted to. Several
        labels are joined with ``'_'``.
    emulator_label : str
        Emulator label used in the fit, e.g. ``'lace_mpg'`` or
        ``'forest_mpg_gp'``.
    ic_type : {'global', 'at_a_time'}
        Reduced-global fit or independent-redshift fits.
    variation : str, optional
        Name of the analysis variation. ``None`` or ``'None'`` names the
        baseline of the sample.

    Returns
    -------
    str
        ``{data_label}_{emulator_label}[_{variation}]_ic_{ic_type}.npy``, e.g.
        ``DESIY1_QMLE3_forest_mpg_gp_ic_global.npy`` or
        ``DESIY1_QMLE3_forest_mpg_gp_fixcosmo_ic_global.npy``.

    Raises
    ------
    ValueError
        If ``ic_type`` is not ``'global'`` or ``'at_a_time'``.
    """

    if ic_type not in _TYPES:
        raise ValueError("ic_type must be 'global' or 'at_a_time'")
    if not isinstance(data_label, str):
        data_label = "_".join(data_label)
    middle = f"{data_label}_{emulator_label}"
    if variation not in (None, "None"):
        middle += f"_{variation}"
    return f"{middle}_ic_{ic_type}.npy"


def get_ic_path(data_label, emulator_label, ic_type, variation=None):
    """Return the standard IC file path under ``cup1d/data/ics``.

    Parameters are those of :func:`get_ic_filename`.
    """

    return Path(get_path_repo("cup1d")) / "data" / "ics" / get_ic_filename(
        data_label, emulator_label, ic_type, variation
    )


def ic_type_from_filename(filename, default="global"):
    """Infer ``'global'`` or ``'at_a_time'`` from an IC filename."""

    match = re.search(r"_ic_(global|at_a_time)", Path(filename).name)
    return match.group(1) if match else default


def ic_candidates(args):
    """List IC files to try for an analysis, most specific first.

    Parameters
    ----------
    args : cup1d.configuration.args.Args
        Resolved configuration. ``file_ic`` selects the IC type (from its
        name) and is the last resort; ``None`` disables initial conditions.

    Returns
    -------
    list of pathlib.Path
        1. the variation's own file (only if ``name_variation`` is set),
        2. the baseline file of the same data and emulator labels,
        3. the configured ``file_ic``.
        Duplicates are removed.
    """

    if args.file_ic is None:
        return []
    ic_type = ic_type_from_filename(args.file_ic)
    directory = Path(args.path_ic)
    names = []
    variation = getattr(args, "name_variation", None)
    if variation not in (None, "None"):
        names.append(
            get_ic_filename(args.data_label, args.emulator_label, ic_type, variation)
        )
    names.append(get_ic_filename(args.data_label, args.emulator_label, ic_type))
    candidates = [directory / name for name in names] + [Path(args.file_ic)]
    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def resolve_ic_file(args):
    """Return the first existing IC file for an analysis, or ``None``.

    See :func:`ic_candidates` for the order: the variation's own file, then
    the baseline of the same sample and emulator, then ``file_ic``.
    """

    for path in ic_candidates(args):
        if path.is_file():
            return path
    return None

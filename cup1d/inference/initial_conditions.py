"""Generate initial conditions from independent-redshift minimizations."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.show_results import print_results
from cup1d.utils.utils import get_path_repo


def get_ic_filename(data_label, emulator_label, ic_type):
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

    Returns
    -------
    str
        ``{data_label}_{emulator_label}_ic_{ic_type}.npy``, e.g.
        ``DESIY1_QMLE3_forest_mpg_gp_ic_global.npy``.

    Raises
    ------
    ValueError
        If ``ic_type`` is not ``'global'`` or ``'at_a_time'``.
    """

    if ic_type not in ("global", "at_a_time"):
        raise ValueError("ic_type must be 'global' or 'at_a_time'")
    if not isinstance(data_label, str):
        data_label = "_".join(data_label)
    return f"{data_label}_{emulator_label}_ic_{ic_type}.npy"


def get_at_a_time_ic_path(data_label, emulator_label):
    """Return the standard per-redshift initial-condition file path.

    Parameters
    ----------
    data_label : str or sequence of str
        P1D data label(s) the initial conditions were fitted to.
    emulator_label : str
        Emulator label used in the fit.

    Returns
    -------
    pathlib.Path
        ``cup1d/data/ics/{data_label}_{emulator_label}_ic_at_a_time.npy``.
    """

    return Path(get_path_repo("cup1d")) / "data" / "ics" / get_ic_filename(
        data_label, emulator_label, "at_a_time"
    )


def get_global_ic_path(data_label, emulator_label):
    """Return the standard reduced-global initial-condition file path.

    Parameters
    ----------
    data_label : str or sequence of str
        P1D data label(s) the initial conditions were fitted to.
    emulator_label : str
        Emulator label used in the fit.

    Returns
    -------
    pathlib.Path
        ``cup1d/data/ics/{data_label}_{emulator_label}_ic_global.npy``.
    """

    return Path(get_path_repo("cup1d")) / "data" / "ics" / get_ic_filename(
        data_label, emulator_label, "global"
    )


def generate_at_a_time_initial_conditions(
    config_path,
    output_path=None,
    overwrite=False,
    verbose=True,
):
    """Fit each P1D redshift bin and save the resulting initial conditions.

    Parameters
    ----------
    config_path : path-like
        YAML configuration for an at-a-time global analysis.
    output_path : path-like, optional
        Destination ``.npy`` file. Defaults to the at-a-time IC file for the
        configured data and emulator labels under ``data/ics``.
    overwrite : bool, optional
        Replace an existing output file.
    verbose : bool, optional
        Print fitted redshifts and the final goodness-of-fit table.

    Returns
    -------
    pathlib.Path or None
        Saved file on MPI rank zero; ``None`` on worker ranks.

    Raises
    ------
    ValueError
        If the YAML does not select ``at_a_time_global`` fitting.
    FileExistsError
        If the destination exists and ``overwrite`` is false.

    Notes
    -----
    A fresh :class:`Analysis` is constructed for every redshift bin so the
    generated starting points follow the normal at-a-time workflow.
    """

    args = Args.from_yaml(config_path, verbose=False)
    if args.fit_type != "at_a_time_global":
        raise ValueError(
            "Initial-condition generation requires fit_type: at_a_time_global"
        )

    output_path = (
        get_at_a_time_ic_path(args.data_label, args.emulator_label)
        if output_path is None
        else Path(output_path)
    )
    output_path = output_path.expanduser()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0 and output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it."
        )

    # This analysis supplies the P1D redshift grid. A fresh analysis is made
    # for every local fit, matching the current tutorial workflow.
    grid_analysis = Analysis(args)
    data = next(iter(grid_analysis.data.values()))
    redshifts = np.asarray(data.z)

    output = {"z": redshifts, "pnames": [], "mle_cube": [], "mle": [], "chi2": []}
    final_analysis = None

    for index, redshift in enumerate(redshifts):
        if rank == 0 and verbose:
            print(f"Fitting redshift bin {index}: z = {redshift:.2f}", flush=True)

        local_analysis = Analysis(args)
        initial_point = local_analysis.fitter.initial_parameters()
        local_analysis.run_minimizer(
            initial_point,
            zmask=np.asarray([redshift]),
            restart=True,
        )
        final_analysis = local_analysis

        if rank == 0:
            output["pnames"].append(list(local_analysis.like.free_param_names))
            output["mle_cube"].append(local_analysis.fitter.mle_cube.copy())
            output["mle"].append(dict(local_analysis.fitter.mle))
            output["chi2"].append(local_analysis.fitter.mle_chi2)

    if rank != 0:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output)
    if verbose:
        print(f"Saved initial conditions to {output_path}", flush=True)
        print_results(final_analysis.like, output["chi2"], output["mle_cube"])
    return output_path


def generate_global_initial_conditions(
    config_path,
    output_path=None,
    overwrite=False,
    verbose=True,
):
    """Run the reduced global fit and save its initial conditions.

    Parameters
    ----------
    config_path : path-like
        YAML configuration for the reduced global optimization.
    output_path : path-like, optional
        Destination ``.npy`` file. Defaults to the global IC file for the
        configured data and emulator labels under ``data/ics``.
    overwrite : bool, default=False
        Replace an existing destination file.
    verbose : bool, default=True
        Print fit and output summaries on MPI rank zero.

    Returns
    -------
    pathlib.Path or None
        Saved file on MPI rank zero; ``None`` on worker ranks.

    Raises
    ------
    ValueError
        If the YAML is not a ``global_opt`` setup or it consumes an existing
        global IC file.
    FileExistsError
        If the destination exists and ``overwrite`` is false.

    Notes
    -----
    The associated YAML must set ``file_ic: null``. This is the YAML-native
    counterpart of the legacy ``ic_global=False`` option: a global IC file is
    being created, so it must not first be used as input.
    """

    args = Args.from_yaml(config_path, verbose=False)
    if args.fit_type != "global_opt":
        raise ValueError(
            "Global IC generation requires fit_type: global_opt"
        )
    if args.file_ic is not None:
        raise ValueError(
            "Global IC generation requires file_ic: null so an existing "
            "global IC is not used as input"
        )

    output_path = (
        get_global_ic_path(args.data_label, args.emulator_label)
        if output_path is None
        else Path(output_path)
    )
    output_path = output_path.expanduser()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0 and output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it."
        )

    analysis = Analysis(args)
    initial_point = analysis.fitter.initial_parameters()
    if rank == 0 and verbose:
        print("Running reduced global fit to generate global ICs", flush=True)
    analysis.run_minimizer(initial_point, restart=True)

    if rank != 0:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.save_global_ic(output_path)
    if verbose:
        print(f"Saved global initial conditions to {output_path}", flush=True)
        print(f"chi2 = {analysis.fitter.mle_chi2:.3f}", flush=True)
    return output_path

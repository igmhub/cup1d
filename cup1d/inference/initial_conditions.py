"""Generate initial conditions from independent-redshift minimizations."""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.show_results import print_results
from cup1d.utils.utils import get_path_repo


def get_at_a_time_ic_path(emulator_label):
    """Return the standard initial-condition path for an emulator family."""

    emulator_family = "nyx" if "nyx" in emulator_label.lower() else "mpg"
    return Path(get_path_repo("cup1d")) / "data" / "ics" / (
        f"{emulator_family}_ic_at_a_time.npy"
    )


def get_global_ic_path(emulator_label):
    """Return the standard global-fit initial-condition path."""

    emulator_family = "nyx" if "nyx" in emulator_label.lower() else "mpg"
    return Path(get_path_repo("cup1d")) / "data" / "ics" / (
        f"{emulator_family}_ic_global_red.npy"
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
        Destination ``.npy`` file. By default this is the standard MPG or Nyx
        at-a-time IC file under ``data/ics``.
    overwrite : bool, optional
        Replace an existing output file.
    verbose : bool, optional
        Print fitted redshifts and the final goodness-of-fit table.
    """

    args = Args.from_yaml(config_path, verbose=False)
    if args.fit_type != "at_a_time_global":
        raise ValueError(
            "Initial-condition generation requires fit_type: at_a_time_global"
        )

    output_path = (
        get_at_a_time_ic_path(args.emulator_label)
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
        initial_point = local_analysis.fitter.sampling_point_from_parameters().copy()
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
        get_global_ic_path(args.emulator_label)
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
    initial_point = analysis.fitter.sampling_point_from_parameters().copy()
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

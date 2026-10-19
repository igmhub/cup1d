import argparse
import os
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
from mpi4py import MPI
from cup1d.configuration import Args
from cup1d.inference import Analysis
from cup1d.postprocessing.plots_corner import plots_chain
from cup1d.utils.utils import get_path_repo


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the minimizer and sampler for one configuration."
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        default=None,
        help="YAML file (absolute or relative to the cup1d repository) or "
        "CM2026 variation name; omitted or 'None' runs the baseline.",
    )
    parser.add_argument(
        "--path_out",
        default=None,
        help="Root output directory. Results are written to "
        "<path_out>/<data_label>/<name_variation or fit_type>/<emulator_label>/chain_N, "
        "ignoring out_folder from the configuration.",
    )
    return parser.parse_args()


def main():
    ## MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    arguments = parse_arguments()
    options = {}
    if arguments.path_out is not None:
        # out_folder: null derives the folder structure below path_out
        options = {"path_out": arguments.path_out, "out_folder": None}

    configuration = arguments.configuration
    if configuration is not None and configuration.endswith(".yaml"):
        config_path = Path(configuration)
        if not config_path.is_absolute():
            config_path = Path(get_path_repo("cup1d")) / config_path
        args = Args.from_yaml(config_path, verbose=(rank == 0), **options)
    else:
        args = Args.from_variation(configuration, **options)

    analysis = Analysis(args, out_folder=args.out_folder)
    input_pars = analysis.fitter.initial_parameters()

    # initial_sampling_values are unit-cube coordinates keyed by parameter name
    for name, value in args.initial_sampling_values.items():
        if name not in input_pars:
            raise ValueError(
                f"Initial sampling value provided for unknown parameter {name}"
            )
        input_pars[name]["value"] = analysis.fitter.value_from_cube(name, value)

    analysis.run_minimizer(input_pars, restart=True)
    analysis.run_sampler()

    if rank == 0:
        plots_chain(
            analysis.fitter.save_directory,
            folder_out=analysis.fitter.save_directory,
        )


if __name__ == "__main__":
    main()

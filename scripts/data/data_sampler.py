import argparse
import os
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
from mpi4py import MPI
from cup1d.configuration import Args
from cup1d.inference import Analysis
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
        "--stage",
        choices=("all", "minimize", "sample"),
        default="all",
        help="'minimize': only the minimizer (rank 0; run with one task); "
        "'sample': only emcee, starting from the minimizer result given by "
        "--results and writing to its folder; 'all' (default): both.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="With --stage sample: minimizer_results.npy, or the chain_N "
        "folder holding it, written by a --stage minimize run.",
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

    if arguments.stage == "sample":
        run_sampler_from_results(arguments.results, rank)
        return

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
    # restart from this configuration's own latest fit when there is one
    # (earlier chain_N folder of the same out_folder); the IC file and
    # initial_sampling_values are only the fallback
    previous = latest_saved_fit(analysis.fitter.save_directory)
    if previous is not None:
        if rank == 0:
            print(f"Starting from the saved fit {previous}", flush=True)
        input_pars = analysis.fitter.initial_parameters_from_results(previous)
        initial_values = {}
    else:
        input_pars = analysis.fitter.initial_parameters()
        initial_values = args.initial_sampling_values

    # initial_sampling_values are unit-cube coordinates keyed by parameter name
    for name, value in initial_values.items():
        if name not in input_pars:
            raise ValueError(
                f"Initial sampling value provided for unknown parameter {name}"
            )
        input_pars[name]["value"] = analysis.fitter.value_from_cube(name, value)

    # Gauss-Newton errors: saved with the fit, and the walkers of the sampler start from them
    analysis.run_minimizer(input_pars, restart=True, estimate_errors=True)
    if arguments.stage == "minimize":
        if rank == 0:
            print(
                f"Minimizer done: chi2 = {analysis.fitter.mle_chi2:.2f}, "
                f"results in {analysis.fitter.save_directory}",
                flush=True,
            )
        return
    analysis.run_sampler()

    if rank == 0:
        chain_products(analysis)


def chain_products(analysis):
    """Contours, summary, log-probability traces and posterior figures of a chain."""
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from cup1d.postprocessing.chains.products import write_chain_products
    from cup1d.postprocessing.io import save_figure
    from cup1d.postprocessing.posterior import PosteriorPlotter
    from cup1d.postprocessing.primitives import plot_traces

    folder = analysis.fitter.save_directory
    write_chain_products(folder)
    lnprob = np.load(os.path.join(folder, "lnprob.npy"))
    save_figure(plot_traces(lnprob, ylabel=r"$\ln P$", show_mean=True), os.path.join(folder, "lnprob"),
                close=True)
    PosteriorPlotter.from_analysis(analysis).plot_all()


def latest_saved_fit(save_directory):
    """Newest result of an earlier ``chain_N`` sibling of ``save_directory``.

    Returns the ``minimizer_results.npy`` (or ``sampler_results.npy``) of the
    highest-numbered earlier ``chain_N`` folder that has one, or ``None``.
    """
    current = Path(save_directory)
    folders = []
    for folder in current.parent.glob("chain_*"):
        suffix = folder.name.removeprefix("chain_")
        if folder != current and suffix.isdigit():
            folders.append((int(suffix), folder))
    for _, folder in sorted(folders, reverse=True):
        for name in ("minimizer_results.npy", "sampler_results.npy"):
            if (folder / name).is_file():
                return folder / name
    return None


def run_sampler_from_results(results, rank):
    """Run emcee from a saved minimizer result, writing next to it."""
    if results is None:
        raise ValueError("--stage sample needs --results")
    results = Path(results)
    if results.is_dir():
        results = results / "minimizer_results.npy"
    if not results.is_file():
        raise FileNotFoundError(f"No minimizer result at {results}")

    # the configuration is the one recorded in the result; the sampler starts
    # at its best fit and saves to the same chain_N folder
    analysis = Analysis.from_results(results)
    if rank == 0:
        print(
            f"Sampling from {results} (chi2 = {analysis.fitter.mle_chi2:.2f})",
            flush=True,
        )
    analysis.run_sampler()

    if rank == 0:
        chain_products(analysis)


if __name__ == "__main__":
    main()

"""Run a sequence of YAML-configured Cup1D samplers under MPI."""

import argparse
from pathlib import Path

from mpi4py import MPI

from cup1d import Args
from cup1d.utils.utils import get_path_repo
from sam_sim import run


REPOSITORY = Path(get_path_repo("cup1d")).resolve()
DEFAULT_CONFIGS = sorted((REPOSITORY / "configs" / "mocks").glob("**/*.yaml"))


def _output_directory(config_path):
    args = Args.from_yaml(config_path, synthetic=True, verbose=False)
    output = Path(args.out_folder).expanduser()
    if not output.is_absolute():
        output = REPOSITORY / output
    return output.resolve()


def _completed(output_directory):
    """Return whether any chain directory contains completed sampler results."""

    return any(output_directory.glob("chain_*/sampler_results.npy"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=DEFAULT_CONFIGS,
        help="synthetic YAML configurations (default: every configs/mocks YAML)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even when sampler_results.npy already exists",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="do not minimize from the best point in each completed chain",
    )
    return parser.parse_args()


def main():
    options = parse_args()
    if not options.configs:
        raise ValueError("No mock YAML configurations were found or provided")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for requested_path in options.configs:
        config_path = requested_path.expanduser().resolve()
        if rank == 0:
            output_directory = _output_directory(config_path)
            should_run = options.force or not _completed(output_directory)
            message = "Running" if should_run else "Skipping completed run"
            print(f"{message}: {config_path}")
        else:
            should_run = None
        should_run = comm.bcast(should_run, root=0)
        if should_run:
            run(config_path, refine=not options.no_refine)
        comm.Barrier()

    if rank == 0:
        print("Finished all sampler configurations")


if __name__ == "__main__":
    main()

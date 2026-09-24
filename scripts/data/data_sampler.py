import os
import sys
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
from mpi4py import MPI
from cup1d.configuration import Args
from cup1d.inference import Analysis
from cup1d.postprocessing.plots_corner import plots_chain
from cup1d.utils.utils import get_path_repo


def main():
    ## MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    configuration = sys.argv[1] if len(sys.argv) > 1 else None
    if configuration is not None and configuration.endswith(".yaml"):
        config_path = Path(configuration)
        if not config_path.is_absolute():
            config_path = Path(get_path_repo("cup1d")) / config_path
        args = Args.from_yaml(config_path)
    else:
        args = Args.from_variation(configuration)

    analysis = Analysis(args, out_folder=args.out_folder)
    input_pars = analysis.fitter.sampling_point_from_parameters().copy()

    for name, value in args.initial_sampling_values.items():
        matches = [
            index
            for index, parameter in enumerate(analysis.like.free_params)
            if parameter.name == name
        ]
        if not matches:
            raise ValueError(
                f"Initial sampling value provided for unknown parameter {name}"
            )
        input_pars[matches[0]] = value

    analysis.run_minimizer(input_pars, restart=True)
    analysis.run_sampler()

    if rank == 0:
        plots_chain(
            analysis.fitter.save_directory,
            folder_out=analysis.fitter.save_directory,
        )


if __name__ == "__main__":
    main()

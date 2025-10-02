import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from mpi4py import MPI
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo
from cup1d.plots.plots_corner import plots_chain


def main():
    ## MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    emu = "mpg"
    data_label = "DESIY1_QMLE3"
    name_variation = sys.argv[1]

    if name_variation == "nyx":
        name_variation = None
        emu = "nyx"
    elif name_variation == "DESIY1_QMLE":
        name_variation = None
        data_label = "DESIY1_QMLE"
    elif name_variation == "DESIY1_FFT3_dir":
        name_variation = None
        data_label = "DESIY1_FFT3_dir"
    elif name_variation == "DESIY1_FFT_dir":
        name_variation = None
        data_label = "DESIY1_FFT_dir"
    elif name_variation == "None":
        name_variation = None

    args = Args(
        data_label=data_label,
        emulator_label="CH24_" + emu + "cen_gpr",
    )
    args.set_baseline(
        fit_type="global_opt",
        fix_cosmo=True,
        P1D_type=data_label,
        name_variation=name_variation,
        mcmc_conf="explore",
        # mcmc_conf="test",
    )
    pip = Pipeline(args, out_folder=args.out_folder)
    input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

    pip.run_minimizer(input_pars, restart=True)
    pip.run_sampler()

    if rank == 0:
        plots_chain(
            pip.fitter.save_directory, folder_out=pip.fitter.save_directory
        )


if __name__ == "__main__":
    main()

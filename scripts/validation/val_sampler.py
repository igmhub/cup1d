import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from mpi4py import MPI
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo
from cup1d.plots_and_tables.plots_corner import plots_chain


def main():
    ## MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    emu = "mpg"
    fit_type = "global_opt"
    mcmc_conf = "explore"
    # path_data = "jjchaves"
    path_data = "nersc"
    cov_label = "DESIY1_QMLE3"

    # baseline
    if sys.argv[1].endswith("igm") | sys.argv[1].endswith("igm0"):
        data_label = "mpg_central"
        name_variation = "sim_" + sys.argv[1]
    else:
        data_label = sys.argv[1]
        name_variation = "sim_" + data_label

    zmin = 2.2
    zmax = 4.2

    args = Args(
        data_label=data_label,
        cov_label=cov_label,
        emulator_label="CH24_" + emu + "cen_gpr",
        true_cosmo_label=data_label,
        apply_smoothing=True,
        add_noise=False,
        seed_noise=0,
        emu_cov_type="full",
    )

    args.set_baseline(
        fit_type=fit_type,
        fix_cosmo=False,
        fid_cosmo_label=data_label,
        P1D_type=cov_label,
        name_variation=name_variation,
        z_min=zmin,
        z_max=zmax,
        mcmc_conf=mcmc_conf,
    )

    if data_label == "accel2":
        if path_data == "jjchaves":
            args.path_data = (
                "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid"
            )
        elif path_data == "nersc":
            args.path_data = "/global/cfs/cdirs/desi/users/ravouxco/accel2/shared_files/frontier_grid"
        else:
            raise ValueError(
                "path_data not defined for data_label = " + data_label
            )

    pip = Pipeline(args)

    input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

    print("starting minimization")
    pip.run_minimizer(input_pars, restart=True)
    pip.run_sampler()

    if rank == 0:
        plots_chain(
            pip.fitter.save_directory, folder_out=pip.fitter.save_directory
        )


if __name__ == "__main__":
    main()

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
import numpy as np
from mpi4py import MPI
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    ## MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    emu = "mpg"
    # emu = "nyx"
    # fit_type = "global_igm"
    fit_type = "global_opt"
    mcmc_conf = "explore"
    path_data = "jjchaves"

    # baseline
    if sys.argv[1].endswith("igm") | sys.argv[1].endswith("igm0"):
        data_label = "mpg_central"
        name_variation = "sim_" + sys.argv[1]
    else:
        data_label = sys.argv[1]
        name_variation = "sim_" + data_label

    if data_label == "mpg_central":
        zmin = 2.25
        zmax = 4.25
    else:
        zmin = 2.2
        zmax = 4.2

    cov_label = "DESIY1_QMLE3"

    args = Args(
        data_label=data_label,
        cov_label=cov_label,
        emulator_label="CH24_" + emu + "cen_gpr",
        path_data=path_data,
        true_cosmo_label=data_label,
        fid_cosmo_label=data_label,
        apply_smoothing=True,
        add_noise=False,
        seed_noise=0,
    )

    args.set_baseline(
        fit_type=fit_type,
        fix_cosmo=False,
        P1D_type=cov_label,
        name_variation=name_variation,
        z_min=zmin,
        z_max=zmax,
        mcmc_conf=mcmc_conf,
    )
    pip = Pipeline(args, out_folder=args.out_folder)

    input_pars = pip.fitter.like.sampling_point_from_parameters().copy()

    print("starting minimization")
    # type_minimizer = "NM"
    # if type_minimizer == "NM":
    #     pip.fitter.run_minimizer(
    #         pip.fitter.like.minus_log_prob,
    #         p0=input_pars,
    #         restart=True,
    #         # burn_in=True,
    #     )
    # else:
    #     pip.fitter.run_minimizer_da(
    #         pip.fitter.like.minus_log_prob, p0=input_pars, restart=True
    #     )
    pip.run_minimizer(input_pars, restart=True)
    pip.run_sampler()

    if rank == 0:
        plots_chain(
            pip.fitter.save_directory, folder_out=pip.fitter.save_directory
        )

    # out_dict = {
    #     "best_chi2": pip.fitter.mle_chi2,
    #     "mle_cosmo_cen": pip.fitter.mle_cosmo,
    #     "mle_cube": pip.fitter.mle_cube,
    #     "mle": pip.fitter.mle,
    # }

    # file_out = os.path.join(args.out_folder, "best_dircosmo.npy")

    # print("saving output to:", file_out)
    # np.save(file_out, out_dict)

    # pip.fitter.save_fitter()


if __name__ == "__main__":
    main()

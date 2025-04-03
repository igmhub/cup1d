import os, socket

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI

from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    """Launch forecast for DESI-Y1"""

    version = "1.0fxh"
    name_system = socket.gethostname()
    if "login" in name_system:
        path_p1d_mock = [
            os.path.sep,
            "global",
            "cfs",
            "cdirs",
            "desicollab",
            "science",
            "lya",
            "y1-p1d",
            "likelihood_files",
            "data_files",
            "MockChallengeSnapshot",
            "mockchallenge-" + version,
            "mockchallenge-" + version + "_nonoise_fiducial.fits.gz",
        ]
        path_p1d_mock = os.path.join(*path_p1d_mock)
    else:
        path_p1d_mock = os.path.join(
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "mock_challenge",
            "MockChallengeSnapshot",
            "mockchallenge-" + version,
            "mockchallenge-" + version + "_nonoise_fiducial.fits.gz",
        )

    data_like = "mock_challenge_DESIY1"

    # emulator_label = "CH24_mpg_gp"
    emulator_label = "CH24_nyx_gp"
    # emulator_label = "CH24"
    # emulator_label = "CH24_NYX"
    # emu_cov_factor = None
    emu_cov_factor = 1.0
    # emu_cov_type = "diagonal"
    emu_cov_type = "block"
    # emu_cov_type = "full"

    if emu_cov_factor is None:
        lab_err = "no_emu_err"
    else:
        if emu_cov_type == "diagonal":
            lab_err = "emu_err_diag"
        elif emu_cov_type == "block":
            lab_err = "emu_err_block"
        else:
            lab_err = "emu_err_full"

    base_out_folder = os.path.join(
        os.path.dirname(get_path_repo("cup1d")),
        "data",
        "forecast",
        data_like + "_" + version,
        emulator_label,
        lab_err,
    )

    run_forecast(
        emulator_label,
        base_out_folder,
        data_like,
        path_p1d_mock,
        emu_cov_factor,
        emu_cov_type,
    )


def run_forecast(
    emulator_label,
    base_out_folder,
    data_like,
    path_p1d_mock,
    emu_cov_factor,
    emu_cov_type,
):
    """Validate assuming a fiducial cosmology against forecast data"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set fiducial args
    args = Args(emulator_label=emulator_label)
    args.emu_cov_factor = emu_cov_factor
    args.emu_cov_type = emu_cov_type

    args.n_steps = 1000
    args.n_burn_in = 500
    # args.n_steps = 1000
    # args.n_burn_in = 100
    if size > 1:
        args.parallel = True
    else:
        args.parallel = False

    if args.n_burn_in == 0:
        args.explore = True
    else:
        args.explore = False

    # set emulator
    args.emulator = set_emulator(emulator_label=args.emulator_label)

    # set covariance matrix
    args.data_label = data_like
    args.p1d_fname = path_p1d_mock

    # set true igm and cosmo
    if "mpg" in emulator_label:
        true_sim = "mpg_central"
    elif "nyx" in emulator_label:
        true_sim = "nyx_central"
    else:
        true_sim = "mpg_central"

    args.true_label_mF = true_sim
    args.true_label_T = true_sim
    args.true_label_kF = true_sim
    args.true_cosmo_label = true_sim
    # args.true_label_mF = "mpg_central"
    # args.true_label_T = "mpg_central"
    # args.true_label_kF = "kF_both"
    # args.true_cosmo_label = "Planck18_low"

    # set fiducial igm and cosmo
    args.mF_model_type = "chunks"
    args.n_tau = 11  # XXX use number of z by default
    args.n_sigT = 1
    args.n_gamma = 1
    args.n_kF = 1

    args.fid_label_mF = true_sim
    args.fid_label_T = true_sim
    args.fid_label_kF = true_sim
    args.fid_cosmo_label = true_sim
    # args.fid_label_mF = "mpg_central"
    # args.fid_label_T = "mpg_central"
    # args.fid_label_kF = "kF_both"
    # args.fid_cosmo_label = "Planck18_low"

    # set number of free IGM parameters

    # set systematics
    ## SiIII
    args.n_x_SiIII = 1
    args.n_d_SiIII = 1
    args.n_a_SiIII = 1
    args.fid_SiIII_X = [0, -10]
    args.fid_SiIII_D = [0, 5]
    args.fid_SiIII_A = [0, 1]
    ## HCD
    args.hcd_model_type = "new"
    args.n_d_dla = 1
    args.n_s_dla = 1
    args.fid_A_damp = [0, -9]
    args.fid_A_scale = [0, 5]

    # redshift range
    args.z_min = 2.1
    args.z_max = 4.3

    # not varying alpha_s for the time being
    args.vary_alphas = False
    # if "Nyx" in emulator_label:
    #     args.vary_alphas = True
    # else:
    #     args.vary_alphas = False

    pip = Pipeline(args, make_plots=False, out_folder=base_out_folder)
    # run minimizer on fiducial (may not get to minimum)
    p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    ## XXX do this by default if p0 is not provided!!
    pip.run_minimizer(p0)
    # pip.fitter.mle_cube = p0

    # run sampler, it uses as ini the results of the minimizer
    pip.run_sampler()

    # run minimizer again, now on MLE
    if rank == 0:
        pip.run_minimizer(pip.fitter.mle_cube, save_chains=True)


if __name__ == "__main__":
    main()

# //global/cfs/cdirs/desicollab/science/lya/y1-p1d/likelihood_files/data_files/MockChallengeSnapshot

import socket, os, sys, glob

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from mpi4py import MPI
from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline, set_cosmo
from cup1d.likelihood import CAMB_model
from cup1d.utils.utils import get_path_repo


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    version = "1.9fsh"
    # version = "1.10qsh"
    run_sampler = True

    name_system = socket.gethostname()
    if "login" in name_system:
        path_in_challenge = [
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
        ]
        path_in_challenge = os.path.join(*path_in_challenge)
        path_out_challenge = os.path.join(
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "mock_challenge",
            "v" + version,
        )
    else:
        path_in_challenge = os.path.join(
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "mock_challenge",
            "MockChallengeSnapshot",
            "mockchallenge-" + version,
        )
        path_out_challenge = os.path.join(
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "mock_challenge",
            "v" + version,
        )

    print(path_in_challenge)
    print(path_out_challenge)

    search = os.path.join(
        path_in_challenge,
        "*" + version + "_nonoise_fiducial*"
        # "*CGAN_4096_base*",
        # "*CGAN_4096_val*",
        # "*ACCEL2_6144_160*",
        # "*Sherwood_2048_40*",
        # "*cosmo_grid_3*",
    )

    # print(search)
    files = np.sort(glob.glob(search))
    if rank == 0:
        for ii in range(len(files)):
            print(ii, files[ii])

    # sys.exit()

    # include all contaminants in the fit or not
    full_cont = True  # IMPORTANT!!!

    # emulator_label = "Pedersen23_ext"
    # emulator_label = "CH24"
    # training_set = "Cabayol23"
    # vary_alphas = False

    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
    # vary_alphas = False

    # emulator_label = "Nyx_alphap_cov"
    # training_set = "Nyx23_Jul2024"
    # # vary_alphas = True
    # vary_alphas = False

    # args = Args(emulator_label=emulator_label, training_set=training_set)

    # emulator_label = "CH24_mpg_gp"
    # emulator_label = "CH24_nyx_gp"
    emulator_label = "CH24_nyxcen_gpr"
    vary_alphas = False

    args = Args(emulator_label=emulator_label)
    args.data_label = "challenge_DESIY1"

    impose_fid_cosmo_label = None
    # impose_fid_cosmo_label = "Planck18"
    # impose_fid_cosmo_label = "Planck18_h74"

    # note redshift range!
    args.z_min = 2.1
    args.z_max = 4.3

    # args.emu_cov_factor = None
    args.emu_cov_factor = 1.0
    # args.emu_cov_type = "diagonal"
    # args.emu_cov_type = "block"
    args.emu_cov_type = "full"

    args.n_steps = 2500
    if "Sherwood_2048_40" in files[0]:
        args.n_burn_in = 2500
    elif "CGAN" in files[0]:
        args.n_burn_in = 2500
    else:
        # args.n_burn_in = 1250
        args.n_burn_in = 750

    if size > 1:
        args.parallel = True
    else:
        args.parallel = False

    if args.n_burn_in == 0:
        args.explore = True
    else:
        args.explore = False

    base_out_folder = os.path.join(path_out_challenge, emulator_label)

    # set number of free IGM parameters
    args.mF_model_type = "chunks"
    # I set it below so it is equal to number of z
    args.n_tau = 11
    args.n_sigT = 2
    args.n_gamma = 2
    args.n_kF = 1
    if full_cont:
        args.n_x_SiIII = 1
        args.n_d_SiIII = 1
        args.n_a_SiIII = 1
        args.n_d_dla = 1
        args.n_s_dla = 1
        args.fid_SiIII_X = [0, -10]
        args.fid_SiIII_D = [0, 5]
        args.fid_SiIII_A = [0, 1]
        args.fid_A_damp = [0, -9]
        args.fid_A_scale = [0, 5]
        args.hcd_model_type = "new"

    # set archive and emulator
    if rank == 0:
        if emulator_label not in [
            "CH24_mpg_gp",
            "CH24_nyx_gp",
            "CH24_nyx_gpr",
            "CH24_nyxcen_gpr",
        ]:
            args.archive = set_archive(args.training_set)
            args.emulator = set_emulator(
                emulator_label=args.emulator_label,
                archive=args.archive,
            )
            if "Nyx" in emulator_label:
                args.emulator.list_sim_cube = args.archive.list_sim_cube
                if "nyx_14" in args.emulator.list_sim_cube:
                    args.emulator.list_sim_cube.remove("nyx_14")
            else:
                args.emulator.list_sim_cube = args.archive.list_sim_cube
        else:
            args.emulator = set_emulator(emulator_label=args.emulator_label)

    if ("Nyx" in emulator_label) and vary_alphas:
        args.vary_alphas = True
    else:
        args.vary_alphas = False

    for isim in range(len(files)):
        if "ACCEL2_6144_160" in files[isim]:
            args.n_tau = 4
        else:
            args.n_tau = 11

        args.p1d_fname = files[isim]
        if rank == 0:
            print("Analyzing:", args.p1d_fname)
        file_sim = os.path.basename(args.p1d_fname)[:-8]

        if args.emu_cov_factor is None:
            lab_err = ""
        else:
            if args.emu_cov_type == "diagonal":
                lab_err = "_emu_err_diag"
            elif args.emu_cov_type == "block":
                lab_err = "_emu_err_block"
            else:
                lab_err = "_emu_err_full"

        dir_out = os.path.join(base_out_folder + lab_err, file_sim)
        if rank == 0:
            os.makedirs(dir_out, exist_ok=True)
            print("Output in:", dir_out)

        # same true and fiducial IGM
        if "fiducial" in args.p1d_fname:
            true_sim_label = "nyx_central"
        elif "CGAN_4096_base" in args.p1d_fname:
            true_sim_label = "nyx_seed"
        elif "CGAN_4096_val" in args.p1d_fname:
            true_sim_label = "nyx_seed_val"
        elif "grid_3" in args.p1d_fname:
            true_sim_label = "nyx_3_ic"
        elif "Sherwood_2048_40" in args.p1d_fname:
            true_sim_label = "sherwood"
        elif "ACCEL2_6144_160" in args.p1d_fname:
            true_sim_label = "accel2"
        else:
            print("Missing true sim label!!")
            sys.exit()

        args.true_cosmo_label = true_sim_label
        args.true_igm_label = true_sim_label
        args.true_label_mF = true_sim_label
        args.true_label_T = true_sim_label
        args.true_label_kF = true_sim_label

        if ("Nyx" in emulator_label) | ("nyx" in emulator_label):
            fid_sim_label = "nyx_central"
        else:
            fid_sim_label = "mpg_central"

        args.fid_label_mF = fid_sim_label
        args.fid_label_T = fid_sim_label
        args.fid_label_kF = fid_sim_label
        if impose_fid_cosmo_label is not None:
            args.fid_cosmo_label = impose_fid_cosmo_label
        else:
            args.fid_cosmo_label = fid_sim_label

        fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

        # Planck18 0.354 -2.300 -0.2155
        # 5 sigma 0.056 0.011 0.0028
        # args.use_star_priors = None
        # blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params()
        # args.use_star_priors = {}
        # args.use_star_priors["alpha_star"] = [
        #     blob["alpha_star"] - 0.0028,
        #     blob["alpha_star"] + 0.0028,
        # ]

        if "bar_ic" in args.p1d_fname:
            args.ic_correction = True
            # args.ic_correction = False
        else:
            args.ic_correction = False

        pip = Pipeline(args, make_plots=False, out_folder=dir_out)

        # run minimizer on fiducial (may not get to minimum)
        p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
        pip.run_minimizer(p0)

        if run_sampler:
            # run samplers, it uses as ini the results of the minimizer
            pip.run_sampler()

            # run minimizer again, now on MLE
            if rank == 0:
                pip.run_minimizer(pip.fitter.mle_cube, save_chains=True)


if __name__ == "__main__":
    main()

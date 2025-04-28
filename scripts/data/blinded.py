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

    test = False
    # data_type = "FFT"
    data_type = "QMLE"
    ic_correction = False

    name_system = socket.gethostname()
    if "login" in name_system:
        path_in_challenge = [
            os.path.sep,
            "global",
            "cfs",
            "cdirs",
            "desi",
            "science",
            "lya",
            "y1-p1d",
            "iron-baseline",
        ]
        if data_type == "FFT":
            path_in_challenge += [
                "fft_measurement",
                "p1d_fft_y1_measurement_kms_v6.fits",
            ]

        elif data_type == "QMLE":
            path_in_challenge += [
                "qmle_measurement",
                "DataProducts",
                "desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v2.fits",
            ]
    else:
        path_in_challenge = [
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "cup1d",
            "obs",
        ]
        if data_type == "FFT":
            path_in_challenge += ["p1d_fft_y1_measurement_kms_v6.fits"]
        elif data_type == "QMLE":
            path_in_challenge += [
                "desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v2.fits"
            ]

    path_in_challenge = os.path.join(*path_in_challenge)
    path_out_challenge = os.path.join(
        os.path.dirname(get_path_repo("cup1d")), "data", "obs", data_type
    )

    print(path_in_challenge)
    print(path_out_challenge)

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
    # emulator_label = "CH24_nyx_gpr"
    emulator_label = "CH24_nyxcen_gpr"
    vary_alphas = False

    args = Args(emulator_label=emulator_label)
    args.data_label = "DESIY1"

    # impose_fid_cosmo_label = None
    impose_fid_cosmo_label = "Planck18"
    # impose_fid_cosmo_label = "Planck18_h74"

    # note redshift range!
    args.z_min = 2.1
    args.z_max = 4.3

    # args.emu_cov_factor = None
    args.emu_cov_factor = 1.0
    # args.emu_cov_type = "diagonal"
    # args.emu_cov_type = "block"
    args.emu_cov_type = "full"

    args.cov_only_diag = False
    args.sys_only_diag = False

    if test:
        args.n_steps = 500
        args.n_burn_in = 0
    else:
        args.n_steps = 2500
        args.n_burn_in = 1250

    if size > 1:
        args.parallel = True
    else:
        args.parallel = False

    if args.n_burn_in == 0:
        args.explore = True
    else:
        args.explore = False

    flags = emulator_label
    if args.sys_only_diag:
        flags = emulator_label + "_sysdiag"
    if args.emu_cov_factor is not None:
        if args.emu_cov_type == "diagonal":
            flags += "_emudiag"
        elif args.emu_cov_type == "block":
            flags += "_emublock"
        else:
            flags += "_emufull"

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
        # args.fid_SiIII_X = [0, -10]
        # args.fid_SiIII_D = [0, 5]
        # args.fid_SiIII_A = [0, 1]
        # args.fid_A_damp = [0, -9]
        # args.fid_A_scale = [0, 5]
        args.fid_SiIII_X = [0, -5]
        args.fid_SiIII_D = [0, 5]
        args.fid_SiIII_A = [0, 1]
        args.fid_A_damp = [0, -1]
        args.fid_A_scale = [0, 5]
        args.hcd_model_type = "new"

    flags_igm = (
        "ntau"
        + str(args.n_tau)
        + "_nsigT"
        + str(args.n_sigT)
        + "_ngamma"
        + str(args.n_gamma)
        + "_nkF"
        + str(args.n_kF)
        + "_nxSiIII"
        + str(args.n_x_SiIII)
        + "_ndSiIII"
        + str(args.n_d_SiIII)
        + "_naSiIII"
        + str(args.n_a_SiIII)
        + "_nddla"
        + str(args.n_d_dla)
        + "_nsdla"
        + str(args.n_s_dla)
    )
    dir_out = os.path.join(path_out_challenge, flags, flags_igm)

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

    args.p1d_fname = path_in_challenge
    if rank == 0:
        print("Analyzing:", args.p1d_fname)

    if rank == 0:
        os.makedirs(dir_out, exist_ok=True)
        print("Output in:", dir_out)

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

    if ic_correction:
        args.ic_correction = True
    else:
        args.ic_correction = False

    pip = Pipeline(args, make_plots=False, out_folder=dir_out)

    # run minimizer on fiducial (may not get to minimum)
    p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    pip.run_minimizer(p0)

    # run samplers, it uses as ini the results of the minimizer
    pip.run_sampler()

    # run minimizer again, now on MLE
    if rank == 0:
        pip.run_minimizer(pip.fitter.mle_cube, save_chains=True)


if __name__ == "__main__":
    main()

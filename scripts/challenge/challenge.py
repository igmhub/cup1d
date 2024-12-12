# //global/cfs/cdirs/desicollab/science/lya/y1-p1d/likelihood_files/data_files/MockChallengeSnapshot

import socket, os, sys, glob

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

    version = "8"

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
            "mockchallenge-0." + version,
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
            "mockchallenge-0." + version,
        )
        path_out_challenge = os.path.join(
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "mock_challenge",
            "v" + version,
        )

    print(path_in_challenge)
    print(path_out_challenge)

    # files = np.sort(glob.glob(folder_in + "*CGAN*.fits"))
    # files = np.sort(glob.glob(folder_in + "*grid_3.fits"))
    # files = np.sort(glob.glob(folder_in + "*bar_ic*.fits"))
    search = os.path.join(
        path_in_challenge,
        "mockchallenge-0." + version + "_nonoise_fiducial.fits.gz",
    )
    files = np.sort(glob.glob(search))
    if rank == 0:
        for ii in range(len(files)):
            print(ii, files[ii])

    # emulator_label = "Pedersen23_ext"
    # training_set = "Cabayol23"
    emulator_label = "Nyx_alphap_cov"
    training_set = "Nyx23_Jul2024"
    vary_alphas = True
    # vary_alphas = False
    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
    args = Args(emulator_label=emulator_label, training_set=training_set)
    args.data_label = "challenge_DESIY1"

    impose_fid_cosmo_label = None
    # impose_fid_cosmo_label = "Planck18"
    # impose_fid_cosmo_label = "Planck18_h74"

    # note redshift range!
    args.z_min = 2.1
    args.z_max = 4.2

    args.emu_cov_factor = 0.0

    args.n_steps = 1250
    args.n_burn_in = 1250
    # args.n_steps = 10
    # args.n_burn_in = 0
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
    args.n_tau = 2
    args.n_sigT = 2
    args.n_gamma = 2
    args.n_kF = 2

    # set archive and emulator
    if rank == 0:
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

    if ("Nyx" in emulator_label) and vary_alphas:
        args.vary_alphas = True
    else:
        args.vary_alphas = False

    for isim in range(len(files)):
        args.p1d_fname = files[isim]
        if rank == 0:
            print("Analyzing:", args.p1d_fname)
        file_sim = os.path.basename(args.p1d_fname)[:-8]
        if args.emu_cov_factor != 0:
            dir_out = os.path.join(base_out_folder + "err", file_sim)
        else:
            dir_out = os.path.join(base_out_folder, file_sim)
        if rank == 0:
            os.makedirs(dir_out, exist_ok=True)
            print("Output in:", dir_out)

        # same true and fiducial IGM
        if "fiducial" in args.p1d_fname:
            true_sim_label = "nyx_central"
        elif "CGAN" in args.p1d_fname:
            true_sim_label = "nyx_seed"
        elif "grid_3" in args.p1d_fname:
            true_sim_label = "nyx_3"
        else:
            true_sim_label = None

        if "Nyx" in emulator_label:
            args.true_cosmo_label = true_sim_label
            args.true_igm_label = true_sim_label
            args.fid_cosmo_label = true_sim_label
            args.fid_igm_label_mF = true_sim_label
            args.fid_igm_label_T = true_sim_label
            args.fid_igm_label_kF = true_sim_label
        else:
            args.true_cosmo_label = true_sim_label
            args.true_igm_label = true_sim_label
            args.fid_cosmo_label = true_sim_label
            args.fid_igm_label_mF = true_sim_label
            args.fid_igm_label_T = true_sim_label
            args.fid_igm_label_kF = "mpg_central"

        if impose_fid_cosmo_label is not None:
            args.fid_cosmo_label = impose_fid_cosmo_label

        fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

        # Planck18 0.354 -2.300 -0.2155
        # 5 sigma 0.056 0.011 0.0028
        blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params()
        # args.use_star_priors = None
        args.use_star_priors = {}
        args.use_star_priors["alpha_star"] = [
            blob["alpha_star"] - 0.0028,
            blob["alpha_star"] + 0.0028,
        ]

        if "bar_ic" in args.p1d_fname:
            args.ic_correction = True
            args.ic_correction = False
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
            ind = np.argmax(pip.fitter.lnprob.reshape(-1))
            p0 = pip.fitter.chain.reshape(-1, pip.fitter.chain.shape[-1])[ind]
        pip.run_minimizer(p0)


if __name__ == "__main__":
    main()

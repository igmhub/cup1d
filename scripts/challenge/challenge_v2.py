import socket
import time, os, sys
import glob

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from mpi4py import MPI
from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline
from cup1d.utils.utils import get_path_repo


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    version = "2"

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
    search = os.path.join(path_in_challenge, "*nonoise_fiducial.fits.gz")
    files = np.sort(glob.glob(search))
    if rank == 0:
        for ii in range(len(files)):
            print(ii, files[ii])

    # emulator_label = "Nyx_alphap_cov"
    # training_set = "Nyx23_Jul2024"
    emulator_label = "Cabayol23+"
    training_set = "Cabayol23"
    args = Args(emulator_label=emulator_label, training_set=training_set)
    args.data_label = "DESIY1"

    # args.n_steps = 5000
    # args.n_burn_in = 0
    args.n_steps = 20
    args.n_burn_in = 0
    # args.n_steps = 200
    # args.n_burn_in = 50
    # args.n_steps = 100
    # args.n_burn_in = 0
    if size > 1:
        args.parallel = True
    else:
        args.parallel = False
    args.explore = True

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
            args.vary_alphas = True
            if "nyx_14" in args.emulator.list_sim_cube:
                args.emulator.list_sim_cube.remove("nyx_14")
        else:
            args.emulator.list_sim_cube = args.archive.list_sim_cube
            args.vary_alphas = False

    args.emu_cov_factor = 0.0

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
        elif "grid_3" in args.p1d_fname:
            true_sim_label = "nyx_3"
        else:
            true_sim_label = "nyx_central"

        if "Nyx" in emulator_label:
            args.true_cosmo_label = true_sim_label
            args.true_sim_label = true_sim_label
            args.fid_cosmo_label = true_sim_label
            args.fid_igm_label = true_sim_label
        else:
            args.true_cosmo_label = true_sim_label
            args.true_sim_label = true_sim_label
            args.fid_cosmo_label = "mpg_central"
            args.fid_igm_label = "mpg_central"

        if "bar_ic" in args.p1d_fname:
            args.ic_correction = True
            args.ic_correction = False
        else:
            args.ic_correction = False

        pip = Pipeline(args, make_plots=False, out_folder=dir_out)
        pip.run_minimizer()
        pip.run_sampler()


if __name__ == "__main__":
    main()

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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    version = "2"
    folder_base1 = "/global/cfs/cdirs/desicollab/science/lya/y1-p1d/likelihood_files/data_files/MockChallengeSnapshot/"
    folder_base2 = "/global/homes/j/jjchaves/data/cup1d/mock_challenge/"
    # folder_base1 = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/"
    # folder_base2 = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/"

    folder_in = folder_base1 + "/mockchallenge-0." + version + "/"
    folder_out = folder_base2 + "/v" + version + "/"
    # files = np.sort(glob.glob(folder_in + "*CGAN*.fits"))
    # files = np.sort(glob.glob(folder_in + "*grid_3.fits"))
    # files = np.sort(glob.glob(folder_in + "*bar_ic*.fits"))
    files = np.sort(glob.glob(folder_in + "*nonoise_fiducial.fits.gz"))
    if rank == 0:
        for ii in range(len(files)):
            print(ii, files[ii])

    emulator_label = "Nyx_alphap_cov"
    training_set = "Nyx23_Jul2024"
    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
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

    base_out_folder = folder_out + emulator_label

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
        if args.emu_cov_factor != 0:
            dir_out = (
                base_out_folder
                + "_err/"
                + os.path.basename(args.p1d_fname)[:-5]
            )
        else:
            dir_out = (
                base_out_folder + "/" + os.path.basename(args.p1d_fname)[:-5]
            )
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

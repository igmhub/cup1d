import time, os, sys
import glob
import numpy as np
from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline


def main():
    version = "3"
    folder_in = (
        "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."
        + version
        + "/"
    )
    folder_out = (
        "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/v"
        + version
        + "/"
    )
    files = np.sort(glob.glob(folder_in + "*.fits"))
    for ii in range(len(files)):
        print(ii, files[ii])

    # emulator_label = "Pedersen23_ext"
    # emulator_label = "Cabayol23+"
    # training_set = "Cabayol23"
    emulator_label = "Nyx_alphap_cov"
    # emulator_label = "Nyx_alphap"
    training_set = "Nyx23_Jul2024"
    args = Args(emulator_label=emulator_label, training_set=training_set)
    args.data_label = "DESI_Y1"

    base_out_folder = folder_out + emulator_label

    # set number of free IGM parameters
    args.n_tau = 2
    args.n_sigT = 2
    args.n_gamma = 2
    args.n_kF = 2

    # set archive and emulator
    args.archive = set_archive(args.training_set)
    args.emulator = set_emulator(
        emulator_label=args.emulator_label,
        archive=args.archive,
    )

    if "Nyx" in args.emulator.emulator_label:
        args.emulator.list_sim_cube = args.archive.list_sim_cube
        args.vary_alphas = True
        if "nyx_14" in args.emulator.list_sim_cube:
            args.emulator.list_sim_cube.remove("nyx_14")
    else:
        args.emulator.list_sim_cube = args.archive.list_sim_cube
        args.vary_alphas = False

    # same true and fiducial IGM
    if "Nyx" in args.emulator.emulator_label:
        args.true_cosmo_label = "nyx_central"
        args.true_sim_label = "nyx_central"
        args.fid_cosmo_label = "nyx_central"
        args.fid_igm_label = "nyx_central"
    else:
        args.true_cosmo_label = "nyx_central"
        args.true_sim_label = "nyx_central"
        args.fid_cosmo_label = "mpg_central"
        args.fid_igm_label = "mpg_central"

    args.emu_cov_factor = 0.0

    for isim in range(len(files)):
        args.p1d_fname = files[isim]
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
        os.makedirs(dir_out, exist_ok=True)
        print("Output in:", dir_out)

        pip = Pipeline(args, make_plots=False, out_folder=dir_out)
        pip.run_minimizer()


if __name__ == "__main__":
    main()

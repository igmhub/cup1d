import os

os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI
import numpy as np

from cup1d.likelihood.input_pipeline import Args
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.pipeline import set_archive, Pipeline


def main():
    """Launch validate cosmology"""

    # emulator_label = "Pedersen23_ext"
    emulator_label = "Cabayol23+"
    training_set = "Cabayol23"
    # emulator_label = "Nyx_alphap_cov"
    # training_set = "Nyx23_Jul2024"

    include_sys = False

    base_out_folder = (
        "/home/jchaves/Proyectos/projects/lya/data/cup1d/validate_cosmo_igm/"
    )

    validate_cosmo(emulator_label, training_set, base_out_folder, include_sys)


def validate_cosmo(emulator_label, training_set, base_out_folder, include_sys):
    """Validate assuming a fiducial cosmology against forecast data"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = Args(emulator_label=emulator_label, training_set=training_set)

    args.n_steps = 1250
    args.n_burn_in = 500
    # args.n_burn_in = 1250
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

    # set true cosmology
    # args.true_cosmo_label = "Planck18"
    args.true_cosmo_label = "mpg_central"
    args.true_label_mF = "mpg_central"
    args.true_label_T = "mpg_central"
    args.true_label_kF = "mpg_central"

    # set covariance matrix
    args.data_label = "mock_DESIY1"
    args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"

    # set number of free IGM parameters
    args.mF_model_type = "chunks"
    args.n_tau = 11
    # args.mF_model_type = "pivot"
    # args.n_tau = 2

    args.n_sigT = 1
    args.n_gamma = 1
    args.n_kF = 1

    args.z_min = 2.1
    args.z_max = 4.3

    # set archive and emulator
    if rank == 0:
        args.archive = set_archive(args.training_set)
        args.emulator = set_emulator(
            emulator_label=args.emulator_label, archive=args.archive
        )

        if "Nyx" in args.emulator.emulator_label:
            args.emulator.list_sim_cube = args.archive.list_sim_cube
            if "nyx_14" in args.emulator.list_sim_cube:
                args.emulator.list_sim_cube.remove("nyx_14")
        else:
            args.emulator.list_sim_cube = args.archive.list_sim_cube

    if "Nyx" in emulator_label:
        args.vary_alphas = True
        list_sims = []
        for ii in range(14):
            list_sims.append("nyx_" + str(ii))
        list_sims.append("nyx_central")
    else:
        args.vary_alphas = False
        list_sims = []
        for ii in range(30):
            list_sims.append("mpg_" + str(ii))
        # list_sims.append("mpg_central")

    for kk, sim_label in enumerate(list_sims):
        print("\n\n\n")
        print(sim_label)
        print("\n\n\n")
        # if kk == 0:
        # continue

        # set fiducial cosmology and igm (the only thing that changes)
        args.fid_cosmo_label = sim_label
        args.fid_label_mF = sim_label
        args.fid_label_T = sim_label
        args.fid_label_kF = sim_label

        if include_sys:
            flag = "syst_"
        else:
            flag = ""

        out_folder = os.path.join(
            base_out_folder,
            args.data_label[5:],
            emulator_label,
            args.mF_model_type,
            sim_label
            + "_"
            + flag
            + str(args.n_tau)
            + "_"
            + str(args.n_sigT)
            + "_"
            + str(args.n_gamma)
            + "_"
            + str(args.n_kF),
        )

        pip = Pipeline(args, out_folder=out_folder)
        p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
        pip.run_minimizer(p0)
        pip.run_sampler()
        if rank == 0:
            ind = np.argmax(pip.fitter.lnprob.reshape(-1))
            p0 = pip.fitter.chain.reshape(-1, pip.fitter.chain.shape[-1])[ind]
        pip.run_minimizer(p0)


if __name__ == "__main__":
    main()

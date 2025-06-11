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

    # test = True
    test = False

    ## set data to use

    # data_type = "FFT"
    # data_type = "QMLE"
    data_type = "QMLE3"

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
    else:
        path_in_challenge = [
            os.path.dirname(get_path_repo("cup1d")),
            "data",
            "DESI-DR1",
        ]
    if data_type == "FFT":
        path_in_challenge += [
            "fft_measurement",
            "p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits",
        ]

    elif data_type == "QMLE":
        path_in_challenge += [
            "qmle_measurement",
            "DataProducts",
            "v3",
            "desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits",
        ]
    elif data_type == "QMLE3":
        path_in_challenge += [
            "qmle_measurement",
            "DataProducts",
            "v3",
            "desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits",
        ]

    path_in_challenge = os.path.join(*path_in_challenge)
    path_out_challenge = os.path.join(
        os.path.dirname(get_path_repo("cup1d")), "data", "obs", data_type
    )

    print(path_in_challenge)
    print(path_out_challenge)

    ## set baseline

    emulator_label = "CH24_mpgcen_gpr"
    # emulator_label = "CH24_nyxcen_gpr"

    args = Args(emulator_label=emulator_label)
    args.data_label = "DESIY1"
    args.cov_syst_type = "red"
    # note redshift range!
    args.z_min = 2.1
    args.z_max = 4.3

    args.set_baseline()

    args.p1d_fname = path_in_challenge
    if rank == 0:
        print("Analyzing:", args.p1d_fname)

    # stuff for fitter
    if test:
        args.n_steps = 10
        args.n_burn_in = 0
    else:
        args.n_steps = 3100
        args.n_burn_in = 100

    if size > 1:
        args.parallel = True
    else:
        args.parallel = False

    if args.n_burn_in == 0:
        args.explore = True
    else:
        args.explore = False

    flags = emulator_label

    args.cov_factor = 1e50

    flags_igm = "priors"
    dir_out = os.path.join(path_out_challenge, flags, flags_igm)
    if rank == 0:
        os.makedirs(dir_out, exist_ok=True)

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

    pip = Pipeline(args, make_plots=False, out_folder=dir_out)

    # run minimizer on fiducial (may not get to minimum)
    p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    if rank == 0:
        base_save_dir = pip.fitter.save_directory
        print("base_save_dir:", base_save_dir)

    for ii in range(1, len(pip.fitter.like.data.z)):
        # for ii in range(1):
        zmask = np.array([pip.fitter.like.data.z[ii]])

        if rank == 0:
            pip.fitter.save_directory = os.path.join(
                base_save_dir, str(np.round(zmask[0], 2))
            )
            print(ii, zmask, pip.fitter.save_directory)
            os.makedirs(pip.fitter.save_directory, exist_ok=True)

        pip.run_minimizer(
            p0=p0, zmask=zmask, nsamples=1, restart=True, make_plots=False
        )

        # run samplers, it uses as ini the results of the minimizer
        pip.run_sampler(pini=pip.fitter.mle_cube, zmask=zmask)

        # run minimizer again, now on MLE
        if rank == 0:
            pip.run_minimizer(
                p0=pip.fitter.mle_cube,
                zmask=zmask,
                nsamples=1,
                save_chains=True,
            )


if __name__ == "__main__":
    main()

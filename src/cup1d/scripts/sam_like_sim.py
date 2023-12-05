import os, sys, time, psutil
import numpy as np
import configargparse
import multiprocessing as mp

# our own modules
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from cup1d.data import data_gadget, data_nyx
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler


def fprint(*args, verbose=True):
    if verbose:
        print(*args, flush=True)


def parse_args():
    def str_to_bool(s):
        if s == "True":
            return True
        elif s == "False":
            return False

    parser = configargparse.ArgumentParser(
        description="Passing options to sampler"
    )

    # archive and emulator
    parser.add_argument(
        "--training_set",
        default=None,
        choices=["Pedersen21", "Cabayol23", "Nyx23_Oct2023"],
        required=True,
    )

    # emulator
    parser.add_argument(
        "--emulator_label",
        default=None,
        choices=[
            "Pedersen21",
            "Cabayol23",
            "Cabayol23_extended",
            "Nyx_v0",
            "Nyx_v0_extended",
        ],
        required=True,
    )
    parser.add_argument(
        "--test_sim_label",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--igm_sim_label",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--drop_sim",
        default="False",
        choices=["True", "False"],
        help="Drop test_sim_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--add_hires",
        default="False",
        choices=["True", "False"],
        help="Include high-res data (Karacayli2022)",
    )
    parser.add_argument(
        "--use_polyfit",
        default="True",
        choices=["True", "False"],
        help="Fit data after fitting polynomial",
    )

    # likelihood
    parser.add_argument(
        "--cov_label",
        type=str,
        default="Chabanier2019",
        choices=["Chabanier2019", "QMLE_Ohio"],
        help="Data covariance to use, Chabanier2019 or QMLE_Ohio",
    )
    parser.add_argument(
        "--emu_cov_factor",
        type=float,
        default=0,
        help="scale contribution of emulator covariance",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--prior_Gauss_rms",
        default=None,
        help="Width of Gaussian prior",
    )

    parser.add_argument(
        "--no_verbose",
        action="store_false",
        help="print information",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test job",
    )

    parser.add_argument(
        "--no_parallel",
        action="store_false",
        help="Parallelize",
    )
    # not implemented yet!
    # parser.add_argument(
    #     "--vary_fiducial_cosmo",
    #     default="False",
    #     choices=["True", "False"],
    #     help="Use as fiducial cosmology the one of the target mock",
    # )

    # parser.add_argument(
    #     "--z_min", type=float, default=2.0, help="Minimum redshift"
    # )
    # parser.add_argument(
    #     "--z_max", type=float, default=4.5, help="Maximum redshift"
    # )
    # parser.add_argument(
    #     "--cosmo_fid_label",
    #     type=str,
    #     default="default",
    #     help="Fiducial cosmology to use (default,truth)",
    # )

    #######################
    # print args
    args = parser.parse_args()
    fprint("--- print options from parser ---", verbose=args.no_verbose)
    fprint(args, verbose=args.no_verbose)
    fprint("----------", verbose=args.no_verbose)
    fprint(parser.format_values(), verbose=args.no_verbose)
    fprint("----------", verbose=args.no_verbose)

    args.drop_sim = str_to_bool(args.drop_sim)
    args.add_hires = str_to_bool(args.add_hires)
    args.use_polyfit = str_to_bool(args.use_polyfit)
    args.archive = None

    assert "CUP1D_PATH" in os.environ, "Define CUP1D_PATH variable"

    return args


def load_emu(
    archive,
    label_training_set,
    emulator_label,
    test_sim_label,
    drop_sim,
):
    def set_emu_path(
        label_training_set,
        emulator_label,
        test_sim_label,
        drop_sim,
    ):
        folder = "NNmodels/" + label_training_set + "/"
        # set file name
        fname = emulator_label
        if drop_sim != False:
            fname += "_drop_sim_" + test_sim_label
        fname += ".pt"

        return folder + fname

    if emulator_label == "Pedersen21":
        if label_training_set != "Pedersen21":
            raise (
                "Combination of training_set ("
                + label_training_set
                + ") and emulator_label ("
                + emulator_label
                + ") not allowed:"
            )
        if drop_sim:
            _drop_sim = test_sim_label
        else:
            _drop_sim = None
        fprint("Training emulator " + emulator_label, verbose=args.no_verbose)
        emulator = GPEmulator(
            training_set=label_training_set,
            emulator_label=emulator_label,
            drop_sim=_drop_sim,
        )
    else:
        if (label_training_set == "Cabayol23") & (
            (emulator_label == "Cabayol23")
            | (emulator_label == "Cabayol23_extended")
        ):
            pass
        elif (label_training_set[:5] == "Nyx23") & (
            (emulator_label == "Nyx_v0") | (emulator_label == "Nyx_v0_extended")
        ):
            pass
        else:
            msg = (
                "Combination of training_set ("
                + label_training_set
                + ") and emulator_label ("
                + emulator_label
                + ") not allowed:"
            )
            raise ValueError(msg)

        emu_path = set_emu_path(
            label_training_set, emulator_label, test_sim_label, drop_sim
        )
        if drop_sim:
            _drop_sim = test_sim_label
        else:
            _drop_sim = None
        fprint("Loading emulator " + emulator_label, verbose=args.no_verbose)
        emulator = NNEmulator(
            archive=archive,
            training_set=label_training_set,
            emulator_label=emulator_label,
            model_path=emu_path,
            drop_sim=_drop_sim,
            train=False,
        )

    return emulator


def path_sampler(args):
    if args.drop_sim:
        flag_drop = "ydrop"
    else:
        flag_drop = "ndrop"
    if args.use_polyfit:
        flag_poly = "ypoly"
    else:
        flag_poly = "npoly"
    if args.add_hires:
        flag_hires = "hres"
    else:
        flag_hires = "lres"

    path = os.environ["LYA_DATA_PATH"]
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "cup1d/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "sampler/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += args.training_set + "_" + flag_hires + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += (
        args.emulator_label
        + "_"
        + args.cov_label
        + "_"
        + args.test_sim_label
        + "_igm"
        + str(args.n_igm)
        + "_"
        + flag_drop
        + "_"
        + flag_poly
        + "/"
    )
    if os.path.isdir(path) == False:
        os.mkdir(path)

    return path


def log_prob(theta):
    return log_prob.sampler.like.log_prob_and_blobs(theta)


def set_log_prob(sampler):
    log_prob.sampler = sampler
    return log_prob


def sample(args, like, free_parameters):
    """Sample the posterior distribution"""

    path = path_sampler(args)

    sampler = emcee_sampler.EmceeSampler(like=like, rootdir=path)
    _log_prob = set_log_prob(sampler)

    _ = sampler.run_sampler(
        args.n_burn_in,
        args.n_steps,
        log_func=_log_prob,
        parallel=args.no_parallel,
    )
    sampler.write_chain_to_file()


def sam_like_sim(args):
    """Sample the posterior distribution for a of a mock"""

    nthreads = psutil.cpu_count(logical=True)
    ncores = psutil.cpu_count(logical=False)
    nthreads_per_core = nthreads // ncores
    nthreads_available = len(os.sched_getaffinity(0))
    ncores_available = nthreads_available // nthreads_per_core

    assert nthreads == os.cpu_count()
    assert nthreads == mp.cpu_count()

    fprint(f"{nthreads=}", verbose=args.no_verbose)
    fprint(f"{ncores=}", verbose=args.no_verbose)
    fprint(f"{nthreads_per_core=}", verbose=args.no_verbose)
    fprint(f"{nthreads_available=}", verbose=args.no_verbose)
    fprint(f"{ncores_available=}", verbose=args.no_verbose)

    start_all = time.time()

    # os.environ["OMP_NUM_THREADS"] = "1"

    #######################
    # load training set
    start = time.time()
    fprint("----------", verbose=args.no_verbose)
    fprint("Setting training set " + args.training_set, verbose=args.no_verbose)

    args.n_steps = 1000
    if args.cov_label == "Chabanier2019":
        if args.n_igm == 0:
            args.n_burn_in = 75
        else:
            args.n_burn_in = 250
    elif args.cov_label == "QMLE_Ohio":
        if args.n_igm == 0:
            # TBD (need to check)
            args.n_burn_in = 200
        else:
            args.n_burn_in = 1200
    else:
        if args.n_igm == 0:
            args.n_burn_in = 200
        else:
            args.n_burn_in = 500

    if args.test == True:
        args.n_steps = 10
        args.n_burn_in = 0

    if args.archive is None:
        if args.training_set == "Pedersen21":
            archive = gadget_archive.GadgetArchive(postproc=args.training_set)
            set_P1D = data_gadget.Gadget_P1D
            z_min = 2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
        elif args.training_set == "Cabayol23":
            archive = gadget_archive.GadgetArchive(postproc=args.training_set)
            set_P1D = data_gadget.Gadget_P1D
            z_min = 2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
        elif args.training_set[:5] == "Nyx23":
            archive = nyx_archive.NyxArchive(nyx_version=args.training_set[6:])
            set_P1D = data_nyx.Nyx_P1D
            z_min = 2.2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
        else:
            raise ValueError("Training_set not implemented")
    else:
        archive = args.archive
        z_max = args.z_max
        set_P1D = args.set_P1D

    if args.test_sim_label not in archive.list_sim:
        fprint(
            args.test_sim_label + " is not in part of " + args.training_set,
            verbose=args.no_verbose,
        )
        fprint(
            "List of simulations available: ",
            archive.list_sim,
            verbose=args.no_verbose,
        )
        sys.exit()
    end = time.time()
    multi_time = str(np.round(end - start, 2))
    fprint("z in range ", z_min, ", ", z_max, verbose=args.no_verbose)
    fprint("Training set loaded " + multi_time + " s", verbose=args.no_verbose)

    #######################
    # set emulator
    fprint("----------", verbose=args.no_verbose)
    fprint("Setting emulator", verbose=args.no_verbose)
    start = time.time()
    if args.drop_sim:
        ## only drop sim if it was in the training set
        if args.test_sim_label in archive.list_sim_cube:
            _drop_sim = True
        else:
            _drop_sim = False
    else:
        _drop_sim = False
    emulator = load_emu(
        archive,
        args.training_set,
        args.emulator_label,
        args.test_sim_label,
        _drop_sim,
    )
    multi_time = str(np.round(time.time() - start, 2))
    fprint("Emulator loaded " + multi_time + " s", verbose=args.no_verbose)

    if args.use_polyfit:
        polyfit_kmax_Mpc = emulator.kmax_Mpc
        polyfit_ndeg = emulator.ndeg
    else:
        polyfit_kmax_Mpc = None
        polyfit_ndeg = None

    #######################
    # set target P1D
    data = set_P1D(
        archive=archive,
        input_sim=args.test_sim_label,
        # z_min=z_min,
        z_max=z_max,
        data_cov_label=args.cov_label,
        polyfit_kmax_Mpc=polyfit_kmax_Mpc,
        polyfit_ndeg=polyfit_ndeg,
    )
    if args.add_hires:
        extra_data = set_P1D(
            archive=archive,
            input_sim=args.test_sim_label,
            # z_min=z_min,
            z_max=z_max,
            data_cov_label="Karacayli2022",
            polyfit_kmax_Mpc=polyfit_kmax_Mpc,
            polyfit_ndeg=polyfit_ndeg,
        )
    else:
        extra_data = None

    #######################
    # set likelihood
    ## set cosmo free parameters
    fprint("----------", verbose=args.no_verbose)
    fprint("Set likelihood", verbose=args.no_verbose)
    free_parameters = ["As", "ns"]
    fprint(
        "Using {} parameters for IGM model".format(args.n_igm),
        verbose=args.no_verbose,
    )
    for ii in range(args.n_igm):
        for par in ["tau", "sigT_kms", "gamma", "kF"]:
            free_parameters.append("ln_{}_{}".format(par, ii))
    fprint("free parameters", free_parameters, verbose=args.no_verbose)
    ## set theory
    theory = lya_theory.Theory(
        zs=data.z,
        emulator=emulator,
        free_param_names=free_parameters,
        fid_sim_igm=args.igm_sim_label,
        true_sim_igm=args.test_sim_label,
    )
    ## set like
    like = likelihood.Likelihood(
        data=data,
        theory=theory,
        free_param_names=free_parameters,
        prior_Gauss_rms=args.prior_Gauss_rms,
        emu_cov_factor=args.emu_cov_factor,
        extra_p1d_data=extra_data,
    )

    #######################
    # sample likelihood
    fprint("----------", verbose=args.no_verbose)
    fprint("Sampler", verbose=args.no_verbose)
    start = time.time()
    sample(args, like, free_parameters)
    multi_time = str(np.round(time.time() - start, 2))
    fprint("Sample in " + multi_time + " s", verbose=args.no_verbose)
    fprint("", verbose=args.no_verbose)
    fprint("", verbose=args.no_verbose)
    multi_time = str(np.round(time.time() - start_all, 2))
    fprint("Program took " + multi_time + " s", verbose=args.no_verbose)
    fprint("", verbose=args.no_verbose)
    fprint("", verbose=args.no_verbose)


if __name__ == "__main__":
    args = parse_args()
    sam_like_sim(args)

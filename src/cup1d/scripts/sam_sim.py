import os, sys, time, psutil
import numpy as np
import configargparse
from mpi4py import MPI

# our own modules
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from cup1d.data import data_gadget, data_nyx
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler
from cup1d.utils.utils import create_print_function, mpi_hello_world


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
        "--mock_sim_label",
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
        "--cosmo_sim_label",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--drop_sim",
        default="False",
        choices=["True", "False"],
        help="Drop mock_sim_label simulation from the training set",
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
        "--version",
        default="v1",
        help="Version of the pipeline",
    )

    parser.add_argument(
        "--prior_Gauss_rms",
        default=None,
        help="Width of Gaussian prior",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print information",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test job",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
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
    mpi_hello_world()

    fprint = create_print_function(verbose=args.verbose)
    fprint("--- print options from parser ---")
    fprint(args)
    fprint("----------")
    fprint(parser.format_values())
    fprint("----------")

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
    mock_sim_label,
    drop_sim,
    fprint=print,
):
    def set_emu_path(
        label_training_set,
        emulator_label,
        mock_sim_label,
        drop_sim,
    ):
        folder = "NNmodels/" + label_training_set + "/"
        # set file name
        fname = emulator_label
        if drop_sim != False:
            fname += "_drop_sim_" + mock_sim_label
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
            _drop_sim = mock_sim_label
        else:
            _drop_sim = None
        fprint("Training emulator " + emulator_label)
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
            label_training_set, emulator_label, mock_sim_label, drop_sim
        )
        if drop_sim:
            _drop_sim = mock_sim_label
        else:
            _drop_sim = None
        fprint("Loading emulator " + emulator_label)
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
    path += args.version + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += args.training_set + "_" + flag_hires + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += (
        "emu_"
        + args.emulator_label
        + "_cov_"
        + args.cov_label
        + "_mocksim_"
        + args.mock_sim_label
        + "_cosmosim_"
        + args.cosmo_sim_label
        + "_igmsim_"
        + args.igm_sim_label
        + "_nigm_"
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


def sample(args, like, free_parameters, fprint=print):
    """Sample the posterior distribution"""

    path = path_sampler(args)
    fprint("\n\n Output in folder: " + path + "\n\n")

    sampler = emcee_sampler.EmceeSampler(
        like=like,
        rootdir=path,
        save_chain=False,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
    )
    _log_prob = set_log_prob(sampler)

    _ = sampler.run_sampler(log_func=_log_prob)

    if MPI.COMM_WORLD.Get_rank() == 0:
        sampler.write_chain_to_file()


def sam_sim(args):
    """Sample the posterior distribution for a of a mock"""

    #######################
    fprint = create_print_function(verbose=args.verbose)

    start_all = time.time()

    #######################
    # load training set
    start = time.time()
    fprint("----------")
    fprint("Setting training set " + args.training_set)

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
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
            set_P1D = data_gadget.Gadget_P1D
            archive = gadget_archive.GadgetArchive(postproc=args.training_set)
            z_min = 2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
        elif args.training_set == "Cabayol23":
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
            set_P1D = data_gadget.Gadget_P1D
            archive = gadget_archive.GadgetArchive(postproc=args.training_set)
            z_min = 2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
        elif args.training_set[:5] == "Nyx23":
            get_cosmo = camb_cosmo.get_Nyx_cosmology
            set_P1D = data_nyx.Nyx_P1D
            archive = nyx_archive.NyxArchive(nyx_version=args.training_set[6:])
            z_min = 2.2
            z_max = 4.5
            # z_max = np.max(archive.list_sim_redshifts)
        else:
            raise ValueError("Training_set not implemented")
    else:
        archive = args.archive
        z_max = args.z_max
        set_P1D = args.set_P1D
        get_cosmo = args.get_cosmo

    if args.mock_sim_label not in archive.list_sim:
        fprint(
            args.mock_sim_label + " is not in part of " + args.training_set,
            verbose=args.verbose,
        )
        fprint(
            "List of simulations available: ",
            archive.list_sim,
            verbose=args.verbose,
        )
        sys.exit()
    end = time.time()
    multi_time = str(np.round(end - start, 2))
    # fprint("z in range ", z_min, ", ", z_max)
    fprint("Training set loaded " + multi_time + " s")

    #######################
    # set emulator
    fprint("----------")
    fprint("Setting emulator")
    start = time.time()
    if args.drop_sim:
        ## only drop sim if it was in the training set
        if args.mock_sim_label in archive.list_sim_cube:
            _drop_sim = True
        else:
            _drop_sim = False
    else:
        _drop_sim = False

    emulator = load_emu(
        archive,
        args.training_set,
        args.emulator_label,
        args.mock_sim_label,
        _drop_sim,
        fprint=fprint,
    )

    multi_time = str(np.round(time.time() - start, 2))
    fprint("Emulator loaded " + multi_time + " s")

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
        input_sim=args.mock_sim_label,
        # z_min=z_min,
        z_max=z_max,
        data_cov_label=args.cov_label,
        polyfit_kmax_Mpc=polyfit_kmax_Mpc,
        polyfit_ndeg=polyfit_ndeg,
    )
    if args.add_hires:
        extra_data = set_P1D(
            archive=archive,
            input_sim=args.mock_sim_label,
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
    fprint("----------")
    fprint("Set likelihood")
    free_parameters = ["As", "ns"]
    fprint(
        "Using {} parameters for IGM model".format(args.n_igm),
        verbose=args.verbose,
    )
    for ii in range(args.n_igm):
        for par in ["tau", "sigT_kms", "gamma", "kF"]:
            free_parameters.append("ln_{}_{}".format(par, ii))
    fprint("free parameters", free_parameters)

    # set fiducial cosmology
    testing_data = archive.get_testing_data(args.cosmo_sim_label, z_max=z_max)
    cosmo_fid = get_cosmo(testing_data[0]["cosmo_params"])

    ## set theory
    theory = lya_theory.Theory(
        zs=data.z,
        emulator=emulator,
        free_param_names=free_parameters,
        fid_sim_igm=args.igm_sim_label,
        true_sim_igm=args.mock_sim_label,
        cosmo_fid=cosmo_fid,
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
    fprint("----------")
    fprint("Sampler")
    start = time.time()
    sample(args, like, free_parameters, fprint=fprint)
    multi_time = str(np.round(time.time() - start, 2))
    fprint("Sample in " + multi_time + " s")
    fprint("")
    fprint("")
    multi_time = str(np.round(time.time() - start_all, 2))
    fprint("Program took " + multi_time + " s")
    fprint("")
    fprint("")


if __name__ == "__main__":
    args = parse_args()
    sam_sim(args)

import os, sys, time, psutil
import numpy as np
import configargparse
from mpi4py import MPI

# our own modules
import lace
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from cup1d.data import (
    data_gadget,
    data_nyx,
    data_eBOSS_mock,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Karacayli2023,
    data_Ravoux2023,
)
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler
from cup1d.utils.utils import create_print_function, mpi_hello_world


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Passing options to sampler"
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
        action="store_true",
        help="Drop mock_sim_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--add_hires",
        action="store_true",
        help="Include high-res data (Karacayli2022)",
    )
    parser.add_argument(
        "--use_polyfit",
        action="store_true",
        help="Fit data after fitting polynomial",
    )

    # likelihood
    parser.add_argument(
        "--cov_label",
        type=str,
        default="Chabanier2019",
        choices=["Chabanier2019", "QMLE_Ohio"],
        help="Data covariance",
    )
    parser.add_argument(
        "--cov_label_hires",
        type=str,
        default="Karacayli2022",
        choices=["Karacayli2022"],
        help="Data covariance for high-res data",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to P1D mock according to covariance matrix",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Steps of emcee chains",
    )
    parser.add_argument(
        "--seed_noise",
        type=int,
        default=0,
        help="Seed for noise",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--fix_cosmo",
        action="store_true",
        help="Fix cosmological parameters while sampling",
    )

    parser.add_argument(
        "--version",
        default="v2",
        help="Version of the pipeline",
    )

    parser.add_argument(
        "--prior_Gauss_rms",
        default=None,
        help="Width of Gaussian prior",
    )

    parser.add_argument(
        "--emu_cov_factor",
        type=float,
        default=0,
        help="scale contribution of emulator covariance",
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
    #     "--z_min", type=float, default=2.0, help="Minimum redshift"
    # )
    parser.add_argument(
        "--z_max", type=float, default=4.5, help="Maximum redshift"
    )

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

    args.archive = None

    # set label_training_set
    if args.emulator_label == "Pedersen21":
        args.training_set = "Pedersen21"
    elif args.emulator_label == "Pedersen23":
        args.training_set = "Cabayol23"
    elif args.emulator_label == "Cabayol23":
        args.training_set = "Cabayol23"
    elif args.emulator_label == "Cabayol23_extended":
        args.training_set = "Cabayol23"
    elif args.emulator_label == "Nyx_v0":
        args.training_set = "Nyx23_Oct2023"
    elif args.emulator_label == "Nyx_v0_extended":
        args.training_set = "Nyx23_Oct2023"

    # set n_steps and n_burn_in
    if args.test == True:
        args.n_steps = 10
        args.n_burn_in = 0
    else:
        if args.cov_label == "Chabanier2019":
            if args.n_igm == 0:
                args.n_burn_in = 100
            else:
                args.n_burn_in = 500
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

    return args


def set_emu(
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
        if drop_sim is not None:
            fname += "_drop_sim_" + mock_sim_label
        fname += ".pt"

        return folder + fname

    if drop_sim:
        # only drop sim if it is in the training set
        if mock_sim_label in archive.list_sim_cube:
            _drop_sim = mock_sim_label
        else:
            _drop_sim = None
    else:
        _drop_sim = None

    # set emulator
    if emulator_label == "Pedersen21":
        fprint("Training emulator " + emulator_label)
        emulator = GPEmulator(
            training_set=label_training_set,
            emulator_label=emulator_label,
            drop_sim=_drop_sim,
        )
    else:
        emu_path = set_emu_path(
            label_training_set, emulator_label, mock_sim_label, _drop_sim
        )
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
    path += "emu_" + args.emulator_label + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += "cov_" + args.cov_label + "_" + flag_hires + "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)
    path += (
        "mock_"
        + args.mock_sim_label
        + "_igm_"
        + args.igm_sim_label
        + "_cosmo_"
        + args.cosmo_sim_label
        + "_nigm_"
        + str(args.n_igm)
        + "_"
        + flag_drop
        + "_"
        + flag_poly
    )
    if args.add_noise:
        path += "_noise_" + str(args.seed_noise)
    if args.fix_cosmo:
        path += "_fix_cosmo"
    path += "/"
    if os.path.isdir(path) == False:
        os.mkdir(path)

    return path


def log_prob(theta):
    return log_prob.sampler.like.log_prob_and_blobs(theta)


def set_log_prob(sampler):
    log_prob.sampler = sampler
    return log_prob


def sample(args, like, fprint=print):
    """Sample the posterior distribution"""

    fprint("----------")
    fprint("Sampler")

    path = path_sampler(args)
    fprint("\n\n Output in folder: " + path + "\n\n")

    sampler = emcee_sampler.EmceeSampler(
        like=like,
        rootdir=path,
        save_chain=False,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        fix_cosmology=args.fix_cosmo,
    )
    _log_prob = set_log_prob(sampler)

    _ = sampler.run_sampler(log_func=_log_prob)

    if MPI.COMM_WORLD.Get_rank() == 0:
        sampler.write_chain_to_file()


def set_archive(args):
    if (args.training_set == "Pedersen21") | (args.training_set == "Cabayol23"):
        archive = gadget_archive.GadgetArchive(postproc=args.training_set)
    elif args.training_set[:5] == "Nyx23":
        archive = nyx_archive.NyxArchive(nyx_version=args.training_set[6:])
    else:
        raise ValueError("Training_set not implemented")
    return archive


def set_p1ds(args, testing_data=None):
    if (args.mock_sim_label[:3] == "mpg") | (args.mock_sim_label[:3] == "nyx"):
        if testing_data is None:
            raise ValueError(
                f"You must provide testing_data to set_p1ds for {args.mock_sim_label} mock_sim_label"
            )
        if args.mock_sim_label[:3] == "mpg":
            set_P1D = data_gadget.Gadget_P1D
        elif args.mock_sim_label[:3] == "nyx":
            set_P1D = data_nyx.Nyx_P1D

        # set target P1D
        data = set_P1D(
            testing_data=testing_data,
            input_sim=args.mock_sim_label,
            # z_min=z_min,
            z_max=args.z_max,
            data_cov_label=args.cov_label,
            polyfit_kmax_Mpc=args.polyfit_kmax_Mpc,
            polyfit_ndeg=args.polyfit_ndeg,
            add_noise=args.add_noise,
            seed=args.seed_noise,
        )
        if args.add_hires:
            extra_data = set_P1D(
                testing_data=testing_data,
                input_sim=args.mock_sim_label,
                # z_min=z_min,
                z_max=args.z_max,
                data_cov_label=args.cov_label_hires,
                polyfit_kmax_Mpc=args.polyfit_kmax_Mpc,
                polyfit_ndeg=args.polyfit_ndeg,
                add_noise=args.add_noise,
                seed=args.seed_noise,
            )
        else:
            extra_data = None

    elif args.mock_sim_label == "eBOSS_mock":
        data = data_eBOSS_mock.P1D_eBOSS_mock(
            add_noise=args.add_noise,
            seed=args.seed_noise,
        )
        if args.add_hires:
            raise ValueError("Hires not implemented for eBOSS_mock")
        else:
            extra_data = None

    elif args.mock_sim_label == "Chabanier19":
        data = data_Chabanier2019.P1D_Chabanier2019()
        if args.add_hires:
            extra_data = data_Karacayli2022.P1D_Karacayli2022()
        else:
            extra_data = None
    elif args.mock_sim_label == "Ravoux23":
        data = data_Ravoux2023.P1D_Ravoux23()
        if args.add_hires:
            extra_data = data_Karacayli2022.P1D_Karacayli2022()
        else:
            extra_data = None
    elif args.mock_sim_label == "Karacayli23":
        data = data_Karacayli2023.P1D_Karacayli2023()
        if args.add_hires:
            extra_data = data_Karacayli2022.P1D_Karacayli2022()
        else:
            extra_data = None
    else:
        raise ValueError(
            f"mock_sim_label {args.mock_sim_label} not implemented"
        )

    return data, extra_data


def set_fid_cosmo(args):
    if (args.cosmo_sim_label[:3] == "mpg") | (
        args.cosmo_sim_label[:3] == "nyx"
    ):
        if args.cosmo_sim_label[:3] == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        elif args.cosmo_sim_label[:3] == "nyx":
            fname = os.environ["NYX_PATH"] + "nyx_emu_cosmo_Oct2023.npy"
            get_cosmo = camb_cosmo.get_Nyx_cosmology

        try:
            data_cosmo = np.load(fname, allow_pickle=True)
        except:
            ValueError(f"{fname} not found")

        cosmo_fid = None
        for ii in range(len(data_cosmo)):
            if data_cosmo[ii]["sim_label"] == args.cosmo_sim_label:
                cosmo_fid = get_cosmo(data_cosmo[ii]["cosmo_params"])
                break
        if cosmo_fid is None:
            raise ValueError(
                f"Cosmo not found in {fname} for {args.cosmo_sim_label}"
            )
    else:
        raise ValueError(
            f"cosmo_sim_label {args.cosmo_sim_label} not implemented"
        )
    return cosmo_fid


def set_like(args, emulator, data, extra_data, cosmo_fid, fprint=print):
    ## set cosmo and IGM parameters
    fprint("----------")
    fprint("Set likelihood")
    if args.fix_cosmo:
        free_parameters = []
    else:
        free_parameters = ["As", "ns"]
    fprint(f"Using {args.n_igm} parameters for IGM model")
    for ii in range(args.n_igm):
        for par in ["tau", "sigT_kms", "gamma", "kF"]:
            free_parameters.append(f"ln_{par}_{ii}")
    fprint("free parameters", free_parameters)

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

    return like


def sam_sim(args):
    """Sample the posterior distribution for a of a mock"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #######################
    fprint = create_print_function(verbose=args.verbose)

    if rank == 0:
        start_all = time.time()

    #######################

    # set training set

    if rank == 0:
        start = time.time()
        fprint("----------")
        fprint("Setting training set " + args.training_set)
        if args.archive is None:
            # if calling the script from the command line
            archive = set_archive(args)
        else:
            # if calling the script from a python script (and providing an archive)
            # MAKE SURE YOU ONLY PASS IT FOR RANK 0
            archive = args.archive
        end = time.time()
        multi_time = str(np.round(end - start, 2))
        fprint("Training set loaded " + multi_time + " s")

    #######################

    # set emulator
    if rank == 0:
        fprint("----------")
        fprint("Setting emulator")
        start = time.time()

        emulator = set_emu(
            archive,
            args.training_set,
            args.emulator_label,
            args.mock_sim_label,
            args.drop_sim,
            fprint=fprint,
        )

        multi_time = str(np.round(time.time() - start, 2))
        fprint("Emulator loaded " + multi_time + " s")

        # distribute emulator to all tasks
        for irank in range(1, size):
            comm.send(emulator, dest=irank, tag=irank)
    else:
        # receive emulator from task 0
        emulator = comm.recv(source=0, tag=rank)

    # send emulator to all tasks

    # Apply the same polyfit to the data as to the emulator
    if args.use_polyfit:
        args.polyfit_kmax_Mpc = emulator.kmax_Mpc
        args.polyfit_ndeg = emulator.ndeg
    else:
        args.polyfit_kmax_Mpc = None
        args.polyfit_ndeg = None

    #######################

    # set P1D
    if rank == 0:
        fprint("----------")
        fprint("Setting P1D")
        start = time.time()

        if (args.mock_sim_label[:3] == "mpg") | (
            args.mock_sim_label[:3] == "nyx"
        ):
            if args.mock_sim_label in archive.list_sim:
                archive_mock = archive
            else:
                if args.mock_sim_label[:3] == "mpg":
                    archive_mock = gadget_archive.GadgetArchive()
                elif args.mock_sim_label[:3] == "nyx":
                    archive_mock = nyx_archive.NyxArchive()
            try:
                assert args.mock_sim_label in archive_mock.list_sim
            except AssertionError:
                raise ValueError(
                    "Simulation "
                    + args.mock_sim_label
                    + " not included in the archive. Available options: ",
                    archive_mock.list_sim,
                )
            else:
                testing_data = archive_mock.get_testing_data(
                    args.mock_sim_label, z_max=args.z_max
                )
            if len(testing_data) == 0:
                raise ValueError(
                    "could not set testing data from", args.mock_sim_label
                )
            # reset all archives to free space
            archive = None
            archive_mock = None
        else:
            archive = None
            testing_data = None

        # distribute testing_data to all tasks
        for irank in range(1, size):
            comm.send(testing_data, dest=irank, tag=irank + 1)
    else:
        # get testing_data from task 0
        testing_data = comm.recv(source=0, tag=rank + 1)

    data, extra_data = set_p1ds(args, testing_data=testing_data)

    if rank == 0:
        multi_time = str(np.round(time.time() - start, 2))
        fprint("P1D set in " + multi_time + " s")

    #######################

    # set fiducial cosmology=
    cosmo_fid = set_fid_cosmo(args)

    #######################

    # set likelihood
    like = set_like(args, emulator, data, extra_data, cosmo_fid, fprint=fprint)

    #######################

    # sample likelihood

    if rank == 0:
        start = time.time()

    sample(args, like, fprint=fprint)

    if rank == 0:
        multi_time = str(np.round(time.time() - start, 2))
        fprint("Sample in " + multi_time + " s \n\n")

    #######################

    if rank == 0:
        multi_time = str(np.round(time.time() - start_all, 2))
        fprint("Program took " + multi_time + " s \n\n")


if __name__ == "__main__":
    args = parse_args()
    sam_sim(args)

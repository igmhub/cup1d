import os
import configargparse
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
            "Pedersen23",
            "Pedersen21_ext",
            "Pedersen23_ext",
            "CH24",
            "Cabayol23",
            "Cabayol23_extended",
            "Nyx_v0",
            "Nyx_v0_extended",
        ],
        required=True,
        help="Type of emulator to be used",
    )
    parser.add_argument(
        "--data_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to create mock P1Ds",
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=2,
        help="Minimum redshift of P1D measurements to be analyzed",
    )
    parser.add_argument(
        "--z_max",
        type=float,
        default=4.5,
        help="Maximum redshift of P1D measurements to be analyzed",
    )
    parser.add_argument(
        "--igm_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial IGM model",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--cosmo_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial cosmology",
    )

    parser.add_argument(
        "--drop_sim",
        action="store_true",
        help="Drop data_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--add_hires",
        action="store_true",
        help="Include high-res data (Karacayli2022)",
    )
    parser.add_argument(
        "--apply_smoothing",
        default=None,
        type=bool,
        required=False,
        help="Apply smoothing to data, None for whatever is best for selected emulator",
    )

    # likelihood
    parser.add_argument(
        "--cov_label",
        type=str,
        default="Chabanier2019",
        choices=["Chabanier2019", "QMLE_Ohio"],
        required=False,
        help="Data covariance",
    )
    parser.add_argument(
        "--cov_label_hires",
        type=str,
        default="Karacayli2022",
        choices=["Karacayli2022"],
        required=False,
        help="Data covariance for high-res data",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to P1D mock according to covariance matrix",
    )
    parser.add_argument(
        "--seed_noise",
        type=int,
        default=0,
        help="Seed for noise",
    )
    parser.add_argument(
        "--fix_cosmo",
        action="store_true",
        help="Fix cosmological parameters while sampling",
    )

    parser.add_argument(
        "--version",
        default="v3",
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
    parser.add_argument(
        "--n_burn_in",
        type=int,
        default=0,
        help="For emcee, n_burn_in",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=0,
        help="For emcee, n_steps",
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
    dict_training_set = {
        "Pedersen21": "Pedersen21",
        "Pedersen23": "Cabayol23",
        "Pedersen21_ext": "Cabayol23",
        "Pedersen23_ext": "Cabayol23",
        "CH24": "Cabayol23",
        "Cabayol23": "Cabayol23",
        "Cabayol23_extended": "Cabayol23",
        "Nyx_v0": "Nyx23_Oct2023",
        "Nyx_v0_extended": "Nyx23_Oct2023",
    }
    dict_apply_smoothing = {
        "Pedersen21": False,
        "Pedersen23": True,
        "Pedersen21_ext": False,
        "Pedersen23_ext": True,
        "CH24": True,
        "Cabayol23": True,
        "Cabayol23_extended": True,
        "Nyx_v0": True,
        "Nyx_v0_extended": True,
    }

    args.training_set = dict_training_set[args.emulator_label]
    if args.apply_smoothing is None:
        args.apply_smoothing = dict_apply_smoothing[args.emulator_label]
    else:
        args.apply_smoothing = args.apply_smoothing

    return args

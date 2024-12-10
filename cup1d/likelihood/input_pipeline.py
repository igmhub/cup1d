import sys, os
import configargparse
from cup1d.utils.utils import create_print_function, mpi_hello_world


def parse_args():
    """
    Parse input arguments

    Returns
    -------
    args : Namespace
        Namespace of input arguments
    """
    parser = configargparse.ArgumentParser(
        description="Passing options to sampler"
    )

    # emulator
    parser.add_argument(
        "--emulator_label",
        default=None,
        choices=[
            "Pedersen21",
            "Pedersen21_ext",
            "Pedersen21_ext8",
            "Pedersen23",
            "Pedersen23_ext",
            "Pedersen23_ext8",
            "CH24",
            "Cabayol23",
            "Cabayol23_extended",
            "Cabayol23+",  # recommended for mpg
            "Cabayol23+_extended",  # recommended for mpg small scales
            "Nyx_v0",
            # "Nyx_v0_extended",
            "Nyx_alphap",  # recommended for nyx
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
        "--data_label_hires",
        default=None,
        type=str,
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
        "--fid_igm_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial IGM model",
    )
    parser.add_argument(
        "--true_igm_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set IGM model to create mock",
    )
    parser.add_argument(
        "--n_igm",
        type=int,
        default=2,
        help="Number of free parameters for IGM model",
    )
    parser.add_argument(
        "--n_metals",
        type=int,
        default=0,
        help="Number of free parameters for SiIII metal contamination",
    )
    parser.add_argument(
        "--true_SiIII",
        type=float,
        default=-10,
        help="Metal contamination to create mock",
    )
    parser.add_argument(
        "--fid_SiIII",
        type=float,
        default=-10,
        help="Metal contamination to set fiducial",
    )
    parser.add_argument(
        "--true_SiII",
        type=float,
        default=-10,
        help="Metal contamination to create mock",
    )
    parser.add_argument(
        "--fid_SiII",
        type=float,
        default=-10,
        help="Metal contamination to set fiducial",
    )
    parser.add_argument(
        "--true_HCD",
        type=float,
        default=-10,
        help="HCD contamination to create mock",
    )
    parser.add_argument(
        "--fid_HCD",
        type=float,
        default=-6,
        help="HCD contamination to set fiducial",
    )

    parser.add_argument(
        "--n_dla",
        type=int,
        default=0,
        help="Number of free parameters for DLA contamination",
    )

    parser.add_argument(
        "--fid_cosmo_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set fiducial cosmology",
    )
    parser.add_argument(
        "--true_cosmo_label",
        default=None,
        type=str,
        required=True,
        help="Input simulation to set true cosmology for mock",
    )

    parser.add_argument(
        "--drop_sim",
        action="store_true",
        help="Drop data_label simulation from the training set",
    )

    # P1D
    parser.add_argument(
        "--apply_smoothing",
        default=None,
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
        "--vary_alphas",
        action="store_true",
        help="Fit running power spectrum",
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
        "--explore",
        action="store_true",
        help="Save all chains",
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
        "Pedersen21_ext": "Cabayol23",
        "Pedersen21_ext8": "Cabayol23",
        "Pedersen23": "Pedersen21",
        "Pedersen23_ext": "Cabayol23",
        "Pedersen23_ext8": "Cabayol23",
        "CH24": "Cabayol23",
        "Cabayol23": "Cabayol23",
        "Cabayol23_extended": "Cabayol23",
        "Cabayol23+": "Cabayol23",
        "Cabayol23+_extended": "Cabayol23",
        "Nyx_v0": "Nyx23_Oct2023",
        # "Nyx_v0_extended": "Nyx23_Oct2023",
        "Nyx_alphap": "Nyx23_Oct2023",
    }
    dict_apply_smoothing = {
        "Pedersen21": False,
        "Pedersen21_ext": False,
        "Pedersen21_ext8": False,
        "Pedersen23": True,
        "Pedersen23_ext": True,
        "Pedersen23_ext8": True,
        "CH24": True,
        "Cabayol23": True,
        "Cabayol23_extended": True,
        "Cabayol23+": True,
        "Cabayol23+_extended": True,
        "Nyx_v0": True,
        # "Nyx_v0_extended": True,
        "Nyx_alphap": True,
    }

    args.training_set = dict_training_set[args.emulator_label]
    if args.apply_smoothing is None:
        args.apply_smoothing = dict_apply_smoothing[args.emulator_label]
    else:
        if args.apply_smoothing == "True":
            args.apply_smoothing = True
        else:
            args.apply_smoothing = False

    if args.test:
        args.explore = True

    return args


class Args:
    """
    Class to store input arguments
    """

    def __init__(
        self,
        archive=None,
        emulator=None,
        training_set="Pedersen21",
        emulator_label="Pedersen21",
        data_label="mpg_central",
        p1d_fname=None,
        data_label_hires=None,
        z_min=0,
        z_max=10,
        fid_igm_label_mF="mpg_central",
        fid_igm_label_T="mpg_central",
        fid_igm_label_kF="mpg_central",
        true_igm_label=None,
        n_tau=2,
        n_sigT=2,
        n_gamma=2,
        n_kF=2,
        n_SiIII=0,
        n_SiII=0,
        n_d_SiIII=0,
        n_d_SiII=0,
        n_dla=0,
        n_sn=0,
        n_agn=0,
        ic_correction=False,
        igm_priors="hc",
        fid_cosmo_label="mpg_central",
        true_cosmo_label=None,
        fid_SiIII=[[0, 0], [2, -10]],
        true_SiIII=[[0, 0], [2, -10]],
        fid_SiII=[[0, 0], [2, -10]],
        true_SiII=[[0, 0], [2, -10]],
        fid_HCD=[0, -4],
        true_HCD=[0, -4],
        fid_SN=[0, -4],
        true_SN=[0, -4],
        fid_AGN=[0, -5],
        true_AGN=[0, -5],
        drop_sim=False,
        apply_smoothing=False,
        cov_label="Chabanier2019",
        cov_label_hires="Karacayli2022",
        add_noise=False,
        seed_noise=0,
        fix_cosmo=False,
        vary_alphas=False,
        prior_Gauss_rms=None,
        emu_cov_factor=0,
        verbose=True,
        test=False,
        explore=False,
        parallel=True,
        n_burn_in=0,
        n_steps=0,
    ):
        # see sam_sim to see what each parameter means
        self.archive = archive
        self.emulator = emulator
        self.training_set = training_set
        self.emulator_label = emulator_label
        self.data_label = data_label
        self.p1d_fname = p1d_fname
        self.data_label_hires = data_label_hires
        self.z_min = z_min
        self.z_max = z_max
        self.true_igm_label = true_igm_label
        self.fid_igm_label_mF = fid_igm_label_mF
        self.fid_igm_label_T = fid_igm_label_T
        self.fid_igm_label_kF = fid_igm_label_kF
        self.n_tau = n_tau
        self.n_sigT = n_sigT
        self.n_gamma = n_gamma
        self.n_kF = n_kF
        self.n_SiIII = n_SiIII
        self.n_SiII = n_SiII
        self.n_d_SiIII = n_d_SiIII
        self.n_d_SiII = n_d_SiII
        self.n_dla = n_dla
        self.n_sn = n_sn
        self.n_agn = n_agn
        self.igm_priors = igm_priors
        self.fid_SiIII = fid_SiIII
        self.true_SiIII = true_SiIII
        self.fid_SiII = fid_SiII
        self.true_SiII = true_SiII
        self.fid_HCD = fid_HCD
        self.true_HCD = true_HCD
        self.fid_SN = fid_SN
        self.true_SN = true_SN
        self.fid_AGN = fid_AGN
        self.true_AGN = true_AGN
        self.ic_correction = ic_correction
        self.fid_cosmo_label = fid_cosmo_label
        self.true_cosmo_label = true_cosmo_label
        self.drop_sim = drop_sim
        self.apply_smoothing = apply_smoothing
        self.cov_label = cov_label
        self.cov_label_hires = cov_label_hires
        self.add_noise = add_noise
        self.seed_noise = seed_noise
        self.fix_cosmo = fix_cosmo
        self.vary_alphas = vary_alphas
        self.prior_Gauss_rms = prior_Gauss_rms
        self.emu_cov_factor = emu_cov_factor
        self.verbose = verbose
        self.test = test
        self.explore = explore
        self.parallel = parallel
        self.n_burn_in = n_burn_in
        self.n_steps = n_steps

    #     self.par2save = [
    #         "emulator_label",
    #         "data_label",
    #         "data_label_hires",
    #         "z_min",
    #         "z_max",
    #         "fid_igm_label",
    #         "true_igm_label",
    #         "fix_cosmo",
    #         "cov_label",
    #         "cov_label_hires",
    #     ]

    # def save(self):
    #     out = {}
    #     for par in self.par2save:
    #         out[par] = getattr(self, par)
    #     return out

    def check_emulator_label(self):
        avail_emulator_label = [
            "Pedersen21",
            "Pedersen21_ext",
            "Pedersen21_ext8",
            "Pedersen23",
            "Pedersen23_ext",
            "Pedersen23_ext8",
            "CH24",
            "Cabayol23",
            "Cabayol23_extended",
            "Cabayol23+",
            "Cabayol23+_extended",
            "Nyx_v0",
            # "Nyx_v0_extended",
            "Nyx_alphap",
            "Nyx_alphap_cov",
        ]
        if self.emulator_label not in avail_emulator_label:
            raise ValueError(
                "emulator_label " + self.emulator_label + " not implemented"
            )

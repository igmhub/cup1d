import sys, os, configargparse
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class Args:
    """
    Class to store input arguments
    """

    archive: str | None = None
    emulator: str | None = None
    training_set: str = "Pedersen21"
    nyx_training_set: str = "models_Nyx_Mar2025_with_CGAN_val_3axes"
    emulator_label: str = "CH24_nyxcen_gpr"
    data_label: str = "mpg_central"
    p1d_fname: str | None = None
    cov_syst_type: str = "red"
    data_label_hires: str | None = None
    z_min: float = 0
    z_max: float = 10
    ic_correction: bool = False
    fid_cosmo_label: str = "mpg_central"
    true_cosmo_label: str | None = None
    fix_cosmo: bool = False
    drop_sim: bool = False
    apply_smoothing: bool = False
    cov_label: str = "Chabanier2019"
    cov_label_hires: str = "Karacayli2022"
    use_star_priors: Optional[dict] = None
    add_noise: bool = False
    seed_noise: int = 0
    vary_alphas: bool = False
    prior_Gauss_rms: float | None = None
    emu_cov_factor: int | None = 1
    cov_factor: int = 1
    emu_cov_type: str = "full"
    verbose: bool = True
    test: bool = False
    explore: bool = False
    parallel: bool = True
    n_burn_in: int = 0
    n_steps: int = 0
    z_star: float = 3
    kp_kms: float = 0.009
    fid_igm: dict = field(default_factory=lambda: {})
    true_igm: dict = field(default_factory=lambda: {})
    fid_cont: dict = field(default_factory=lambda: {})
    true_cont: dict = field(default_factory=lambda: {})
    fid_syst: dict = field(default_factory=lambda: {})
    true_syst: dict = field(default_factory=lambda: {})
    Gauss_priors: dict | None = None
    metal_lines: list[str] = field(
        default_factory=lambda: [
            "Lya_SiIII",
            "Lya_SiIIa",
            "Lya_SiIIb",
            "Lya_SiIIc",
            "SiIIa_SiIIb",
            "SiIIa_SiIII",
            "SiIIb_SiIII",
            "SiIIc_SiIII",
            "CIVa_CIVb",
            "MgIIa_MgIIb",
        ]
    )

    def __post_init__(self):
        # Setting up fiducial and true values of IGM parameters
        igms = [self.fid_igm, self.true_igm]
        for ii in range(2):
            igms[ii]["priors"] = "hc"
            igms[ii]["label_mF"] = "mpg_central"
            igms[ii]["mF_model_type"] = "pivot"
            igms[ii]["n_tau"] = 2
            igms[ii]["mF"] = [0, 0]

            igms[ii]["label_T"] = "mpg_central"
            igms[ii]["n_sigT"] = 1
            igms[ii]["sigT"] = [0, 0]
            igms[ii]["n_gamma"] = 2
            igms[ii]["gamma"] = [0, 0]

            igms[ii]["label_kF"] = "mpg_central"
            igms[ii]["n_kF"] = 1
            igms[ii]["kF"] = [0, 0]

        # Setting up fiducial and true values of nuisance parameters
        conts = [self.fid_cont, self.true_cont]
        for ii in range(2):
            # each metal line
            for metal_line in self.metal_lines:
                conts[ii][metal_line + "_X"] = [0, -10.5]
                conts[ii][metal_line + "_A"] = [0, 0]
                conts[ii]["n_x_" + metal_line] = 0
                conts[ii]["n_a_" + metal_line] = 0

            # same for dlas
            conts[ii]["hcd_model_type"] = "new"
            conts[ii]["n_d_dla1"] = 0
            conts[ii]["HCD_damp1"] = [0, 0]

            conts[ii]["n_s_dla1"] = 0
            conts[ii]["HCD_scale1"] = [0, 0]

            conts[ii]["n_d_dla2"] = 0
            conts[ii]["HCD_damp2"] = [0, 0]

            conts[ii]["n_s_dla2"] = 0
            conts[ii]["HCD_scale2"] = [0, 0]

            conts[ii]["n_c_dla"] = 0
            conts[ii]["HCD_const"] = [0, 0]

            # and others
            conts[ii]["n_sn"] = 0
            conts[ii]["SN"] = [0, -4]

            conts[ii]["n_agn"] = 0
            conts[ii]["AGN"] = [0, -5.5]

        # Setting up fiducial and true values of systematic parameters
        systs = [self.fid_syst, self.true_syst]
        for ii in range(2):
            systs[ii]["res_model_type"] = "pivot"
            systs[ii]["n_res"] = 0
            systs[ii]["R_coeff"] = [0, 0]

    def check_emulator_label(self):
        avail_emulator_label = [
            "Pedersen21",
            "Pedersen21_ext",
            "Pedersen21_ext8",
            "Pedersen23",
            "Pedersen23_ext",
            "Pedersen23_ext8",
            "Cabayol23",
            "Cabayol23_extended",
            "Cabayol23+",
            "Cabayol23+_extended",
            "Nyx_v0",
            # "Nyx_v0_extended",
            "Nyx_alphap",
            "Nyx_alphap_cov",
            "CH24",
            "CH24_Nyx",
            "CH24_mpgcen_gpr",
            "CH24_nyxcen_gpr",
        ]
        if self.emulator_label not in avail_emulator_label:
            raise ValueError(
                "emulator_label " + self.emulator_label + " not implemented"
            )

    def set_baseline(self, fix_cosmo=True, fit_type="at_a_time"):
        fid_vals_metals = {
            "f_Lya_SiIII": -3.5,
            "s_Lya_SiIII": 5.0,
            "p_Lya_SiIII": 1.0,
            "f_Lya_SiIIa": -9.5,
            "s_Lya_SiIIa": -9,
            "p_Lya_SiIIa": 1.0,
            "f_Lya_SiIIb": -3.5,
            "s_Lya_SiIIb": 6.0,
            "p_Lya_SiIIb": 1.0,
            "f_Lya_SiIIc": -3.5,
            "s_Lya_SiIIc": -9,
            "p_Lya_SiIIc": 1.0,
            "f_SiIIa_SiIIb": -4.0,
            "s_SiIIa_SiIIb": -9,
            "f_SiIIa_SiIII": -6.0,
            "s_SiIIa_SiIII": -9,
            "p_SiIIa_SiIII": 1.0,
            "f_SiIIb_SiIII": -6.0,
            "s_SiIIb_SiIII": -9,
            "p_SiIIb_SiIII": 1.0,
            "f_SiIIc_SiIII": -6.0,
            "s_SiIIc_SiIII": -9,
            "p_SiIIc_SiIII": 1.0,
            "f_CIVa_CIVb": -4.0,
            "s_CIVa_CIVb": -9,
            "f_MgIIa_MgIIb": -4.0,
            "s_MgIIa_MgIIb": -9,
        }
        self.Gauss_priors = {}
        self.flat_priors = {}
        self.null_vals = {}

        self.cov_factor = 1
        self.cov_syst_type = "red"
        self.emu_cov_factor = 1
        self.emu_cov_type = "block"
        self.prior_Gauss_rms = None
        self.rebin_k = 6
        if fix_cosmo:
            self.fix_cosmo = True
        else:
            self.fix_cosmo = False
        self.vary_alphas = False
        self.ic_correction = False
        self.fid_cosmo_label = "Planck18"

        sim_fid = "mpg_central"
        self.fid_igm["label_mF"] = sim_fid
        self.fid_igm["label_T"] = sim_fid
        self.fid_igm["label_kF"] = sim_fid

        self.fid_igm["mF_model_type"] = "pivot"
        self.fid_cont["hcd_model_type"] = "new"

        # z at a time
        if fit_type == "at_a_time":
            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

            self.fid_igm["n_tau"] = 1
            self.fid_igm["n_gamma"] = 1
            self.fid_igm["n_sigT"] = 1
            self.fid_igm["n_kF"] = 1

            # model like lya-metal
            self.fid_cont["n_f_Lya_SiIII"] = 1
            self.fid_cont["n_s_Lya_SiIII"] = 1
            self.fid_cont["n_p_Lya_SiIII"] = 1

            self.fid_cont["n_f_Lya_SiIIa"] = 0
            self.fid_cont["n_s_Lya_SiIIa"] = 0
            self.fid_cont["n_p_Lya_SiIIa"] = 0

            self.fid_cont["n_f_Lya_SiIIb"] = 1
            self.fid_cont["n_s_Lya_SiIIb"] = 1
            self.fid_cont["n_p_Lya_SiIIb"] = 0

            self.fid_cont["n_f_Lya_SiIIc"] = 0
            self.fid_cont["n_s_Lya_SiIIc"] = 0
            self.fid_cont["n_p_Lya_SiIIc"] = 0

            # model like lya-metal (incorrectly)
            self.fid_cont["n_f_SiIIa_SiIII"] = 1
            self.fid_cont["n_s_SiIIa_SiIII"] = 0
            self.fid_cont["n_p_SiIIa_SiIII"] = 0

            self.fid_cont["n_f_SiIIb_SiIII"] = 1
            self.fid_cont["n_s_SiIIb_SiIII"] = 0
            self.fid_cont["n_p_SiIIb_SiIII"] = 0

            self.fid_cont["n_f_SiIIc_SiIII"] = 0
            self.fid_cont["n_s_SiIIc_SiIII"] = 0
            self.fid_cont["n_p_SiIIc_SiIII"] = 0

            # model like metal metal
            self.fid_cont["n_f_SiIIa_SiIIb"] = 1
            self.fid_cont["n_s_SiIIa_SiIIb"] = 0

            self.fid_cont["n_f_CIVa_CIVb"] = 1
            self.fid_cont["n_s_CIVa_CIVb"] = 0

            self.fid_cont["n_f_MgIIa_MgIIb"] = 1
            self.fid_cont["n_s_MgIIa_MgIIb"] = 0

            self.fid_cont["n_d_dla1"] = 1
            self.fid_cont["HCD_damp1"] = [0, -0.5]

            self.fid_cont["n_d_dla2"] = 1
            self.fid_cont["HCD_damp2"] = [0, 1.0]

            self.fid_cont["n_d_dla3"] = 0
            self.fid_cont["HCD_damp3"] = [0, -9.5]

            self.fid_cont["n_s_dla1"] = 1
            self.fid_cont["HCD_scale1"] = [0, 4]

            self.fid_cont["n_s_dla2"] = 1
            self.fid_cont["HCD_scale2"] = [0, 8]

            self.fid_cont["n_s_dla3"] = 0
            self.fid_cont["HCD_scale3"] = [0, 4]

            self.fid_cont["n_c_dla"] = 1
            self.fid_cont["HCD_const"] = [0, 0]

            self.flat_priors["HCD_damp"] = [[-0.5, 0.5], [-10, 5]]
            self.flat_priors["HCD_scale"] = [[-1, 1], [1, 10]]
            self.flat_priors["HCD_const"] = [[-1, 1], [-0.2, 1e-6]]
        else:
            self.fid_syst["res_model_type"] = "chunks"
            self.fid_syst["n_res"] = 11
            self.fid_syst["R_coeff"] = np.zeros((self.fid_syst["n_res"]))

            self.fid_igm["n_tau"] = 2
            self.fid_igm["n_gamma"] = 1
            self.fid_igm["n_sigT"] = 1
            self.fid_igm["n_kF"] = 1

            self.fid_cont["n_f_Lya_SiIII"] = 1
            self.fid_cont["n_s_Lya_SiIII"] = 1

            self.fid_cont["n_f_Lya_SiIIa"] = 0
            self.fid_cont["n_s_Lya_SiIIa"] = 0

            self.fid_cont["n_f_Lya_SiIIb"] = 1
            self.fid_cont["n_s_Lya_SiIIb"] = 1

            self.fid_cont["n_f_Lya_SiIIc"] = 0
            self.fid_cont["n_s_Lya_SiIIc"] = 0

            self.fid_cont["n_x_SiIIa_SiIII"] = 1
            self.fid_cont["n_s_SiIIa_SiIII"] = 0

            self.fid_cont["n_x_SiIIb_SiIII"] = 1
            self.fid_cont["n_s_SiIIb_SiIII"] = 0

            self.fid_cont["n_x_SiIIc_SiIII"] = 1
            self.fid_cont["n_s_SiIIc_SiIII"] = 0

            self.fid_cont["n_x_SiIIa_SiIIb"] = 1
            self.fid_cont["n_s_SiIIa_SiIIb"] = 0

            self.fid_cont["n_x_CIVa_CIVb"] = 0
            self.fid_cont["n_s_CIVa_CIVb"] = 0

            self.fid_cont["n_d_dla1"] = 1
            self.fid_cont["A_damp1"] = [0, -1.4]

            self.fid_cont["n_s_dla1"] = 1
            self.fid_cont["A_scale1"] = [0, 5.2]

            self.fid_cont["n_d_dla2"] = 1
            self.fid_cont["A_damp2"] = [0, -1.4]

            self.fid_cont["n_s_dla2"] = 1
            self.fid_cont["A_scale2"] = [0, 5.2]

            self.fid_cont["n_c_dla"] = 1
            self.fid_cont["A_const"] = [0, 0]

        for metal_label in self.metal_lines:
            if self.fid_cont["n_x_" + metal_label] == 0:
                self.fid_cont["f_" + metal_label] = [0, -10.5]
            else:
                self.fid_cont["f_" + metal_label] = [
                    0,
                    fid_vals_metals["f_" + metal_label],
                ]
            self.fid_cont["s_" + metal_label] = [
                0,
                fid_vals_metals["s_" + metal_label],
            ]
            if "p_" + metal_label in fid_vals_metals:
                self.fid_cont["p_" + metal_label] = [
                    0,
                    fid_vals_metals["p_" + metal_label],
                ]

            self.flat_priors["f_" + metal_label] = [[-3, 3], [-11, -1]]
            self.flat_priors["s_" + metal_label] = [[-1, 1], [-10, 10]]
            if "p_" + metal_label in fid_vals_metals:
                self.flat_priors["p_" + metal_label] = [[-1, 1], [0.95, 1.05]]

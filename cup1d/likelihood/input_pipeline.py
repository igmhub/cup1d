import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from cup1d.utils.utils import get_path_repo


@dataclass
class Args:
    """
    Class to store input arguments
    """

    data_label: str = "DESIY1_QMLE3"
    data_label_hires: str | None = None
    z_min: float = 0
    z_max: float = 10
    emulator_label: str = "CH24_mpgcen_gpr"
    drop_sim: str | None = None
    true_cosmo_label: str | None = "Planck18"
    fid_cosmo_label: str = "Planck18"
    igm_params: list[str] = field(
        default_factory=lambda: [
            "tau_eff",
            "sigT_kms",
            "gamma",
            "kF_kms",
        ]
    )
    true_igm: dict = field(
        default_factory=lambda: {
            "priors": "hc",
            "label_mF": "mpg_central",
            "label_T": "mpg_central",
            "label_kF": "mpg_central",
            "n_tau_eff": 0,
            "n_sigT_kms": 0,
            "n_gamma": 0,
            "n_kF_kms": 0,
        }
    )
    fid_igm: dict = field(
        default_factory=lambda: {
            "priors": "hc",
            "label_mF": "mpg_central",
            "label_T": "mpg_central",
            "label_kF": "mpg_central",
            "n_tau_eff": 0,
            "n_sigT_kms": 0,
            "n_gamma": 0,
            "n_kF_kms": 0,
        }
    )
    cont_params: dict = field(
        default_factory=lambda: {
            "f_Lya_SiIII": [0, -11.5],
            "s_Lya_SiIII": [0, 2],
            "f_Lya_SiII": [0, -11.5],
            "s_Lya_SiII": [0, 2],
            "f_SiIIa_SiIIb": [0, -11.5],
            "s_SiIIa_SiIIb": [0, 2],
            "f_SiIIa_SiIII": [0, 0],  # these are variations from exp(0)=1
            "f_SiIIb_SiIII": [0, 0],  # these are variations from exp(0)=1
        }
    )
    true_cont: dict = field(
        default_factory=lambda: {"hcd_model_type": "new_rogers"}
    )
    fid_cont: dict = field(
        default_factory=lambda: {"hcd_model_type": "new_rogers"}
    )
    syst_params: dict = field(
        default_factory=lambda: {
            "res": 0,
        }
    )
    true_syst: dict = field(
        default_factory=lambda: {
            "res_model_type": "pivot",
            "R_coeff": [0, 0],
            "n_res": 0,
        }
    )
    fid_syst: dict = field(
        default_factory=lambda: {
            "res_model_type": "pivot",
            "R_coeff": [0, 0],
            "n_res": 0,
        }
    )
    apply_smoothing: bool = False
    cov_label: str = "Chabanier2019"
    cov_label_hires: str = "Karacayli2022"
    nyx_training_set: str = "models_Nyx_Mar2025_with_CGAN_val_3axes"
    cov_syst_type: str = "red"
    z_star: float = 3
    kp_kms: float = 0.009
    use_star_priors: Optional[dict] = None
    add_noise: bool = False
    seed_noise: int = 0
    verbose: bool = True
    ic_correction: bool = False
    fix_cosmo: bool = False
    vary_alphas: bool = False
    prior_Gauss_rms: float | None = None
    emu_cov_factor: int | None = 1
    cov_factor: int = 1
    emu_cov_type: str = "block"
    file_ic: str | None = None
    test: bool = False
    explore: bool = False
    parallel: bool = True
    n_burn_in: int = 0
    n_steps: int = 0
    out_folder: str | None = "."
    Gauss_priors: dict | None = None

    def __post_init__(self):
        """Initialize some parameters"""
        self.check_emulator_label()
        if "nyx" in self.emulator_label:
            self.training_set = "models_Nyx_Mar2025_with_CGAN_val_3axes"
        elif "mpg" in self.emulator_label:
            self.training_set = "Cabayol23"
        else:
            self.training_set = "Pedersen21"

        if self.true_cont["hcd_model_type"] == "new_rogers":
            for jj in range(1, 5):
                self.cont_params["HCD_damp" + str(jj)] = [0, -11.5]
            self.cont_params["HCD_const"] = [0, 0]
        ##
        for key in self.cont_params.keys():
            self.true_cont[key] = self.cont_params[key]
            self.true_cont["n_" + key] = self.cont_params[key]

        # # and others
        # self.true_cont["n_sn"] = 0
        # self.true_cont["SN"] = [0, -4]

        # self.true_cont["n_agn"] = 0
        # self.true_cont["AGN"] = [0, -5.5]

    def check_emulator_label(self):
        avail_emulator_label = [
            "Pedersen21",
            "Pedersen21_ext",
            "Pedersen23",
            "Pedersen23_ext",
            "CH24_mpgcen_gpr",
            "CH24_nyxcen_gpr",
        ]
        if self.emulator_label not in avail_emulator_label:
            raise ValueError(
                "emulator_label " + self.emulator_label + " not implemented"
            )

    def set_baseline(
        self,
        fix_cosmo=True,
        fit_type="at_a_time",
        ztar=0,
        zmin=2.2,
        zmax=4.2,
        P1D_type="DESIY1_QMLE3",
        name_variation=None,
    ):
        """
        Set baseline parameters
        """
        self.ic_from_file = None
        self.fit_type = fit_type
        self.fid_syst["n_res"] = 0

        if fit_type not in [
            "global_all",  # all params from at_a_time_global
            "global_opt",  # for opt IC
            "global_igm",  # for opt IC
            "at_a_time_igm",  # only IGM parameter
            "at_a_time_orig",  # all possible parameters varied
            "at_a_time_global",  # vary same parameters as in global fits
        ]:
            raise ValueError("fit_type " + fit_type + " not implemented")

        if P1D_type.startswith("DESIY1"):
            self.P1D_type = P1D_type
            self.cov_syst_type = "red"
            self.z_min = 2.1
            self.z_max = 4.3

            path_in_challenge = os.path.join(
                os.path.dirname(get_path_repo("cup1d")), "data", "in_DESI_DR1"
            )
            path_out_challenge = os.path.join(
                os.path.dirname(get_path_repo("cup1d")),
                "data",
                "out_DESI_DR1",
            )

            self.ic_from_file = os.path.join(
                path_out_challenge,
                "ic",
                "allz_snr3_nocosmo_" + fit_type,
                "res.npy",
            )

            if name_variation is None:
                self.out_folder = os.path.join(
                    os.path.dirname(get_path_repo("cup1d")),
                    "data",
                    "out_DESI_DR1",
                    self.P1D_type,
                    self.fit_type,
                    self.emulator_label,
                )
            else:
                self.out_folder = os.path.join(
                    os.path.dirname(get_path_repo("cup1d")),
                    "data",
                    "out_DESI_DR1",
                    name_variation,
                )

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            self.ic_from_file = None

        # set all cont to zero
        cont_props = [
            "n_HCD_damp1",
            "n_HCD_damp2",
            "n_HCD_damp3",
            "n_HCD_damp4",
            "n_HCD_const",
            "n_f_Lya_SiIII",
            "n_s_Lya_SiIII",
            "n_f_Lya_SiII",
            "n_s_Lya_SiII",
            "n_f_SiIIa_SiIIb",
            "n_s_SiIIa_SiIIb",
            "n_f_SiIIa_SiIII",
            "n_f_SiIIb_SiIII",
        ]
        self.fid_cont["HCD_const_otype"] = "const"
        for prop in cont_props:
            self.fid_cont[prop] = 0

        # set igm stuff
        igm_props = ["n_tau_eff", "n_sigT_kms", "n_gamma", "n_kF_kms"]
        self.fid_igm["tau_eff_otype"] = "exp"
        self.fid_igm["gamma_otype"] = "const"
        self.fid_igm["sigT_kms_otype"] = "const"
        self.fid_igm["kF_kms_otype"] = "const"
        for prop in igm_props:
            self.fid_igm[prop] = 0

        self.null_vals = {}

        self.cov_factor = 1
        self.cov_syst_type = "red"
        self.emu_cov_factor = 1
        self.emu_cov_type = "block"
        self.prior_Gauss_rms = None
        self.rebin_k = 8  # good enough to 0.06 chi2
        if fix_cosmo:
            self.fix_cosmo = True
        else:
            self.fix_cosmo = False
        self.vary_alphas = False
        self.ic_correction = False

        if name_variation is None:
            self.fid_cosmo_label = "Planck18"
        elif name_variation == "cosmo":
            self.fid_cosmo_label = "DESIDR2_ACT"
        elif name_variation.startswith("sim_"):
            pass
        else:
            self.fid_cosmo_label = "Planck18"
        print("fid cosmo label", self.fid_cosmo_label)

        if ("mpg" in self.emulator_label) | ("Mpg" in self.emulator_label):
            sim_fid = "mpg_central"
        elif ("nyx" in self.emulator_label) | ("Nyx" in self.emulator_label):
            sim_fid = "nyx_central"
        else:
            sim_fid = "mpg_central"

        self.fid_igm["label_mF"] = sim_fid
        self.fid_igm["label_T"] = sim_fid
        self.fid_igm["label_kF"] = sim_fid

        self.fid_cont["hcd_model_type"] = "new_rogers"

        props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
        props_cont = [
            "f_Lya_SiIII",
            "s_Lya_SiIII",
            "f_Lya_SiII",
            "s_Lya_SiII",
            "f_SiIIa_SiIIb",
            "s_SiIIa_SiIIb",
            "f_SiIIa_SiIII",
            "f_SiIIb_SiIII",
            "HCD_damp1",
            "HCD_damp2",
            "HCD_damp3",
            "HCD_damp4",
        ]

        self.fid_cont["z_max"] = {}

        # z at a time
        #############
        if fit_type == "at_a_time_global":
            baseline_prop = [
                "tau_eff",
                "sigT_kms",
                "gamma",
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp4",
            ]

            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            print("baseline_prop", baseline_prop)

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                # "HCD_damp2",
                # "HCD_damp3",
                "HCD_damp4",
                # "HCD_const",
            ]
            for prop in props_igm:
                if prop in baseline_prop:
                    self.fid_igm["n_" + prop] = 1
                else:
                    self.fid_igm["n_" + prop] = 0
                self.fid_igm[prop + "_ztype"] = "pivot"

            for prop in props_cont:
                if prop in baseline_prop:
                    self.fid_cont["n_" + prop] = 1
                else:
                    self.fid_cont["n_" + prop] = 0
                self.fid_cont[prop + "_ztype"] = "pivot"
                if prop != "HCD_const":
                    self.fid_cont[prop] = [0, -11.5]
                else:
                    self.fid_cont[prop] = [0, 0]

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1.2)]
        #############

        elif fit_type == "at_a_time_igm":
            raise ("Need to check", fit_type)
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            for prop in props_igm:
                self.fid_igm["n_" + prop] = 1
                self.fid_igm[prop + "_ztype"] = "pivot"

            for prop in props_cont:
                self.fid_cont["n_" + prop] = 0
                self.fid_cont[prop + "_ztype"] = "pivot"
                if prop != "HCD_const":
                    self.fid_cont[prop] = [0, -11.5]
                else:
                    self.fid_cont[prop] = [0, 0]

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

        #############

        elif fit_type == "global_all":
            if "mpg" in self.emulator_label:
                fname = "mpg_ic_at_a_time.npy"
            else:
                fname = "nyx_ic_at_a_time.npy"
            self.file_ic = os.path.join(
                os.path.dirname(get_path_repo("cup1d")), "data", "ics", fname
            )

            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
            props_cont = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp4",
            ]

            self.opt_props = props_igm + props_cont
            nodes = np.array(
                [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]
            )

            self.fid_igm["tau_eff_znodes"] = nodes
            self.fid_igm["sigT_kms_znodes"] = nodes
            self.fid_igm["gamma_znodes"] = nodes
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            for prop in props_cont:
                self.fid_cont[prop + "_znodes"] = nodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_spl"

        #############
        elif fit_type == "global_opt":
            if "mpg" in self.emulator_label:
                fname = "mpg_ic_global_red.npy"
            else:
                fname = "nyx_ic_global_red.npy"
            self.file_ic = os.path.join(
                os.path.dirname(get_path_repo("cup1d")), "data", "ics", fname
            )

            if (name_variation is not None) and (
                name_variation.startswith("sim_")
            ):
                self.file_ic = None

            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]

            if name_variation == "HCD":
                props_cont = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    "f_SiIIa_SiIII",
                    "f_SiIIb_SiIII",
                    # "HCD_damp1",
                    # "HCD_damp4",
                ]
            elif name_variation == "metal_trad":
                props_cont = [
                    "f_Lya_SiIII",
                    # "s_Lya_SiIII",
                    "f_Lya_SiII",
                    # "s_Lya_SiII",
                    # "f_SiIIa_SiIIb",
                    # "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp4",
                ]
            elif name_variation == "metal_si2":
                props_cont = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    # "f_SiIIa_SiIIb",
                    # "s_SiIIa_SiIIb",
                    "f_SiIIa_SiIII",
                    "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp4",
                ]
            elif name_variation == "metal_deco":
                props_cont = [
                    "f_Lya_SiIII",
                    # "s_Lya_SiIII",
                    "f_Lya_SiII",
                    # "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    "f_SiIIa_SiIII",
                    "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp4",
                ]
            elif name_variation == "metal_thin":
                props_cont = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp4",
                ]
            else:
                props_cont = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    "f_SiIIa_SiIII",
                    "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp4",
                ]

            self.opt_props = props_igm + props_cont
            var_props = self.opt_props

            # nodes = np.geomspace(2.2, 4.2, 3)
            nodes = np.geomspace(zmin, zmax, 1)

            self.fid_igm["tau_eff_znodes"] = np.geomspace(zmin, zmax, 6)
            self.fid_igm["sigT_kms_znodes"] = np.geomspace(zmin, zmax, 4)
            self.fid_igm["gamma_znodes"] = np.geomspace(zmin, zmax, 4)
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_lin"

                if prop not in var_props:
                    self.fid_igm[prop + "_fixed"] = True
                else:
                    self.fid_igm[prop + "_fixed"] = False

            for prop in props_cont:
                self.fid_cont[prop + "_znodes"] = nodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_lin"

                if prop not in var_props:
                    self.fid_cont[prop + "_fixed"] = True
                else:
                    self.fid_cont[prop + "_fixed"] = False

        elif fit_type == "global_igm":
            if "mpg" in self.emulator_label:
                fname = "mpg_ic_global_red.npy"
            else:
                fname = "nyx_ic_global_red.npy"
            self.file_ic = os.path.join(
                os.path.dirname(get_path_repo("cup1d")), "data", "ics", fname
            )
            if name_variation.startswith("sim_"):
                self.file_ic = None

            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5
            props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]

            props_cont = [
                # "f_Lya_SiIII",
                # "s_Lya_SiIII",
                # "f_Lya_SiII",
                # "s_Lya_SiII",
                # "f_SiIIa_SiIIb",
                # "s_SiIIa_SiIIb",
                # "f_SiIIa_SiIII",
                # "f_SiIIb_SiIII",
                # "HCD_damp1",
                # "HCD_damp4",
            ]

            self.opt_props = props_igm + props_cont
            var_props = self.opt_props
            # nodes = np.geomspace(2.2, 4.2, 3)
            nodes = np.geomspace(2.2, 4.2, 1)

            self.fid_igm["tau_eff_znodes"] = np.linspace(2.25, 4.25, 9)
            self.fid_igm["sigT_kms_znodes"] = []
            self.fid_igm["gamma_znodes"] = []
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_lin"

                if prop not in var_props:
                    self.fid_igm[prop + "_fixed"] = True
                else:
                    self.fid_igm[prop + "_fixed"] = False

            for prop in props_cont:
                self.fid_cont[prop + "_znodes"] = nodes
                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                    if prop != "HCD_const":
                        self.fid_cont[prop] = [0, -11.5]
                    else:
                        self.fid_cont[prop] = [0, 0]
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_lin"

                if prop not in var_props:
                    self.fid_cont[prop + "_fixed"] = True
                else:
                    self.fid_cont[prop + "_fixed"] = False

        #############

        else:
            raise ValueError("Fit type not recognized")

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            fid_vals_conts = {
                "f_Lya_SiIII": -11.5,
                "s_Lya_SiIII": 2.1,
                "f_Lya_SiII": -11.5,
                "s_Lya_SiII": 2.1,
                "f_SiIIa_SiIIb": -11.5,
                "s_SiIIa_SiIIb": 0.1,
                "f_SiIIa_SiIII": 0,
                "f_SiIIb_SiIII": 0,
                "HCD_const": 0,
                "HCD_damp1": -9.5,
                "HCD_damp2": -11.5,
                "HCD_damp3": -11.5,
                "HCD_damp4": -9.5,
            }
        else:
            fid_vals_conts = {
                "f_Lya_SiIII": -4.25,
                "s_Lya_SiIII": 4.75,
                "f_Lya_SiII": -4.5,
                "s_Lya_SiII": 4.75,
                "f_SiIIa_SiIIb": -0.5,
                "s_SiIIa_SiIIb": 4.75,
                "f_SiIIa_SiIII": 0,
                "f_SiIIb_SiIII": 0,
                "HCD_const": 0,
                "HCD_damp1": -1.2,
                "HCD_damp2": -11.5,
                "HCD_damp3": -11.5,
                "HCD_damp4": -4,
            }
        # add_lines = [
        #     "SiIIa_SiIIb",
        #     "CIVa_CIVb",
        #     "MgIIa_MgIIb",
        # ]

        for key in self.cont_params.keys():
            if self.fid_cont["n_" + key] == 0:
                self.fid_cont[key] = self.cont_params[key]
            else:
                self.fid_cont[key] = [
                    0,
                    fid_vals_conts[key],
                ]

        self.fid_cont["flat_priors"] = {}

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
                [-1, 1],
                [-11.5, -2],
            ]
        else:
            self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
                [-1, 1],
                [-6, -2],
            ]

        self.fid_cont["flat_priors"]["s_Lya_SiIII"] = [
            [-1, 1],
            [2, 7],
        ]

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            self.fid_cont["flat_priors"]["f_Lya_SiII"] = [
                [-1, 1],
                [-11.5, -2],
            ]
        else:
            self.fid_cont["flat_priors"]["f_Lya_SiII"] = [
                [-1, 1],
                [-6, -2],
            ]
        self.fid_cont["flat_priors"]["s_Lya_SiII"] = [
            [-1, 1],
            [2, 7],
        ]

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            self.fid_cont["flat_priors"]["f_SiIIa_SiIIb"] = [
                [-1, 4],
                [-11.5, -3],
            ]
        else:
            self.fid_cont["flat_priors"]["f_SiIIa_SiIIb"] = [
                [-1, 4],
                [-3, 3],
            ]

        self.fid_cont["flat_priors"]["s_SiIIa_SiIIb"] = [
            [-1, 3],
            [0, 5.5],
        ]

        self.fid_cont["flat_priors"]["f_SiIIa_SiIII"] = [
            [-1, 2],
            [-1, 4],
        ]
        self.fid_cont["flat_priors"]["f_SiIIb_SiIII"] = [
            [-1, 1],
            [-1, 4],
        ]

        # priors
        # -0.03, 75% of all fluctuations
        self.fid_cont["flat_priors"]["HCD_damp1"] = [[-0.5, 0.5], [-10.0, -1.0]]
        self.fid_cont["flat_priors"]["HCD_damp2"] = [[-0.5, 0.5], [-10.0, -1.0]]
        self.fid_cont["flat_priors"]["HCD_damp3"] = [[-0.5, 0.5], [-10.0, -1.0]]
        self.fid_cont["flat_priors"]["HCD_damp4"] = [[-0.5, 0.5], [-10.0, -1.0]]
        self.fid_cont["flat_priors"]["HCD_const"] = [[-1, 1], [-0.2, 1e-6]]

        for key in self.fid_cont:
            if key in self.fid_cont["flat_priors"]:
                if (
                    self.fid_cont[key][-1]
                    < self.fid_cont["flat_priors"][key][-1][0]
                ):
                    print(
                        key,
                        self.fid_cont["flat_priors"][key][-1][0],
                        self.fid_cont[key][-1],
                    )
                    self.fid_cont["flat_priors"][key][-1][0] = (
                        self.fid_cont[key][-1] - 0.1
                    )
                if (
                    self.fid_cont[key][-1]
                    > self.fid_cont["flat_priors"][key][-1][1]
                ):
                    print(
                        key,
                        self.fid_cont["flat_priors"][key][-1][1],
                        self.fid_cont[key][-1],
                    )
                    self.fid_cont["flat_priors"][key][-1][1] = (
                        self.fid_cont[key][-1] + 0.1
                    )


# Set Gaussian priors
# # self.prior_Gauss_rms = 0.1
# self.prior_Gauss_rms = None

# self.Gauss_priors = {}
# self.Gauss_priors["ln_tau_0"] = [10]
# # self.Gauss_priors["ln_sigT_kms_0"] = [0.02]
# # self.Gauss_priors["ln_gamma_0"] = [0.08]
# # self.Gauss_priors["ln_kF_0"] = [0.003]

# f_Gprior = {
#     "Lya_SiIII": 1,
#     "Lya_SiIIa": 1,
#     "Lya_SiIIb": 1,
#     "SiIIa_SiIIb": 3,
#     "SiIIa_SiIII": 4,
#     "SiIIb_SiIII": 3,
# }

# d_Gprior = {
#     "Lya_SiIII": 1.5,
#     "Lya_SiIIa": 0.05,
#     "Lya_SiIIb": 0.05,
#     "SiIIa_SiIIb": 1,
#     "SiIIa_SiIII": 0.03,
#     "SiIIb_SiIII": 1,
# }

# a_Gprior = {
#     "Lya_SiIII": 10,
#     "Lya_SiIIa": 10,
#     "Lya_SiIIb": 10,
#     "SiIIa_SiIIb": 2,
#     "SiIIa_SiIII": 0.05,
#     "SiIIb_SiIII": 0.01,
# }

# for metal_line in lines_use:
#     self.Gauss_priors["ln_x_"+metal_line+"_0"] = [f_Gprior[metal_line]]
#     self.Gauss_priors["d_"+metal_line+"_0"] = [d_Gprior[metal_line]]
#     self.Gauss_priors["a_"+metal_line+"_0"] = [a_Gprior[metal_line]]
# self.Gauss_priors["ln_A_damp_0"] = [0.3]
# self.Gauss_priors["ln_A_scale_0"] = [1]
# self.Gauss_priors["R_coeff_0"] = [2]

# self.Gauss_priors = {}

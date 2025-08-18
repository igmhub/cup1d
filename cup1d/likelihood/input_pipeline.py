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
            "f_SiIIa_SiIII": [0, 0],
            "f_SiIIb_SiIII": [0, 0],
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
    cov_syst_type: str = "red"
    z_star: float = 3
    kp_kms: float = 0.009
    use_star_priors: Optional[dict] = None
    add_noise: bool = False
    seed_noise: int = 0
    verbose: bool = True
    ic_correction: bool = False
    fid_cosmo_label: str = "mpg_central"
    fix_cosmo: bool = False
    vary_alphas: bool = False
    prior_Gauss_rms: float | None = None
    emu_cov_factor: int | None = 1
    cov_factor: int = 1
    emu_cov_type: str = "block"
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
            "andreu2",
            "global",
            "at_a_time",
            "at_at_time_igm",
            "at_a_time_orig",
            "wip",
            "andreu",
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
        igm_props = ["n_tau_eff", "n_sigT", "n_gamma", "n_kF_kms"]

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
        self.fid_cosmo_label = "Planck18"

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
        if fit_type == "at_a_time":
            baseline_prop = [
                "tau_eff",
                "sigT_kms",
                # "gamma",
                # "kF_kms",
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

            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            if ztar > 3.5:
                baseline_prop.remove("f_SiIIa_SiIII")
                baseline_prop.remove("f_SiIIb_SiIII")
                self.fid_cont["z_max"]["f_SiIIa_SiIII"] = 3.5
                self.fid_cont["z_max"]["f_SiIIb_SiIII"] = 3.5
                baseline_prop.remove("f_Lya_SiII")
                baseline_prop.remove("s_Lya_SiII")
                self.fid_cont["z_max"]["f_Lya_SiII"] = 3.5
                self.fid_cont["z_max"]["s_Lya_SiII"] = 3.5

            if ztar > 3.7:
                baseline_prop.remove("f_SiIIa_SiIIb")
                baseline_prop.remove("s_SiIIa_SiIIb")
                self.fid_cont["z_max"]["f_SiIIa_SiIIb"] = 3.7
                self.fid_cont["z_max"]["s_SiIIa_SiIIb"] = 3.7

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
                "HCD_damp2",
                # "HCD_damp3",
                "HCD_damp4",
                # "HCD_const",
            ]
            # self.fid_igm["tau_eff_znodes"] = []
            # self.fid_igm["sigT_kms_znodes"] = []
            # self.fid_igm["gamma_znodes"] = []
            # self.fid_igm["kF_kms_znodes"] = []
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

            print(self.fid_cont["n_HCD_const"])

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "at_a_time_igm":
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

        elif fit_type == "at_a_time_orig":
            for prop in props_cont:
                self.fid_cont["z_max"][prop] = 5

            for prop in props_igm:
                self.fid_igm["n_" + prop] = 1
                self.fid_igm[prop + "_ztype"] = "pivot"

            for prop in props_cont:
                self.fid_cont["n_" + prop] = 1
                self.fid_cont[prop + "_ztype"] = "pivot"
                if prop != "HCD_const":
                    self.fid_cont[prop] = [0, -11.5]
                else:
                    self.fid_cont[prop] = [0, 0]

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 1
            self.fid_syst["R_coeff"] = [0, 0]

        elif fit_type == "wip":
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

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.5
            z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.7

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.linspace(2.2, 4.2, 8)
            self.fid_igm["sigT_kms_znodes"] = np.linspace(2.2, 4.2, 10)
            self.fid_igm["gamma_znodes"] = []
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            # self.fid_igm["n_kF_kms"] = 0

            # znodes:
            # tau: 2.2, 2.4, 3.2, 3.6, 3.8, 4.2
            # sigT: 2.2, 2.6, 4.2
            # fLya-SiIII: 2.2, 2.8, 3.2, 4.2 as a function of tau-prior? check
            # sLya-SiIII: 2.2, 4.2
            # fLya-SiII: 2.2, 2.6, 3.4
            # sLya-SiII: 2.2, 2.6, 3.4
            # fSiIIa-SiIIb: 2.2, 3.6
            # sSiIIa-SiIIb: 2.2, 2.8, 3.6
            # fSiIIa-SiIII: 2.2, 3.4
            # fSiIIb-SiIII: 2.2, 3.4
            # HCD-damp1: All?
            # HCD-damp4: 2.2, 3.2, 4.2

            # 2, 3, 4, 6, 5, 41 (3.2), 4, 10 (3.6), 10 (3.8), 8 (4.0), 4

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in ["f_SiIIa_SiIII", "f_SiIIb_SiIII"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 2)

                if prop in ["f_Lya_SiII"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 3)

                if prop in ["s_Lya_SiII"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 5)

                if prop in ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 6)

                if prop in ["f_Lya_SiIII", "s_Lya_SiIII"]:
                    _znodes = np.linspace(2.2, 4.2, 3)

                if prop in ["HCD_damp1"]:
                    _znodes = np.linspace(2.2, 4.2, 11)

                if prop in ["HCD_damp4"]:
                    _znodes = np.linspace(2.2, 4.2, 8)

                self.fid_cont[prop + "_znodes"] = _znodes
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

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "global":
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

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.5
            z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.3
            z_max_props = ["f_Lya_SiIII", "s_Lya_SiIII"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.9

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.array(
                [2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 3.8, 4.2]
            )
            self.fid_igm["sigT_kms_znodes"] = np.array([2.2, 4.2])
            self.fid_igm["gamma_znodes"] = np.array([2.2, 2.6, 4.2])
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            self.fid_igm["n_kF_kms"] = 0

            # znodes:
            # tau: 2.2, 2.4, 2.6, 3.2, 3.6, 3.8, 4.2
            # sigT: 2.2, 2.6, 4.2
            # fLya-SiIII: 2.2, 2.8, 3.2, 4.2 as a function of tau-prior? check
            # sLya-SiIII: 2.2, 4.2
            # fLya-SiII: 2.2, 2.6, 3.4
            # sLya-SiII: 2.2, 2.6, 3.4
            # fSiIIa-SiIIb: 2.2, 3.6
            # sSiIIa-SiIIb: 2.2, 2.8, 3.6
            # fSiIIa-SiIII: 2.2, 3.4
            # fSiIIb-SiIII: 2.2, 3.4
            # HCD-damp1: All?
            # HCD-damp4: 2.2, 3.2, 4.2

            # Lya-SiII: 2.2, 2.4, 2.8, DONE
            # SiII-SiII: 3.2 DONE
            # DLA: 3.4, 3.6, 3.8 DONE
            # Lya-SiIII: 3.6, 3.8 DONE

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in ["f_Lya_SiIII", "s_Lya_SiIII"]:
                    _znodes = np.array([2.2, 2.8, 3.4, 3.6, 3.8])

                if prop in ["f_SiIIa_SiIII", "f_SiIIb_SiIII"]:
                    _znodes = np.array([2.2, 3.4])

                if prop in ["f_Lya_SiII"]:
                    _znodes = np.array([2.2, 2.4, 2.6, 2.8, 3.4])
                if prop in ["s_Lya_SiII"]:
                    _znodes = np.array([2.2, 2.4, 2.8, 3.4])

                if prop in ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]:
                    _znodes = np.array([2.2, 2.8, 3.2])

                if prop in ["HCD_damp1"]:
                    _znodes = np.array([2.2, 2.8, 3.4, 3.6, 3.9, 4.2])

                if prop in ["HCD_damp4"]:
                    _znodes = np.array([2.2, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])

                self.fid_cont[prop + "_znodes"] = _znodes
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

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "andreu":
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

            z_max_props = [
                "f_SiIIa_SiIII",
                "f_SiIIb_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
            ]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.5
            z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            for prop in z_max_props:
                self.fid_cont["z_max"][prop] = 3.7

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.linspace(2.2, 4.2, 11)
            self.fid_igm["sigT_kms_znodes"] = np.linspace(2.2, 4.2, 6)
            self.fid_igm["gamma_znodes"] = []
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            # self.fid_igm["n_kF_kms"] = 0

            # znodes = [0, 1]

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in [
                    "f_SiIIa_SiIII",
                    "f_SiIIb_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                ]:
                    _znodes = np.linspace(2.2, self.fid_cont["z_max"][prop], 3)

                if prop in ["f_Lya_SiIII", "s_Lya_SiIII"]:
                    _znodes = np.linspace(2.2, 4.2, 6)

                if prop in ["HCD_damp1", "HCD_damp4"]:
                    _znodes = np.linspace(2.2, 4.2, 6)

                self.fid_cont[prop + "_znodes"] = _znodes
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

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        elif fit_type == "andreu2":
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

            # z_max_props = [
            #     "f_SiIIa_SiIII",
            #     "f_SiIIb_SiIII",
            #     "f_Lya_SiII",
            #     "s_Lya_SiII",
            # ]
            # for prop in z_max_props:
            #     self.fid_cont["z_max"][prop] = 3.5
            # z_max_props = ["f_SiIIa_SiIIb", "s_SiIIa_SiIIb"]
            # for prop in z_max_props:
            #     self.fid_cont["z_max"][prop] = 3.3
            # z_max_props = ["f_Lya_SiIII", "s_Lya_SiIII"]
            # for prop in z_max_props:
            #     self.fid_cont["z_max"][prop] = 3.9

            self.opt_props = props_igm + props_cont

            self.fid_igm["tau_eff_znodes"] = np.linspace(zmin, zmax, 6)
            self.fid_igm["sigT_kms_znodes"] = np.array([zmin, 3.0, zmax])
            self.fid_igm["gamma_znodes"] = np.array([zmin, 3.0, zmax])
            self.fid_igm["kF_kms_znodes"] = []
            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_spl"

            # do not gain anything by using splines?
            # self.fid_igm["n_tau_eff"] = 3
            # self.fid_igm["n_sigT_kms"] = 3  # what about 2?
            # self.fid_igm["n_gamma"] = 0
            self.fid_igm["n_kF_kms"] = 0

            # znodes = [0, 1]

            for prop in props_cont:
                if prop in self.opt_props:
                    _znodes = 0

                if prop in [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                ]:
                    _znodes = np.array([zmin, 2.9, zmax])

                if prop in ["f_SiIIa_SiIII", "f_SiIIb_SiIII"]:
                    _znodes = np.array([zmin, zmax])

                if prop in ["HCD_damp1", "HCD_damp4"]:
                    _znodes = np.linspace(zmin, zmax, 4)

                self.fid_cont[prop + "_znodes"] = _znodes
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

            for ii in range(4):
                if self.fid_cont["n_HCD_damp" + str(ii + 1)] == 0:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -11.5]
                else:
                    self.fid_cont["HCD_damp" + str(ii + 1)] = [0, -(ii + 1)]

        else:
            self.fid_igm["tau_eff_ztype"] = "interp_spl"
            self.fid_igm["sigT_kms_ztype"] = "interp_spl"
            self.fid_igm["gamma_ztype"] = "pivot"
            self.fid_igm["kF_kms_ztype"] = "pivot"
            self.fid_igm["tau_eff_znodes"] = np.arange(2.2, 4.4, 0.2)
            self.fid_igm["sigT_kms_znodes"] = np.linspace(2.2, 4.2, 6)
            # self.fid_igm["gamma_znodes"] = np.linspace(2.2, 4.2, 4)
            # self.fid_igm["kF_kms_znodes"] = np.linspace(2.2, 4.2, 4)
            if self.fid_igm["tau_eff_ztype"].startswith("interp"):
                self.fid_igm["n_tau_eff"] = len(self.fid_igm["tau_eff_znodes"])
            else:
                self.fid_igm["n_tau_eff"] = 1

            if self.fid_igm["sigT_kms_ztype"].startswith("interp"):
                self.fid_igm["n_sigT"] = len(self.fid_igm["sigT_kms_znodes"])
            else:
                self.fid_igm["n_sigT"] = 1

            if self.fid_igm["gamma_ztype"].startswith("interp"):
                self.fid_igm["n_gamma"] = len(self.fid_igm["gamma_znodes"])
            else:
                self.fid_igm["n_gamma"] = 1

            if self.fid_igm["kF_kms_ztype"].startswith("interp"):
                self.fid_igm["n_kF_kms"] = len(self.fid_igm["kF_kms_znodes"])
            else:
                self.fid_igm["n_kF_kms"] = 1
            self.fid_igm["n_gamma"] = 0
            self.fid_igm["n_kF_kms"] = 0

            self.fid_syst["res_model_type"] = "pivot"
            self.fid_syst["n_res"] = 0
            self.fid_syst["R_coeff"] = [0, 0]

            # model like lya-metal
            self.fid_cont["n_f_Lya_SiIII"] = 0
            self.fid_cont["n_s_Lya_SiIII"] = 0

            self.fid_cont["n_f_Lya_SiII"] = 0
            self.fid_cont["n_s_Lya_SiII"] = 0

            self.fid_cont["n_f_SiIIa_SiIIb"] = 0
            self.fid_cont["n_s_SiIIa_SiIIb"] = 0

            self.fid_cont["n_f_SiIIa_SiIII"] = 0
            self.fid_cont["n_f_SiIIb_SiIII"] = 0

            self.fid_cont["n_f_CIVa_CIVb"] = 0
            self.fid_cont["n_s_CIVa_CIVb"] = 0

            self.fid_cont["n_f_MgIIa_MgIIb"] = 0
            self.fid_cont["n_s_MgIIa_MgIIb"] = 0

            self.fid_cont["n_d_dla1"] = 0
            self.fid_cont["n_d_dla2"] = 0
            self.fid_cont["n_d_dla3"] = 0
            self.fid_cont["n_d_dla4"] = 0
            self.fid_cont["n_c_dla"] = 0

        fid_vals_conts = {
            "f_Lya_SiIII": -4.25,
            "s_Lya_SiIII": 4.75,
            "f_Lya_SiII": -4.5,
            "s_Lya_SiII": 4.75,
            "f_SiIIa_SiIIb": -0.5,
            "s_SiIIa_SiIIb": 4.75,
            "f_SiIIa_SiIII": 1,
            "f_SiIIb_SiIII": 1,
            "HCD_const": 0,
            "HCD_damp1": -1,
            "HCD_damp2": -1.5,
            "HCD_damp3": -2,
            "HCD_damp4": -2.5,
        }
        add_lines = [
            "SiIIa_SiIIb",
            "CIVa_CIVb",
            "MgIIa_MgIIb",
        ]

        for key in self.cont_params.keys():
            if self.fid_cont["n_" + key] == 0:
                self.fid_cont[key] = self.cont_params[key]
            else:
                self.fid_cont[key] = [
                    0,
                    fid_vals_conts[key],
                ]

        self.fid_cont["flat_priors"] = {}

        self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
            [-1, 1],
            [-5, -2],
        ]
        self.fid_cont["flat_priors"]["s_Lya_SiIII"] = [
            [-1, 1],
            [2.5, 7],
        ]

        self.fid_cont["flat_priors"]["f_Lya_SiII"] = [
            [-1, 1],
            [-5, -2.5],
        ]
        self.fid_cont["flat_priors"]["s_Lya_SiII"] = [
            [-1, 1],
            [2.5, 7],
        ]

        self.fid_cont["flat_priors"]["f_SiIIa_SiIIb"] = [
            [-1, 4],
            [-1, 3],
        ]
        self.fid_cont["flat_priors"]["s_SiIIa_SiIIb"] = [
            [-1, 3],
            [2, 6],
        ]

        self.fid_cont["flat_priors"]["f_SiIIa_SiIII"] = [
            [-1, 2],
            [0, 5],
        ]
        self.fid_cont["flat_priors"]["f_SiIIb_SiIII"] = [
            [-1, 1],
            [0, 5],
        ]

        # priors
        # -0.03, 75% of all fluctuations
        self.fid_cont["flat_priors"]["HCD_damp"] = [[-0.5, 0.5], [-12.0, -0.15]]
        self.fid_cont["flat_priors"]["HCD_const"] = [[-1, 1], [-0.2, 1e-6]]


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

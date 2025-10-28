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
    rebin_k: int = 8
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
    cont_params: dict = field(
        default_factory=lambda: {
            "f_Lya_SiIII": [0, -20.0],
            "s_Lya_SiIII": [0, 2.1],
            "f_Lya_SiII": [0, -20.0],
            "s_Lya_SiII": [0, 2.1],
            "f_SiIIa_SiIIb": [0, -20.0],
            "s_SiIIa_SiIIb": [0, 0.1],
            "f_SiIIa_SiIII": [0, 0.0],  # these are variations from exp(0)=1
            "f_SiIIb_SiIII": [0, 0.0],  # these are variations from exp(0)=1
        }
    )
    syst_params: dict = field(
        default_factory=lambda: {
            "R_coeff": [0, 0],
        }
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
    true_cont: dict = field(
        default_factory=lambda: {
            "hcd_model_type": "new_rogers",
            "metal_model_type": "McDonald",
        }
    )
    fid_cont: dict = field(
        default_factory=lambda: {
            "hcd_model_type": "new_rogers",
            "metal_model_type": "McDonald",
        }
    )
    true_syst: dict = field(
        default_factory=lambda: {
            "R_coeff": [0, 0],
            "n_R_coeff": 0,
            "R_coeff_ztype": "pivot",
            "R_coeff_otype": "const",
        }
    )
    fid_syst: dict = field(
        default_factory=lambda: {
            "R_coeff": [0, 0],
            "n_R_coeff": 0,
            "R_coeff_ztype": "pivot",
            "R_coeff_otype": "const",
        }
    )
    apply_smoothing: bool = False
    cov_label: str = "Chabanier2019"
    cov_label_hires: str = "Karacayli2022"
    nyx_training_set: str = "models_Nyx_Mar2025_with_CGAN_val_3axes"
    # nyx_training_set: str = "models_Nyx_Sept2025_include_Nyx_fid_rseed"
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
    emu_cov_type: str = "diagonal"
    file_ic: str | None = None
    mcmc: dict = field(
        default_factory=lambda: {
            "explore": False,
            "parallel": True,
            "n_burn_in": 0,
            "n_steps": 1,
            "n_walkers": 1,
            "thin": 1,
        }
    )
    out_folder: str | None = "."
    Gauss_priors: dict | None = None
    path_data: str = "cup1d"

    def __post_init__(self, val_null=-20):
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
                self.cont_params["HCD_damp" + str(jj)] = [0, val_null]
            self.cont_params["HCD_const"] = [0, 0]
        ##
        for key in self.cont_params.keys():
            self.true_cont[key] = self.cont_params[key]
            self.true_cont["n_" + key] = 0
            self.true_cont[key + "_ztype"] = "pivot"
            if key == "HCD_const":
                self.true_cont[key + "_otype"] = "const"
            else:
                self.true_cont[key + "_otype"] = "exp"

        if self.path_data == "jjchaves":
            self.path_data = os.path.join(
                os.sep, "pscratch", "sd", "j", "jjchaves"
            )
        elif self.path_data == "cup1d":
            self.path_data = os.path.dirname(get_path_repo("cup1d"))

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

    def set_params_zero(self):
        # IGM parameters

        props_igm = {
            "tau_eff": "exp",
            "sigT_kms": "const",
            "gamma": "const",
            "kF_kms": "const",
        }

        # self.fid_cont["z_max"] = {}
        for prop in props_igm:
            self.fid_igm["n_" + prop] = 0
            self.fid_igm[prop + "_ztype"] = "pivot"
            self.fid_igm[prop + "_otype"] = props_igm[prop]
            self.fid_igm[prop + "_fixed"] = True
            # self.fid_igm["z_max"][prop] = 5

        # Contaminants
        props_cont = {
            "f_Lya_SiIII": "exp",
            "s_Lya_SiIII": "exp",
            "f_Lya_SiII": "exp",
            "s_Lya_SiII": "exp",
            "f_SiIIa_SiIIb": "exp",
            "s_SiIIa_SiIIb": "exp",
            "f_SiIIa_SiIII": "exp",
            "f_SiIIb_SiIII": "exp",
            "HCD_damp1": "exp",
            "HCD_damp2": "exp",
            "HCD_damp3": "exp",
            "HCD_damp4": "exp",
            "HCD_const": "const",
        }
        for prop in props_cont:
            self.fid_cont["n_" + prop] = 0
            self.fid_cont[prop + "_ztype"] = "pivot"
            self.fid_cont[prop + "_otype"] = props_cont[prop]
            self.fid_cont[prop + "_fixed"] = True
            # self.fid_cont["z_max"][prop] = 5

        # Systematics
        self.fid_syst["n_R_coeff"] = 0
        self.fid_syst["R_coeff_ztype"] = "pivot"
        self.fid_syst["R_coeff_otype"] = "const"
        self.fid_syst["R_coeff_fixed"] = True
        # self.fid_syst["z_max"] = {}

    # def set_ic_file(self, name_variation):
    #     self.ic_from_file = None
    #     if self.P1D_type.startswith("DESIY1"):

    #         path_out_challenge = os.path.join(
    #             os.path.dirname(get_path_repo("cup1d")),
    #             "data",
    #             "out_DESI_DR1",
    #         )

    #         self.ic_from_file = os.path.join(
    #             path_out_challenge,
    #             "ic",
    #             "allz_snr3_nocosmo_" + fit_type,
    #             "res.npy",
    #         )

    #     if (name_variation is not None) and (name_variation.startswith("sim_")):
    #         self.ic_from_file = None

    def set_fiducial(self, name_variation=None, fit_type=None, val_null=-20):
        null_vals_params = {
            "tau_eff": 0,
            "sigT_kms": 1,
            "gamma": 1,
            "kF_kms": 1,
            "f_Lya_SiIII": val_null,
            "s_Lya_SiIII": 2.1,
            "f_Lya_SiII": val_null,
            "s_Lya_SiII": 2.1,
            "f_SiIIa_SiIIb": val_null,
            "s_SiIIa_SiIIb": 0.1,
            "f_SiIIa_SiIII": 0,
            "f_SiIIb_SiIII": 0,
            "HCD_damp1": val_null,
            "HCD_damp2": val_null,
            "HCD_damp3": val_null,
            "HCD_damp4": val_null,
            "HCD_const": 0,
            "R_coeff": 0,
        }
        fid_vals_igm = {
            "tau_eff": 0,
            "sigT_kms": 1,
            "gamma": 1,
            "kF_kms": 1,
        }

        fid_vals_conts = {
            "f_Lya_SiIII": -4.0,
            "s_Lya_SiIII": 5.0,
            "f_Lya_SiII": -4.0,
            "s_Lya_SiII": 5.5,
            "f_SiIIa_SiIIb": 0.5,
            "s_SiIIa_SiIIb": 4.0,
            "f_SiIIa_SiIII": 1,
            "f_SiIIb_SiIII": 1,
            "HCD_damp1": -1.4,
            "HCD_damp2": -6.0,
            "HCD_damp3": -5.0,
            "HCD_damp4": -5.0,
            "HCD_const": 0.0,
        }
        fid_vals_syst = {
            "R_coeff": 0.0,
        }

        if (fit_type is not None) and (fit_type == "global_igm"):
            fid_vals_igm = null_vals_params
            fid_vals_conts = null_vals_params
            fid_vals_syst = null_vals_params

        for key in self.igm_params:
            if self.fid_igm["n_" + key] == 0:
                self.fid_igm[key + "_fixed"] = True
                use_val = null_vals_params[key]
            else:
                self.fid_igm[key + "_fixed"] = False
                use_val = fid_vals_igm[key]

            if self.fid_igm[key + "_ztype"] == "pivot":
                self.fid_igm[key] = [0, use_val]
            else:
                self.fid_igm[key] = (
                    np.zeros(len(self.fid_igm[key + "_znodes"])) + use_val
                )

        for key in self.cont_params.keys():
            if self.fid_cont["n_" + key] == 0:
                self.fid_cont[key + "_fixed"] = True
                use_val = null_vals_params[key]
            else:
                self.fid_cont[key + "_fixed"] = False
                use_val = fid_vals_conts[key]

            if self.fid_cont[key + "_ztype"] == "pivot":
                self.fid_cont[key] = [0, use_val]
            else:
                self.fid_cont[key] = (
                    np.zeros(len(self.fid_cont[key + "_znodes"])) + use_val
                )

        for key in self.syst_params.keys():
            if self.fid_syst["n_" + key] == 0:
                self.fid_syst[key + "_fixed"] = True
                use_val = null_vals_params[key]
            else:
                self.fid_syst[key + "_fixed"] = False
                use_val = fid_vals_syst[key]

            if self.fid_syst[key + "_ztype"] == "pivot":
                self.fid_syst[key] = [0, use_val]
            else:
                self.fid_syst[key] = (
                    np.zeros(len(self.fid_syst[key + "_znodes"])) + use_val
                )

        self.fid_cont["flat_priors"] = {}

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
                [-1, 1],
                [val_null - 0.5, -2],
            ]
        elif (name_variation is not None) and (name_variation == "Ma2025"):
            self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
                [-1, 1],
                [-6, 2],
            ]
        else:
            self.fid_cont["flat_priors"]["f_Lya_SiIII"] = [
                [-1, 1],
                [-6, -2],
            ]

        if (name_variation is not None) and (name_variation == "Ma2025"):
            self.fid_cont["flat_priors"]["s_Lya_SiIII"] = [
                [-1, 1],
                [-20, 6],
            ]
        else:
            self.fid_cont["flat_priors"]["s_Lya_SiIII"] = [
                [-1, 1],
                [2, 7],
            ]

        if (name_variation is not None) and (name_variation.startswith("sim_")):
            self.fid_cont["flat_priors"]["f_Lya_SiII"] = [
                [-1, 1],
                [val_null - 0.5, -2],
            ]
        elif (name_variation is not None) and (name_variation == "Ma2025"):
            self.fid_cont["flat_priors"]["f_Lya_SiII"] = [
                [-1, 1],
                [-6, 2],
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
                [val_null - 0.5, 3],
            ]
        else:
            self.fid_cont["flat_priors"]["f_SiIIa_SiIIb"] = [
                [-1, 4],
                [-3, 3],
            ]

        self.fid_cont["flat_priors"]["s_SiIIa_SiIIb"] = [
            [-1, 3],
            [0, 7.5],
        ]

        self.fid_cont["flat_priors"]["f_SiIIa_SiIII"] = [
            [-1, 2],
            [-1, 3],
        ]
        self.fid_cont["flat_priors"]["f_SiIIb_SiIII"] = [
            [-1, 1],
            [-1, 3],
        ]

        # priors
        # -0.03, 75% of all fluctuations
        # self.fid_cont["flat_priors"]["HCD_damp1"] = [[-0.5, 0.5], [-10.0, -1.0]]
        if (name_variation is not None) and (name_variation.startswith("sim_")):
            min_hcd = val_null - 0.5
        else:
            min_hcd = -10.0
        self.fid_cont["flat_priors"]["HCD_damp1"] = [
            [-0.5, 0.5],
            [min_hcd, -0.03],
        ]
        self.fid_cont["flat_priors"]["HCD_damp2"] = [
            [-0.5, 0.5],
            [min_hcd, -1.0],
        ]
        self.fid_cont["flat_priors"]["HCD_damp3"] = [
            [-0.5, 0.5],
            [min_hcd, -1.0],
        ]
        self.fid_cont["flat_priors"]["HCD_damp4"] = [
            [-0.5, 0.5],
            [min_hcd, -1.0],
        ]
        self.fid_cont["flat_priors"]["HCD_const"] = [[-1, 1], [-0.2, 0.2]]

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

    def set_out_folder(self, name_variation):
        if name_variation is None:
            self.out_folder = os.path.join(
                self.path_data,
                "data",
                "out_DESI_DR1",
                self.P1D_type,
                self.fit_type,
                self.emulator_label,
            )
        else:
            self.out_folder = os.path.join(
                self.path_data,
                "data",
                "out_DESI_DR1",
                self.P1D_type,
                name_variation,
                self.emulator_label,
            )

    def set_baseline(
        self,
        z_min=2.2,
        z_max=4.2,
        fit_type="at_a_time",
        fix_cosmo=True,
        P1D_type="DESIY1_QMLE3",
        fid_cosmo_label="Planck18",
        name_variation=None,
        inflate=1.0,
        mcmc_conf="explore",
    ):
        """
        Set baseline parameters
        """

        if fit_type not in [
            "global_all",  # all params from at_a_time_global
            "global_opt",  # for opt all
            "global_igm",  # for opt err
            "at_a_time_global",  # vary same parameters as in global fits
        ]:
            raise ValueError("fit_type " + fit_type + " not implemented")

        ## store input parameters
        self.z_min = z_min
        self.z_max = z_max
        self.fit_type = fit_type
        self.fix_cosmo = fix_cosmo
        self.P1D_type = P1D_type
        self.fid_cosmo_label = fid_cosmo_label
        ##
        self.set_out_folder(name_variation)

        if mcmc_conf == "test":
            self.mcmc["explore"] = True
            self.mcmc["parallel"] = True
            self.mcmc["n_burn_in"] = 0
            self.mcmc["n_steps"] = 5
            self.mcmc["n_walkers"] = 1
            self.mcmc["thin"] = 1
        elif mcmc_conf == "explore":
            self.mcmc["explore"] = True
            self.mcmc["parallel"] = True
            self.mcmc["n_burn_in"] = 1500
            self.mcmc["n_steps"] = 2000
            self.mcmc["n_walkers"] = 10
            self.mcmc["thin"] = 20
        elif mcmc_conf == "full":
            self.mcmc["explore"] = True
            self.mcmc["parallel"] = True
            self.mcmc["n_burn_in"] = 1500
            self.mcmc["n_steps"] = 3000
            self.mcmc["n_walkers"] = 10
            self.mcmc["thin"] = 20

        # reset parameters
        self.set_params_zero()

        ## set cosmology
        if (name_variation is not None) and (name_variation == "zmin"):
            self.z_min = 2.6
        if (name_variation is not None) and (name_variation == "zmax"):
            self.z_max = 3.4
        ##

        ## set cosmology
        if (name_variation is not None) and (name_variation == "cosmo"):
            self.fid_cosmo_label = "DESIDR2_ACT"
        elif (name_variation is not None) and (name_variation == "cosmo_low"):
            self.fid_cosmo_label = "Planck18_low_omh2"
        elif (name_variation is not None) and (name_variation == "cosmo_high"):
            self.fid_cosmo_label = "Planck18_high_omh2"
        ##

        ## set ic correction for lyssa emu
        if (name_variation is not None) and (name_variation == "ic_lace-lyssa"):
            self.ic_correction = True
        else:
            self.ic_correction = False
        ##

        ## set IGM params
        if ("mpg" in self.emulator_label) | ("Mpg" in self.emulator_label):
            sim_fid = "mpg_central"
        elif ("nyx" in self.emulator_label) | ("Nyx" in self.emulator_label):
            sim_fid = "nyx_central"
        else:
            sim_fid = "mpg_central"
        self.fid_igm["label_mF"] = sim_fid
        self.fid_igm["label_T"] = sim_fid
        self.fid_igm["label_kF"] = sim_fid

        if (name_variation is not None) and (name_variation == "Turner24"):
            self.fid_igm["label_mF"] = "Turner24"
        elif (name_variation is not None) and (name_variation == "Gaikwad21"):
            self.fid_igm["label_mF"] = "Gaikwad21"
        elif (name_variation is not None) and (
            (name_variation == "Gaikwad21") | (name_variation == "Gaikwad21T")
        ):
            self.fid_igm["label_T"] = "Gaikwad21"
        if (name_variation is not None) and (
            name_variation.startswith("sim_mpg_central_igm")
        ):
            self.fit_type = "global_igm"
            fit_type = self.fit_type
        ##

        if (name_variation is not None) and (name_variation == "Ma2025"):
            self.fid_cont["metal_model_type"] = "SiVid"
        else:
            self.fid_cont["metal_model_type"] = "McDonald"

        ## inflate errors
        # we multiply cov by inflate square
        if "DESIY1" in P1D_type:
            if (name_variation is not None) and (
                "no_inflate" in name_variation
            ):
                inflate = 1.0

            self.cov_factor = {
                "z": np.arange(self.z_min, self.z_max + 1e-3, 0.2),
            }
            self.cov_factor["val"] = (
                np.ones(len(self.cov_factor["z"])) * inflate
            )
            ##

        ## modify emulator error
        if (name_variation is not None) and ("no_emu_cov" in name_variation):
            self.emu_cov_factor = 1e-20
        else:
            self.emu_cov_factor = 1

        props_igm = ["tau_eff", "sigT_kms", "gamma", "kF_kms"]
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
            "HCD_damp2",
            "HCD_damp3",
            "HCD_damp4",
            "HCD_const",
        ]

        # z at a time
        #############
        if fit_type == "at_a_time_global":
            # for prop in props_cont:
            #     self.fid_cont["z_max"][prop] = 5

            self.fid_syst["R_coeff_znodes"] = np.array([3.0])
            self.fid_syst["n_R_coeff"] = 1
            self.fid_syst["R_coeff_ztype"] = "pivot"

            if name_variation is None:
                baseline_prop = [
                    "tau_eff",
                    "sigT_kms",
                    "gamma",
                    "kF_kms",
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            elif name_variation == "Ma2025":
                baseline_prop = [
                    "tau_eff",
                    "sigT_kms",
                    "gamma",
                    "kF_kms",
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    # "s_Lya_SiII",
                    # "f_SiIIa_SiIIb",
                    # "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            else:
                raise ValueError(
                    "name_variation not implemented", name_variation
                )

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

        # global all params
        #############
        elif fit_type == "global_all":
            ## set IC
            if "mpg" in self.emulator_label:
                fname = "mpg_ic_at_a_time.npy"
            else:
                fname = "nyx_ic_at_a_time.npy"
            self.file_ic = os.path.join(self.path_data, "data", "ics", fname)

            # for prop in props_cont:
            #     self.fid_cont["z_max"][prop] = 5

            baseline_prop = [
                "tau_eff",
                "sigT_kms",
                "gamma",
                "kF_kms",
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                # "f_SiIIa_SiIII",
                # "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp2",
                "HCD_damp3",
                "HCD_damp4",
            ]

            nodes = np.arange(self.z_min, self.z_max + 1e-3, 0.2)

            for prop in props_igm:
                if prop in baseline_prop:
                    self.fid_igm["n_" + prop] = 11
                # elif prop == "kF_kms":
                #     self.fid_igm["n_" + prop] = 0
                else:
                    self.fid_igm["n_" + prop] = 0

                if self.fid_igm["n_" + prop] == 0:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_znodes"] = nodes
                    self.fid_igm[prop + "_ztype"] = "interp_lin"

            for prop in props_cont:
                if prop in baseline_prop:
                    self.fid_cont["n_" + prop] = 11
                else:
                    self.fid_cont["n_" + prop] = 0

                if self.fid_cont["n_" + prop] == 0:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                else:
                    self.fid_cont[prop + "_znodes"] = nodes
                    self.fid_cont[prop + "_ztype"] = "interp_lin"

            self.fid_syst["R_coeff_znodes"] = nodes
            self.fid_syst["n_R_coeff"] = 11
            self.fid_syst["R_coeff_ztype"] = "interp_lin"

        #############
        elif fit_type == "global_opt":
            ## set IC
            if "mpg" in self.emulator_label:
                fname = "mpg_ic_global_red.npy"
            else:
                fname = "nyx_ic_global_red.npy"
            self.file_ic = os.path.join(self.path_data, "data", "ics", fname)

            if (name_variation is not None) and (
                name_variation.startswith("sim_")
            ):
                self.file_ic = None
            ##

            # for prop in props_cont:
            #     self.fid_cont["z_max"][prop] = 5

            if name_variation == "metal_trad":
                baseline_prop = [
                    "f_Lya_SiIII",
                    # "s_Lya_SiIII",
                    "f_Lya_SiII",
                    # "s_Lya_SiII",
                    # "f_SiIIa_SiIIb",
                    # "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            elif name_variation == "metal_si2":
                baseline_prop = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    # "f_SiIIa_SiIIb",
                    # "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            elif name_variation == "metal_deco":
                baseline_prop = [
                    "f_Lya_SiIII",
                    # "s_Lya_SiIII",
                    "f_Lya_SiII",
                    # "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            elif name_variation == "metal_thin":
                baseline_prop = [
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
            elif name_variation == "Ma2025":
                baseline_prop = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    # "s_Lya_SiII",
                    # "f_SiIIa_SiIIb",
                    # "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            elif name_variation == "DLAs":
                baseline_prop = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    # "HCD_damp1",
                    # "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]
            elif name_variation == "HCD0":
                baseline_prop = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                    "HCD_const",
                ]
            else:
                baseline_prop = [
                    "f_Lya_SiIII",
                    "s_Lya_SiIII",
                    "f_Lya_SiII",
                    "s_Lya_SiII",
                    "f_SiIIa_SiIIb",
                    "s_SiIIa_SiIIb",
                    # "f_SiIIa_SiIII",
                    # "f_SiIIb_SiIII",
                    "HCD_damp1",
                    "HCD_damp2",
                    "HCD_damp3",
                    "HCD_damp4",
                ]

            zvar = [
                "f_Lya_SiIII",
                "s_Lya_SiIII",
                "f_Lya_SiII",
                "s_Lya_SiII",
                "f_SiIIa_SiIIb",
                "s_SiIIa_SiIIb",
                # "f_SiIIa_SiIII",
                # "f_SiIIb_SiIII",
                "HCD_damp1",
                "HCD_damp2",
                "HCD_damp3",
                "HCD_damp4",
            ]

            ## set IGM
            if name_variation == "more_igm":
                nz_igm = 6
            elif name_variation == "less_igm":
                nz_igm = 2
            else:
                nz_igm = 4

            if (name_variation is not None) and (name_variation == "Gaikwad21"):
                self.fid_igm["tau_eff_znodes"] = []
            elif (name_variation is not None) and (
                name_variation == "Turner24"
            ):
                self.fid_igm["tau_eff_znodes"] = np.array([3.0])
            else:
                self.fid_igm["tau_eff_znodes"] = np.geomspace(
                    self.z_min, self.z_max, nz_igm
                )
                # self.fid_igm["tau_eff_znodes"] = np.linspace(
                #     self.z_min, self.z_max, nz_igm
                # )

            if (name_variation is not None) and (
                (name_variation == "Gaikwad21")
                | (name_variation == "Gaikwad21T")
            ):
                self.fid_igm["sigT_kms_znodes"] = []
                self.fid_igm["gamma_znodes"] = []
            else:
                self.fid_igm["sigT_kms_znodes"] = np.geomspace(
                    self.z_min, self.z_max, nz_igm
                )
                self.fid_igm["gamma_znodes"] = np.geomspace(
                    self.z_min, self.z_max, nz_igm
                )

            self.fid_igm["kF_kms_znodes"] = np.geomspace(
                self.z_min, self.z_max, nz_igm
            )
            # self.fid_igm["sigT_kms_znodes"] = np.linspace(
            #     self.z_min, self.z_max, nz_igm
            # )
            # self.fid_igm["gamma_znodes"] = np.linspace(
            #     self.z_min, self.z_max, nz_igm
            # )
            # self.fid_igm["kF_kms_znodes"] = np.linspace(
            #     self.z_min, self.z_max, nz_igm
            # )

            # if name_variation == "kF_kms":
            #     self.fid_igm["kF_kms_znodes"] = []

            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_lin"
            ##

            ## set contaminants
            nodes = np.geomspace(self.z_min, self.z_max, 2)

            for prop in props_cont:
                if prop not in baseline_prop:
                    self.fid_cont["n_" + prop] = 0
                    continue

                if prop in zvar:
                    self.fid_cont[prop + "_znodes"] = nodes
                else:
                    self.fid_cont[prop + "_znodes"] = np.array([3.0])

                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_lin"
            ##

            ## set systematic
            if name_variation == "no_res":
                self.fid_syst["R_coeff_znodes"] = []
            else:
                self.fid_syst["R_coeff_znodes"] = np.arange(
                    2.2, 4.2 + 1e-5, 0.2
                )

            if name_variation == "zmin":
                self.fid_syst["R_coeff_znodes"] = self.fid_syst[
                    "R_coeff_znodes"
                ][self.fid_syst["R_coeff_znodes"] >= self.z_min]
            elif name_variation == "zmax":
                self.fid_syst["R_coeff_znodes"] = self.fid_syst[
                    "R_coeff_znodes"
                ][self.fid_syst["R_coeff_znodes"] <= self.z_max]
            self.fid_syst["n_R_coeff"] = len(self.fid_syst["R_coeff_znodes"])
            self.fid_syst["R_coeff_ztype"] = "interp_lin"

        #############

        elif fit_type == "global_igm":
            self.file_ic = None
            ##

            baseline_prop = [
                # "f_Lya_SiIII",
                # "s_Lya_SiIII",
                # "f_Lya_SiII",
                # "s_Lya_SiII",
                # "f_SiIIa_SiIIb",
                # "s_SiIIa_SiIIb",
                # "f_SiIIa_SiIII",
                # "f_SiIIb_SiIII",
                # "HCD_damp1",
                # "HCD_damp2",
                # "HCD_damp3",
                # "HCD_damp4",
            ]
            zvar = [
                # "f_Lya_SiIII",
                # "s_Lya_SiIII",
                # "f_Lya_SiII",
                # "s_Lya_SiII",
                # "f_SiIIa_SiIIb",
                # "s_SiIIa_SiIIb",
                # "f_SiIIa_SiIII",
                # "f_SiIIb_SiIII",
                # "HCD_damp1",
                # "HCD_damp2",
                # "HCD_damp3",
                # "HCD_damp4",
            ]

            nz_igm = 6
            if name_variation == "sim_mpg_central_igm0":
                nz_igm = 0

            self.fid_igm["tau_eff_znodes"] = np.geomspace(
                self.z_min, self.z_max, nz_igm
            )
            self.fid_igm["sigT_kms_znodes"] = np.geomspace(
                self.z_min, self.z_max, nz_igm
            )
            self.fid_igm["gamma_znodes"] = np.geomspace(
                self.z_min, self.z_max, nz_igm
            )
            self.fid_igm["kF_kms_znodes"] = []

            for prop in props_igm:
                self.fid_igm["n_" + prop] = len(self.fid_igm[prop + "_znodes"])
                if self.fid_igm["n_" + prop] <= 1:
                    self.fid_igm[prop + "_ztype"] = "pivot"
                else:
                    self.fid_igm[prop + "_ztype"] = "interp_lin"
            ##

            ## set contaminants
            nodes = np.geomspace(self.z_min, self.z_max, 2)

            for prop in props_cont:
                if prop not in baseline_prop:
                    self.fid_cont["n_" + prop] = 0
                    continue

                if prop in zvar:
                    self.fid_cont[prop + "_znodes"] = nodes
                else:
                    self.fid_cont[prop + "_znodes"] = np.array([3.0])

                self.fid_cont["n_" + prop] = len(
                    self.fid_cont[prop + "_znodes"]
                )
                if self.fid_cont["n_" + prop] <= 1:
                    self.fid_cont[prop + "_ztype"] = "pivot"
                else:
                    self.fid_cont[prop + "_ztype"] = "interp_lin"
            ##

            ## set systematic
            self.fid_syst["R_coeff_znodes"] = []
            # self.fid_syst["R_coeff_znodes"] = np.arange(
            #     2.2, 4.2 + 1e-5, 0.2
            # )
            self.fid_syst["n_R_coeff"] = len(self.fid_syst["R_coeff_znodes"])
            self.fid_syst["R_coeff_ztype"] = "interp_lin"

        #############

        else:
            raise ValueError("Fit type not recognized")

        self.set_fiducial(name_variation, fit_type=fit_type)


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

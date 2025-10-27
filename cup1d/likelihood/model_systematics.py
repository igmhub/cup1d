import numpy as np

from cup1d.nuisance import resolution_class


class Systematics(object):
    """Contains all IGM models"""

    def __init__(
        self, free_param_names=None, resolution_model=None, pars_syst=None
    ):
        self.pars_syst = pars_syst

        if "flat_priors" in pars_syst:
            flat_priors = pars_syst["flat_priors"]
        else:
            flat_priors = None
        if "Gauss_priors" in pars_syst:
            Gauss_priors = pars_syst["Gauss_priors"]
        else:
            Gauss_priors = None

        if "z_max" in pars_syst:
            z_max = pars_syst["z_max"]
        else:
            z_max = None

        prop_coeffs = {}
        fid_vals = {}
        for key in pars_syst:
            fid_vals[key] = pars_syst[key]
            for key2 in ["otype", "ztype", "znodes"]:
                key3 = key + "_" + key2
                if key3 in pars_syst:
                    prop_coeffs[key3] = pars_syst[key3]
                else:
                    if key3.endswith("otype"):
                        if key3.startswith("HCD_const"):
                            prop_coeffs[key3] = "const"
                        else:
                            prop_coeffs[key3] = "exp"
                    elif key3.endswith("ztype"):
                        prop_coeffs[key3] = "pivot"

        # setup Resolution model
        if resolution_model:
            self.resolution_model = resolution_model
        else:
            self.resolution_model = resolution_class.Resolution(
                free_param_names=free_param_names,
                fid_vals=fid_vals,
                prop_coeffs=prop_coeffs,
                flat_priors=flat_priors,
                Gauss_priors=Gauss_priors,
            )

    # def get_dict_cont(self):
    #     dict_out = {}

    #     if self.args.fid_syst["res_model_type"] == "pivot":
    #         for ii in range(len(self.args.fid_syst["R_coeff"])):
    #             dict_out["R_coeff_" + str(ii)] = self.args.fid_syst["R_coeff"][
    #                 -1 - ii
    #             ]
    #     else:
    #         for ii in range(self.resolution_model.get_Nparam()):
    #             dict_out["R_coeff_" + str(ii)] = 0

    #     return dict_out

    def get_contamination(self, z, k_kms, like_params=[]):
        # include multiplicative resolution correction
        cont = self.resolution_model.get_contamination(
            z=z, k_kms=k_kms, like_params=like_params
        )

        if len(z) == 1:
            cont_resolution = np.ones_like(k_kms) * cont
        else:
            cont_resolution = []
            for iz in range(len(z)):
                if type(cont) != int:
                    cont_resolution.append(np.ones_like(k_kms[iz]) * cont[iz])
                else:
                    cont_resolution.append(np.ones_like(k_kms[iz]) * cont)

        return cont_resolution

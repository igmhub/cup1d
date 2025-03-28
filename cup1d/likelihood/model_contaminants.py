import numpy as np

from cup1d.nuisance import (
    metal_model,
    hcd_model_McDonald2005,
    hcd_model_Rogers2017,
    hcd_model_new,
    SN_model,
    AGN_model,
)


class Contaminants(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        SiII_model=None,
        SiIII_model=None,
        hcd_model=None,
        sn_model=None,
        agn_model=None,
        hcd_model_type="Rogers2017",
        ic_correction=False,
        fid_SiIII_X=[0, -10],
        fid_SiIII_D=[0, 2],
        fid_SiIII_A=[0, 1.5],
        fid_SiII_X=[0, -10],
        fid_SiII_D=[0, 2],
        fid_SiII_A=[0, 1.5],
        fid_A_damp=[0, -9],
        fid_A_scale=[0, 1],
        fid_SN=[0, -4],
        fid_AGN=[0, -5],
    ):
        self.fid_SiIII_X = fid_SiIII_X
        self.fid_SiIII_D = fid_SiIII_D
        self.fid_SiIII_A = fid_SiIII_A
        self.fid_SiII_X = fid_SiII_X
        self.fid_SiII_D = fid_SiII_D
        self.fid_SiII_A = fid_SiII_A
        self.fid_A_damp = fid_A_damp
        self.fid_A_scale = fid_A_scale
        self.fid_SN = fid_SN
        self.fid_AGN = fid_AGN
        self.ic_correction = ic_correction

        # setup metal models
        self.metal_models = []
        if SiIII_model:
            self.SiIII_model = SiIII_model
        else:
            self.SiIII_model = metal_model.MetalModel(
                metal_label="SiIII",
                free_param_names=free_param_names,
                X_fid_value=self.fid_SiIII_X,
                D_fid_value=self.fid_SiIII_D,
                A_fid_value=self.fid_SiIII_A,
            )
        self.metal_models.append(self.SiIII_model)

        if SiII_model:
            self.SiII_model = SiII_model
        else:
            self.SiII_model = metal_model.MetalModel(
                metal_label="SiII",
                free_param_names=free_param_names,
                X_fid_value=self.fid_SiII_X,
                D_fid_value=self.fid_SiII_D,
                A_fid_value=self.fid_SiII_A,
            )
        self.metal_models.append(self.SiII_model)

        # setup HCD model
        if hcd_model:
            self.hcd_model = hcd_model
        else:
            if hcd_model_type == "Rogers2017":
                self.hcd_model = hcd_model_Rogers2017.HCD_Model_Rogers2017(
                    free_param_names=free_param_names,
                    fid_A_damp=self.fid_A_damp,
                    fid_A_scale=self.fid_A_scale,
                )
            elif hcd_model_type == "McDonald2005":
                self.hcd_model = hcd_model_McDonald2005.HCD_Model_McDonald2005(
                    free_param_names=free_param_names,
                    fid_A_damp=self.fid_A_damp,
                )
            elif hcd_model_type == "new":
                self.hcd_model = hcd_model_new.HCD_Model_new(
                    free_param_names=free_param_names,
                    fid_A_damp=self.fid_A_damp,
                    fid_A_scale=self.fid_A_scale,
                )
            else:
                raise ValueError(
                    "hcd_model_type must be one of 'Rogers2017', 'McDonald2005', or 'new'"
                )

        # setup SN model
        if sn_model:
            self.sn_model = sn_model
        else:
            self.sn_model = SN_model.SN_Model(
                free_param_names=free_param_names,
                fid_value=self.fid_SN,
            )

        # setup AGN model
        if agn_model:
            self.agn_model = sn_model
        else:
            self.agn_model = AGN_model.AGN_Model(
                free_param_names=free_param_names,
                fid_value=self.fid_AGN,
            )

    def get_dict_cont(self):
        dict_out = {}

        for ii in range(2):
            dict_out["ln_x_SiIII_" + str(ii)] = self.fid_SiIII_X[-1 - ii]
            dict_out["ln_d_SiIII_" + str(ii)] = self.fid_SiIII_D[-1 - ii]
            dict_out["a_SiIII_" + str(ii)] = self.fid_SiIII_A[-1 - ii]
            dict_out["ln_x_SiII_" + str(ii)] = self.fid_SiII_X[-1 - ii]
            dict_out["ln_d_SiII_" + str(ii)] = self.fid_SiII_D[-1 - ii]
            dict_out["a_SiII_" + str(ii)] = self.fid_SiII_A[-1 - ii]
            dict_out["ln_A_damp_" + str(ii)] = self.fid_A_damp[-1 - ii]
            dict_out["ln_A_scale_" + str(ii)] = self.fid_A_scale[-1 - ii]
            dict_out["ln_SN_" + str(ii)] = self.fid_SN[-1 - ii]
            dict_out["ln_AGN_" + str(ii)] = self.fid_AGN[-1 - ii]
        dict_out["ic_correction"] = self.ic_correction

        return dict_out

    def get_contamination(self, z, k_kms, mF, M_of_z, like_params=[]):
        # include multiplicative metal contamination
        cont_metals = 1
        for X_model in self.metal_models:
            cont = X_model.get_contamination(
                z=z,
                k_kms=k_kms,
                mF=mF,
                like_params=like_params,
            )
            cont_metals *= cont

        # include HCD contamination
        cont_HCD = self.hcd_model.get_contamination(
            z=z,
            k_kms=k_kms,
            like_params=like_params,
        )

        # include SN contamination
        cont_SN = self.sn_model.get_contamination(
            z=z,
            k_Mpc=k_kms * M_of_z,
            like_params=like_params,
        )

        # include AGN contamination
        cont_AGN = self.agn_model.get_contamination(
            z=z,
            k_kms=k_kms,
            like_params=like_params,
        )

        if self.ic_correction:
            IC_corr = ref_nyx_ic_correction(k_kms, z)
        else:
            IC_corr = 1

        # print("me", cont_metals)
        # print("hcd", cont_HCD)
        # print("sn", cont_SN)
        # print("agn", cont_AGN)
        # print("ic", IC_corr)

        cont_total = cont_metals * cont_HCD * cont_SN * cont_AGN * IC_corr

        if np.any(cont_AGN < 0):
            return None
        else:
            return cont_total


def ref_nyx_ic_correction(k_kms, z):
    # This is the function fitted from the comparison of two Nyx runs,
    # one with 2lpt (single fluid) IC and the other one with monofonic (2 fluid)
    # - The high k points and z evolution are well determined
    # - Low k term: quite uncertain, due to cosmic variance
    ic_corr_z = np.array([0.15261529, -2.30600644, 2.61877894])
    ic_corr_k = 0.003669741766936781
    ancorIC = (ic_corr_z[0] * z**2 + ic_corr_z[1] * z + ic_corr_z[2]) * (
        1 - np.exp(-k_kms / ic_corr_k)
    )
    corICs = 1 / (1 - ancorIC / 100)
    return corICs


# def nyx_ic_correction(k_kms, z):
#     # This is the function fitted from the comparison of two Nyx runs,
#     # one with 2lpt (single fluid) IC and the other one with monofonic (2 fluid)
#     # - The high k points and z evolution are well determined
#     # - Low k term: quite uncertain, due to cosmic variance
#     coeff0 = np.poly1d(np.array([0.00067032, -0.00626953, 0.0073908]))
#     coeff1 = np.poly1d(np.array([0.00767315, -0.04693207, 0.07151469]))
#     cfit = np.zeros(2)
#     cfit[0] = coeff0(z)
#     cfit[1] = coeff1(z)

#     rfit = np.poly1d(cfit)
#     # multiplicative correction
#     ic_corr = 10 ** rfit(np.log10(k_kms))

#     return ic_corr


# def nuisance_nyx_ic_correction(P0, k, z, Aic, Bic):
#     ic_corr_k = 0.003669741766936781
#     correction = (Aic + Bic * (z - 3)) * (1 - np.exp(-k / ic_corr_k))  # in %
#     return P0 / (1 - 0.01 * correction)


# def prior_nyx_ic_correction():
#     # The central values are the result of a 1st order polynomial fit from the
#     # 2nd order function given in _ref_nyx_ic_correction(P0, k, z)
#     return {"Aic": (-2.9, 1.0), "Bic": (-1.4, 0.5)}

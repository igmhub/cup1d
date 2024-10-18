import numpy as np

from cup1d.nuisance import metal_model, hcd_model_McDonald2005, SN_model


class Contaminants(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        SiII_model=None,
        SiIII_model=None,
        hcd_model=None,
        sn_model=None,
        ic_correction=False,
        fid_SiII=-10,
        fid_SiIII=-10,
        fid_HCD=-6,
        fid_SN=-10,
    ):
        self.fid_SiII = fid_SiII
        self.fid_SiIII = fid_SiIII
        self.fid_HCD = fid_HCD
        self.fid_SN = fid_SN
        self.ic_correction = ic_correction

        # setup metal models
        self.metal_models = []
        if SiIII_model:
            self.SiIII_model = SiIII_model
        else:
            self.SiIII_model = metal_model.MetalModel(
                metal_label="SiIII",
                free_param_names=free_param_names,
                fid_value=self.fid_SiIII,
            )
        self.metal_models.append(self.SiIII_model)

        if SiII_model:
            self.SiII_model = SiII_model
        else:
            self.SiII_model = metal_model.MetalModel(
                metal_label="SiII",
                free_param_names=free_param_names,
                fid_value=self.fid_SiII,
            )
        self.metal_models.append(self.SiII_model)

        # setup HCD model
        if hcd_model:
            self.hcd_model = hcd_model
        else:
            self.hcd_model = hcd_model_McDonald2005.HCD_Model_McDonald2005(
                free_param_names=free_param_names,
                fid_value=self.fid_HCD,
            )

        # setup SN model
        if hcd_model:
            self.sn_model = sn_model
        else:
            self.sn_model = SN_model.SN_Model(
                free_param_names=free_param_names,
                fid_value=self.fid_SN,
            )

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

        if self.ic_correction:
            IC_corr = ref_nyx_ic_correction(k_kms, z)
        else:
            IC_corr = 1

        return cont_metals * cont_HCD * cont_SN * IC_corr


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

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
        fid_SiII=-10,
        fid_SiIII=-10,
        fid_HCD=-6,
        fid_SN=-10,
    ):
        self.fid_SiII = fid_SiII
        self.fid_SiIII = fid_SiIII
        self.fid_HCD = fid_HCD
        self.fid_SN = fid_SN

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

        return cont_metals * cont_HCD * cont_SN

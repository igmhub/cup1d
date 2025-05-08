import numpy as np

from cup1d.nuisance import resolution_model


class Systematics(object):
    """Contains all IGM models"""

    def __init__(
        self,
        free_param_names=None,
        resolution_model=None,
        fid_R_coeff=[0.0, 0.0],
    ):
        self.fid_R_coeff = fid_R_coeff

        # setup Resolution model
        if resolution_model:
            self.resolution_model = resolution_model
        else:
            self.resolution_model = resolution_model.Resolution_Model(
                free_param_names=free_param_names, fid_R_coeff=self.fid_R_coeff
            )

    def get_dict_cont(self):
        dict_out = {}

        for ii in range(2):
            dict_out["R_coeff_" + str(ii)] = self.fid_R_coeff[-1 - ii]

        return dict_out

    def get_contamination(self, z, k_kms, like_params=[]):
        # include multiplicative resolution correction
        cont_resolution = self.resolution_model.get_contamination(
            z=z, k_kms=k_kms, like_params=like_params
        )

        cont_total = cont_resolution

        return cont_total

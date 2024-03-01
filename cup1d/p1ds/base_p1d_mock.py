import numpy as np
import matplotlib.pyplot as plt

import cup1d
from cup1d.p1ds.base_p1d_data import BaseDataP1D
from lace.utils.smoothing_manager import apply_smoothing


class BaseMockP1D(BaseDataP1D):
    """Base class to store mock measurements of the 1D power spectrum"""

    def __init__(
        self,
        z,
        k_kms,
        Pk_kms,
        cov_Pk_kms,
        add_noise=False,
        seed=0,
        z_min=0,
        z_max=10,
    ):
        """Construct base P1D class, from measured power and covariance"""

        if add_noise:
            Pk_perturb_kms = self.get_Pk_iz_perturbed(
                Pk_kms, cov_Pk_kms, seed=seed
            )
        else:
            Pk_perturb_kms = Pk_kms

        super().__init__(
            z, k_kms, Pk_perturb_kms, cov_Pk_kms, z_min=z_min, z_max=z_max
        )

    def get_Pk_iz_perturbed(self, Pk_kms, cov_Pk_kms, nsamples=1, seed=0):
        """Perturb data by adding Gaussian noise according to the covariance matrix"""

        np.random.seed(seed)
        Pk_iz_perturb = []

        for iz in range(len(Pk_kms)):
            _ = np.random.multivariate_normal(
                Pk_kms[iz], cov_Pk_kms[iz], nsamples
            )
            if nsamples == 1:
                Pk_iz_perturb.append(_[0])
            else:
                Pk_iz_perturb.append(_)

        return Pk_iz_perturb

    def set_smoothing_kms(self, emulator, fprint=print):
        """Smooth data in 1/(km/s)"""

        list_data_Mpc = []
        for ii in range(len(self.z)):
            data = {}
            data["k_Mpc"] = self.k_kms * self.dkms_dMpc[ii]
            data["p1d_Mpc"] = self.Pk_kms[ii] * self.dkms_dMpc[ii]
            list_data_Mpc.append(data)

        apply_smoothing(emulator, list_data_Mpc, fprint=fprint)

        for ii in range(len(self.z)):
            self.Pk_kms[ii] = (
                list_data_Mpc[ii]["p1d_Mpc_smooth"] / self.dkms_dMpc[ii]
            )

    def set_smoothing_Mpc(self, emulator, list_data_Mpc, fprint=print):
        """Smooth data in 1/Mpc"""

        apply_smoothing(emulator, list_data_Mpc, fprint=fprint)
        for ii in range(len(list_data_Mpc)):
            if "p1d_Mpc_smooth" in list_data_Mpc[ii]:
                list_data_Mpc[ii]["p1d_Mpc"] = list_data_Mpc[ii][
                    "p1d_Mpc_smooth"
                ]

        return list_data_Mpc

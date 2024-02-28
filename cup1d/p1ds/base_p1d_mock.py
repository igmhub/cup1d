import numpy as np
import matplotlib.pyplot as plt

import cup1d
from cup1d.p1ds.base_p1d_data import BaseDataP1D


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

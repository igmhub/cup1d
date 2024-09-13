"""Class to generate a mock P1D from another P1D object and an emulator"""

import numpy as np
from lace.emulator import gp_emulator
from cup1d.p1ds.base_p1d_mock import BaseMockP1D
from cup1d.p1ds import (
    data_Chabanier2019,
    data_Karacayli2022,
    data_QMLE_Ohio,
    data_Karacayli2024,
)
from cup1d.likelihood import lya_theory


class Mock_P1D(BaseMockP1D):
    """Class to generate a mock P1D from another P1D object and a theory"""

    def __init__(
        self,
        emulator,
        data_label="Chabanier2019",
        z_min=0,
        z_max=10,
        add_noise=False,
        seed=0,
        fid_sim_igm="mpg_central",
        cosmo_fid=None,
        zs=None,
        k_kms=None,
        like_params=[],
    ):
        """Copy data and replace P1D signal using theory

        Parameters
        ----------
        emulator : object
            Emulator object
        data_label : string
            Data label to load data from
        z_min : float
            Minimum redshift
        z_max : float
            Maximum redshift
        add_noise : boolean
            Add noise to P1D
        seed : int
            Seed for random number generator
        fid_sim_igm : string
            IGM model to use for generating data
        zs : array
            Redshifts
        k_kms : array
            Wavenumbers in km/s

        """

        # load original data
        self.data_label = data_label
        if data_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(
                z_min=z_min, z_max=z_max
            )
        elif data_label == "QMLE_Ohio":
            data = data_QMLE_Ohio.P1D_QMLE_Ohio(
                z_min=z_min, z_max=z_max, **kwargs
            )
        elif data_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022(
                z_min=z_min, z_max=z_max
            )
        elif data_label == "Karacayli2024":
            data = data_Karacayli2024.P1D_Karacayli2024(
                z_min=z_min, z_max=z_max
            )
        else:
            raise ValueError("Unknown data_label", data_label)

        # keep value of things not provided from data
        if (zs is None) & (k_kms is None):
            zs = data.z
            k_kms = data.k_kms
            # at each z will update value of p1d
            Pk_kms = data.Pk_kms.copy()
            # keep value of cov matrix from data
            cov_Pk_kms = data.cov_Pk_kms.copy()
        else:
            raise ValueError(
                "Providing zs and k_kms to create mock data is not implemented yet"
            )

        # remove nan values
        for iz in range(len(zs)):
            _ = np.argwhere(np.isfinite(np.diag(cov_Pk_kms[iz])))[:, 0]
            k_kms[iz] = k_kms[iz][_]
            Pk_kms[iz] = Pk_kms[iz][_]
            cov_Pk_kms[iz] = cov_Pk_kms[iz][
                slice(_[0], _[-1] + 1), slice(_[0], _[-1] + 1)
            ]

        # setup theory
        theory = lya_theory.Theory(
            zs=zs,
            emulator=emulator,
            fid_sim_igm=fid_sim_igm,
            true_sim_igm=fid_sim_igm,
            cosmo_fid=cosmo_fid,
        )

        # evaluate theory at k_kms, for all redshifts
        out = theory.get_p1d_kms(
            zs,
            k_kms,
            like_params=like_params,
            return_emu_params=True,
            return_blob=True,
        )
        p1ds, blob, emu_params = out

        theory.set_truth()
        self.truth = theory.truth

        for iz, z in enumerate(zs):
            Pk_kms[iz] = p1ds[iz]

        super().__init__(
            z=zs,
            k_kms=k_kms,
            Pk_kms=Pk_kms,
            cov_Pk_kms=cov_Pk_kms,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
        )

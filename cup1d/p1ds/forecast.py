"""Class to generate a mock P1D from another P1D object and an emulator"""

import numpy as np
from cup1d.p1ds.base_p1d_mock import BaseMockP1D

from cup1d.p1ds.observations import (
    data_Chabanier2019,
    data_DESIY1,
)

from cup1d.p1ds.simulations import (
    challenge_DESIY1,
)


class Forecast_P1D(BaseMockP1D):
    """Class to generate a Forecast P1D

    Provide data_label to load covariance matrix and range of redshifts and scales from observation

    Provide theory with an emulator to generate P1D signal at the same redshifts and scales as the data_label
    """

    def __init__(
        self,
        theory,
        data_label="Chabanier2019",
        z_min=0,
        z_max=10,
        add_noise=False,
        seed=0,
        p1d_fname=None,
        path_data=None,
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

        # load covariance from data file
        self.data_label = data_label
        if data_label == "Chabanier2019":
            data_from_obs = data_Chabanier2019.read_from_file()
        # elif data_label == "Karacayli2024":
        #     data_from_obs = data_Karacayli2024.read_from_file()
        elif "DESIY1" in data_label:
            data_from_obs = data_DESIY1.P1D_DESIY1(data_label=data_label)
        elif data_label == "challenge_DESIY1":
            if p1d_fname is None:
                raise ValueError(
                    "Must provide p1d_fname if loading challenge_DESIY1 data"
                )
            else:
                data_from_obs = challenge_DESIY1.P1D_challenge_DESIY1(
                    theory,
                    p1d_fname=p1d_fname,
                    z_min=z_min,
                    z_max=z_max,
                    path_data=path_data,
                )
        else:
            raise ValueError("Unknown data_label", data_label)

        # evaluate theory at k_kms, for all redshifts. get Pk_kms from emulator
        zs = np.array(data_from_obs.z)
        theory.model_igm.set_fid_igm(zs)
        theory.set_fid_cosmo(zs)
        Pk_kms = theory.get_p1d_kms(zs, data_from_obs.k_kms, return_blob=False)
        full_Pk_kms = np.concatenate(np.array(Pk_kms, dtype=object)).reshape(-1)

        super().__init__(
            zs=data_from_obs.z,
            k_kms=data_from_obs.k_kms,
            Pk_kms=Pk_kms,
            cov_Pk_kms=data_from_obs.cov_Pk_kms,
            cov_stat=data_from_obs.covstat_Pk_kms,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            full_zs=data_from_obs.full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=data_from_obs.full_cov_Pk_kms,
            full_cov_stat_kms=data_from_obs.full_cov_stat_Pk_kms,
        )

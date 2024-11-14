"""Class to generate a mock P1D from another P1D object and an emulator"""

import numpy as np
from lace.emulator import gp_emulator
from lace.cosmo import camb_cosmo
from cup1d.p1ds.base_p1d_mock import BaseMockP1D

from cup1d.p1ds import (
    data_Chabanier2019,
    data_Karacayli2022,
    data_QMLE_Ohio,
    data_Karacayli2024,
    data_DESIY1,
)
from cup1d.likelihood import lya_theory
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_igm import IGM


class Mock_P1D(BaseMockP1D):
    """Class to generate a mock P1D from another P1D object and a theory"""

    def __init__(
        self,
        theory,
        true_cosmo,
        data_label="Chabanier2019",
        z_min=0,
        z_max=10,
        add_noise=False,
        seed=0,
        p1d_fname=None,
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
        elif data_label == "DESIY1":
            if p1d_fname is None:
                raise ValueError(
                    "Must provide p1d_fname if loading DESI_Y1 data"
                )
            else:
                data_from_obs = data_DESIY1.read_from_file(p1d_fname=p1d_fname)
        else:
            raise ValueError("Unknown data_label", data_label)

        (
            zs,
            k_kms,
            Pk_kms,
            cov_Pk_kms,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            blind,
        ) = data_from_obs

        # evaluate theory at k_kms, for all redshifts
        theory.model_igm.set_fid_igm(np.array(zs))
        theory.set_fid_cosmo(zs, input_cosmo=true_cosmo)
        Pk_kms = theory.get_p1d_kms(np.array(zs), k_kms, return_blob=False)
        full_Pk_kms = np.concatenate(np.array(Pk_kms)).reshape(-1)

        super().__init__(
            zs=zs,
            k_kms=k_kms,
            Pk_kms=Pk_kms,
            cov_Pk_kms=cov_Pk_kms,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            theory=theory,
        )

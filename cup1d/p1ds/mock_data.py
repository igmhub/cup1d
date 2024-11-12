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
        elif data_label == "Karacayli2024":
            data_from_obs = data_Karacayli2024.read_from_file()
        elif data_label == "DESI_Y1":
            if p1d_fname is None:
                raise ValueError(
                    "Must provide p1d_fname if loading DESI_Y1 data"
                )
            else:
                data_from_obs = data_DESI_Y1.read_from_file(p1d_fname=p1d_fname)
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
        Pk_kms = theory.get_p1d_kms(zs, k_kms)
        full_Pk_kms = Pk_kms.reshape(-1)

        self.set_truth(theory, zs)

        super().__init__(
            z=zs,
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
        )

    def set_truth(self, theory, zs):
        # setup fiducial cosmology
        self.truth = {}

        sim_cosmo = theory.cosmo_model_fid["cosmo"].cosmo

        self.truth["cosmo"] = {}
        self.truth["cosmo"]["ombh2"] = sim_cosmo.ombh2
        self.truth["cosmo"]["omch2"] = sim_cosmo.omch2
        self.truth["cosmo"]["As"] = sim_cosmo.InitPower.As
        self.truth["cosmo"]["ns"] = sim_cosmo.InitPower.ns
        self.truth["cosmo"]["nrun"] = sim_cosmo.InitPower.nrun
        self.truth["cosmo"]["H0"] = sim_cosmo.H0
        self.truth["cosmo"]["mnu"] = camb_cosmo.get_mnu(sim_cosmo)

        self.truth["linP"] = {}
        blob_params = ["Delta2_star", "n_star", "alpha_star"]
        blob = theory.cosmo_model_fid["cosmo"].get_linP_params()
        for ii in range(len(blob_params)):
            self.truth["linP"][blob_params[ii]] = blob[blob_params[ii]]

        self.truth["igm"] = {}
        zs = np.array(zs)
        self.truth["igm"]["z"] = zs
        self.truth["igm"]["tau_eff"] = theory.model_igm.F_model.get_tau_eff(zs)
        self.truth["igm"]["gamma"] = theory.model_igm.T_model.get_gamma(zs)
        self.truth["igm"]["sigT_kms"] = theory.model_igm.T_model.get_sigT_kms(
            zs
        )
        if theory.model_igm.yes_kF:
            self.truth["igm"]["kF_kms"] = theory.model_igm.P_model.get_kF_kms(
                zs
            )
        else:
            self.truth["igm"]["kF_kms"] = np.zeros_like(zs)

        self.truth["cont"] = {}
        for ii in range(2):
            self.truth["cont"][
                "ln_SiIII_" + str(ii)
            ] = theory.model_cont.fid_SiIII[-1 - ii]
            self.truth["cont"][
                "ln_SiII_" + str(ii)
            ] = theory.model_cont.fid_SiII[-1 - ii]
            self.truth["cont"][
                "ln_A_damp_" + str(ii)
            ] = theory.model_cont.fid_HCD[-1 - ii]
            self.truth["cont"]["ln_SN_" + str(ii)] = theory.model_cont.fid_SN[
                -1 - ii
            ]
            self.truth["cont"]["ln_AGN_" + str(ii)] = theory.model_cont.fid_AGN[
                -1 - ii
            ]

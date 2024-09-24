import os, sys
import numpy as np
from scipy.interpolate import interp1d

from lace.utils import poly_p1d
from lace.cosmo import camb_cosmo
from cup1d.likelihood import CAMB_model
from cup1d.p1ds.base_p1d_mock import BaseMockP1D
from cup1d.p1ds import (
    data_PD2013,
    data_Chabanier2019,
    data_QMLE_Ohio,
    data_Karacayli2022,
)
from cup1d.likelihood import lya_theory
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_igm import IGM


class Nyx_P1D(BaseMockP1D):
    """Class to load a Nyx simulation as a mock data object.
    Can use PD2013 or Chabanier2019 covmats"""

    def __init__(
        self,
        testing_data,
        emulator=None,
        apply_smoothing=True,
        input_sim="nyx_central",
        z_min=0,
        z_max=10,
        data_cov_label="Chabanier2019",
        data_cov_factor=1.0,
        add_syst=True,
        add_noise=False,
        seed=0,
        z_star=3.0,
        kp_kms=0.009,
        true_SiII=-10,
        true_SiIII=-10,
        true_HCD=-6,
        true_SN=-10,
        fprint=print,
    ):
        """Read mock P1D from MP-Gadget sims, and returns mock measurement:
        - testing_data: p1d measurements from Nyx sims
        - input_sim: check available options in testing_data
        - z_max: maximum redshift to use in mock data
        - data_cov_label: P1D covariance to use (Chabanier2019 or PD2013)
        - data_cov_factor: multiply covariance by this factor
        - add_syst: Include systematic estimates in covariance matrices
        """

        # covariance matrix settings
        self.add_syst = add_syst
        self.data_cov_factor = data_cov_factor
        self.data_cov_label = data_cov_label
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.true_SiIII = true_SiIII
        self.true_HCD = true_HCD

        # store sim data
        self.input_sim = input_sim

        if apply_smoothing & (emulator is not None):
            self.testing_data = super().set_smoothing_Mpc(
                emulator, testing_data, fprint=fprint
            )
        else:
            fprint("No smoothing is applied")
            self.testing_data = testing_data

        # store cosmology used in the simulation
        dkms_dMpc = []
        for ii in range(len(testing_data)):
            dkms_dMpc.append(testing_data[ii]["dkms_dMpc"])
        self.dkms_dMpc = np.array(dkms_dMpc)

        # setup P1D using covariance and testing sim
        zs, k_kms, Pk_kms, cov = self._load_p1d()

        # setup theory
        model_igm = IGM(np.array(zs), fid_sim_igm=input_sim)
        model_cont = Contaminants(
            fid_SiIII=true_SiIII,
            fid_SiII=true_SiII,
            fid_HCD=true_HCD,
            fid_SN=true_SN,
        )
        true_cosmo = self._get_cosmo()
        theory = lya_theory.Theory(
            zs=np.array(zs),
            emulator=emulator,
            fid_cosmo=true_cosmo,
            model_igm=model_igm,
            model_cont=model_cont,
        )
        self.set_truth(theory, zs)

        # apply contaminants
        for iz, z in enumerate(zs):
            mF = model_igm.F_model.get_mean_flux(z)
            M_of_z = theory.cosmo_model_fid["M_of_zs"][iz]
            cont_total = model_cont.get_contamination(z, k_kms[iz], mF, M_of_z)
            Pk_kms[iz] *= cont_total

        # setup base class
        super().__init__(
            zs,
            k_kms,
            Pk_kms,
            cov,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
        )

        return

    def _get_cosmo(self):
        # get cosmology
        fname = os.environ["NYX_PATH"] + "nyx_emu_cosmo_Oct2023.npy"
        data_cosmo = np.load(fname, allow_pickle=True)

        true_cosmo = None
        for ii in range(len(data_cosmo)):
            if data_cosmo[ii]["sim_label"] == self.input_sim:
                true_cosmo = camb_cosmo.get_Nyx_cosmology(
                    data_cosmo[ii]["cosmo_params"]
                )
                break
        if true_cosmo is None:
            raise ValueError(f"Cosmo not found in {fname} for {self.input_sim}")

        return true_cosmo

    def _get_igm(self):
        """Load IGM history"""
        fname = os.environ["NYX_PATH"] + "/IGM_histories.npy"
        igm_hist = np.load(fname, allow_pickle=True).item()
        if self.input_sim not in igm_hist:
            raise ValueError(
                self.input_sim
                + " not found in "
                + fname
                + r"\n Check out the LaCE script save_"
                + self.input_sim[:3]
                + "_IGM.py"
            )
        else:
            true_igm = igm_hist[self.input_sim]

        return true_igm

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
        self.truth["igm"]["label"] = self.input_sim
        self.truth["igm"]["z"] = zs
        self.truth["igm"]["tau_eff"] = theory.model_igm.F_model.get_tau_eff(zs)
        self.truth["igm"]["gamma"] = theory.model_igm.T_model.get_gamma(zs)
        self.truth["igm"]["sigT_kms"] = theory.model_igm.T_model.get_sigT_kms(
            zs
        )
        self.truth["igm"]["kF_kms"] = theory.model_igm.P_model.get_kF_kms(zs)

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

    def _load_p1d(self):
        # figure out dataset to mimic
        if self.data_cov_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label == "PD2013":
            data = data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label == "QMLE_Ohio":
            data = data_QMLE_Ohio.P1D_QMLE_Ohio()
        elif self.data_cov_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022()
        else:
            raise ValueError("Unknown data_cov_label", self.data_cov_label)

        # get redshifts in testing simulation
        z_sim = np.array([data["z"] for data in self.testing_data])

        # unit conversion, at zmin to get lowest possible k_min_kms
        dkms_dMpc_zmin = self.dkms_dMpc[np.argmin(z_sim)]

        # Get k_min for the sim data & cut k values below that
        k_min_Mpc = self.testing_data[0]["k_Mpc"][0]
        if k_min_Mpc == 0:
            k_min_Mpc = self.testing_data[0]["k_Mpc"][1]
        k_min_kms = k_min_Mpc / dkms_dMpc_zmin

        k_kms = []
        Pk_kms = []
        cov = []
        zs = []
        # Set P1D and covariance for each redshift (from low-z to high-z)
        for iz in range(len(z_sim)):
            z = z_sim[iz]
            iz_data = np.argmin(abs(data.z - z))

            Ncull = np.sum(data.k_kms[iz_data] < k_min_kms)
            _k_kms = data.k_kms[iz_data][Ncull:]
            k_kms.append(_k_kms)

            # convert Mpc to km/s
            data_k_Mpc = np.array(_k_kms) * self.dkms_dMpc[iz]

            # find testing data for this redshift
            sim_k_Mpc = self.testing_data[iz]["k_Mpc"].copy()
            sim_p1d_Mpc = self.testing_data[iz]["p1d_Mpc"].copy()

            # mask k=0 if present
            if sim_k_Mpc[0] == 0:
                sim_k_Mpc = sim_k_Mpc[1:]
                sim_p1d_Mpc = sim_p1d_Mpc[1:]

            interp_sim_Mpc = interp1d(sim_k_Mpc, sim_p1d_Mpc, "cubic")
            sim_p1d_kms = interp_sim_Mpc(data_k_Mpc) * self.dkms_dMpc[iz]

            # append redshift, p1d and covar
            zs.append(z)
            Pk_kms.append(sim_p1d_kms)

            # Now get covariance from the nearest z bin in data
            cov_mat = data.get_cov_iz(iz_data)
            # Cull low k cov data and multiply by input factor
            cov_mat = self.data_cov_factor * cov_mat[Ncull:, Ncull:]
            cov.append(cov_mat)

        return zs, k_kms, Pk_kms, cov

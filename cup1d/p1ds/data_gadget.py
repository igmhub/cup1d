import os
import sys

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from lace.cosmo import camb_cosmo
from cup1d.p1ds.base_p1d_mock import BaseMockP1D
from cup1d.p1ds import (
    data_PD2013,
    data_Chabanier2019,
    data_QMLE_Ohio,
    data_Karacayli2022,
    data_DESIY1,
)


class Gadget_P1D(BaseMockP1D):
    """Class to load an MP-Gadget simulation as a mock data object.
    Can use PD2013 or Chabanier2019 covmats"""

    def __init__(
        self,
        theory,
        testing_data,
        apply_smoothing=True,
        input_sim="mpg_central",
        data_cov_label="Chabanier2019",
        add_syst=True,
        add_noise=False,
        seed=0,
        z_min=0,
        z_max=10,
        path_data=None,
        interp_to_cov=False,
    ):
        """Read mock P1D from MP-Gadget sims, and returns mock measurement:
        - testing_data: p1d measurements from Gadget sims
        - input_sim: check available options in testing_data
        - z_max: maximum redshift to use in mock data
        - data_cov_label: P1D covariance to use (Chabanier2019 or PD2013)
        - data_cov_factor: multiply covariance by this factor
        - add_syst: Include systematic estimates in covariance matrices
        - interp_to_cov: if true, interpolate simulations results to the redshifts
            and scales of the covariance matrix. if not, the other way around
        """

        # covariance matrix settings
        self.add_syst = add_syst
        self.data_cov_label = data_cov_label
        self.input_sim = input_sim

        if apply_smoothing:
            self.testing_data = super().set_smoothing_Mpc(
                theory.emulator, testing_data
            )
        else:
            print("No smoothing is applied")
            self.testing_data = testing_data

        # store cosmology used in the simulation
        dkms_dMpc = []
        for ii in range(len(testing_data)):
            dkms_dMpc.append(testing_data[ii]["dkms_dMpc"])
        self.dkms_dMpc = np.array(dkms_dMpc)

        if interp_to_cov:
            prepare_mock = self._load_p1d_to_cov
        else:
            prepare_mock = self._load_p1d
        (
            zs,
            k_kms,
            Pk_kms,
            cov,
            cov_stat,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
        ) = prepare_mock(theory, path_data=path_data)

        # set theory (just to save truth)
        zs = np.array(zs)
        theory.model_igm.set_fid_igm(zs)
        theory.set_fid_cosmo(zs)

        # apply contaminants
        syst_total = theory.model_syst.get_contamination(zs, k_kms)
        mF = theory.model_igm.models["F_model"].get_mean_flux(zs)
        M_of_z = theory.fid_cosmo["M_of_zs"]
        cont_all = theory.model_cont.get_contamination(zs, k_kms, mF, M_of_z)
        # print("sys", syst_total)
        # print("mult", mult_cont_total)
        # print("add", add_cont_total)

        full_Pk_kms = []
        for iz, z in enumerate(zs):
            # Pcont = (mul_metal * HCD * IC_corr * Pemu + add_metal) * syst
            Pk_kms[iz] = (
                cont_all["cont_HCD"][iz]
                * cont_all["cont_mul_metals"][iz]
                * cont_all["IC_corr"][iz]
                * Pk_kms[iz]
                + cont_all["cont_add_metals"][iz]
            ) * syst_total[iz]
            full_Pk_kms.append(Pk_kms[iz])
        full_Pk_kms = np.concatenate(full_Pk_kms)

        # setup base class
        super().__init__(
            zs,
            k_kms,
            Pk_kms,
            cov,
            full_zs=full_zs,
            full_Pk_kms=full_Pk_kms,
            full_cov_kms=full_cov_kms,
            full_cov_stat_kms=full_cov_stat_kms,
            cov_stat=cov_stat,
            add_noise=add_noise,
            seed=seed,
            z_min=z_min,
            z_max=z_max,
            theory=theory,
        )

        return

    # def set_truth(self, theory, zs):
    #     # setup fiducial cosmology
    #     self.truth = {}

    #     sim_cosmo = theory.fid_cosmo["cosmo"].cosmo

    #     self.truth["cosmo"] = {}
    #     self.truth["cosmo"]["ombh2"] = sim_cosmo.ombh2
    #     self.truth["cosmo"]["omch2"] = sim_cosmo.omch2
    #     self.truth["cosmo"]["As"] = sim_cosmo.InitPower.As
    #     self.truth["cosmo"]["ns"] = sim_cosmo.InitPower.ns
    #     self.truth["cosmo"]["nrun"] = sim_cosmo.InitPower.nrun
    #     self.truth["cosmo"]["H0"] = sim_cosmo.H0
    #     self.truth["cosmo"]["mnu"] = camb_cosmo.get_mnu(sim_cosmo)

    #     self.truth["linP"] = {}
    #     blob_params = ["Delta2_star", "n_star", "alpha_star"]
    #     blob = theory.fid_cosmo["cosmo"].get_linP_params()
    #     for ii in range(len(blob_params)):
    #         self.truth["linP"][blob_params[ii]] = blob[blob_params[ii]]

    #     self.truth["igm"] = {}
    #     self.truth["igm"]["label"] = self.input_sim
    #     zs = np.array(zs)
    #     self.truth["igm"]["z"] = zs
    #     self.truth["igm"]["tau_eff"] = theory.model_igm.F_model.get_tau_eff(zs)
    #     self.truth["igm"]["gamma"] = theory.model_igm.T_model.get_gamma(zs)
    #     self.truth["igm"]["sigT_kms"] = theory.model_igm.T_model.get_sigT_kms(
    #         zs
    #     )
    #     self.truth["igm"]["kF_kms"] = theory.model_igm.P_model.get_kF_kms(zs)

    #     self.truth["cont"] = {}
    #     for ii in range(2):
    #         self.truth["cont"][
    #             "ln_SiIII_" + str(ii)
    #         ] = theory.model_cont.fid_SiIII[-1 - ii][-1]
    #         self.truth["cont"][
    #             "d_SiIII_" + str(ii)
    #         ] = theory.model_cont.fid_SiIII[-1 - ii][0]
    #         self.truth["cont"][
    #             "ln_SiII_" + str(ii)
    #         ] = theory.model_cont.fid_SiII[-1 - ii][-1]
    #         self.truth["cont"][
    #             "d_SiII_" + str(ii)
    #         ] = theory.model_cont.fid_SiII[-1 - ii][0]
    #         self.truth["cont"][
    #             "ln_A_damp_" + str(ii)
    #         ] = theory.model_cont.fid_HCD[-1 - ii]
    #         self.truth["cont"]["ln_SN_" + str(ii)] = theory.model_cont.fid_SN[
    #             -1 - ii
    #         ]
    #         self.truth["cont"]["ln_AGN_" + str(ii)] = theory.model_cont.fid_AGN[
    #             -1 - ii
    #         ]

    def _load_p1d(self, theory, path_data=None):
        """Interpolate data to the redshifts and scales of the covariance matrix"""
        # figure out dataset to mimic
        if self.data_cov_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label == "PD2013":
            data = data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label == "QMLE_Ohio":
            data = data_QMLE_Ohio.P1D_QMLE_Ohio()
        elif self.data_cov_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022()
        elif self.data_cov_label.startswith("DESIY1"):
            data = data_DESIY1.P1D_DESIY1(
                data_label=self.data_cov_label, path_data=path_data
            )
        else:
            raise ValueError("Unknown data_cov_label", self.data_cov_label)

        # set interpolator
        z_sim = []
        k_Mpc_sim = []
        p1d_Mpc_sim = []
        for sim in self.testing_data:
            z_sim.append(sim["z"])
            # select only data within 10 Mpc
            _ = (sim["k_Mpc"] > 0) & (sim["k_Mpc"] < 10)
            k_Mpc_sim.append(sim["k_Mpc"][_])
            p1d_Mpc_sim.append(sim["p1d_Mpc"][_])
        z_sim = np.array(z_sim)
        k_Mpc_sim = np.array(k_Mpc_sim)
        p1d_Mpc_sim = np.array(p1d_Mpc_sim)

        theory.set_fid_cosmo(z_sim)

        interp = RegularGridInterpolator(
            (z_sim, k_Mpc_sim[0]), np.log(p1d_Mpc_sim)
        )

        zs = data.z
        k_kms = []
        Pk_kms = []
        cov = []
        cov_stat = []
        full_zs = []
        for ii, _k_kms in enumerate(data.k_kms):
            dkms_dMpc = theory.fid_cosmo["cosmo"].dkms_dMpc(zs[ii])
            # convert Mpc to km/s
            data_k_Mpc = _k_kms * dkms_dMpc
            # cutting scales too large for the simulation
            _ = data_k_Mpc >= k_Mpc_sim[0].min()
            Pk_Mpc = np.exp(interp((zs[ii], data_k_Mpc[_])))
            # convert Mpc to km/s
            k_kms.append(_k_kms[_])
            Pk_kms.append(Pk_Mpc * dkms_dMpc)
            cov.append(data.cov_Pk_kms[ii][_][:, _])
            cov_stat.append(data.covstat_Pk_kms[ii][_][:, _])
            full_zs.append(zs[ii] * np.ones(len(_k_kms[_])))
        full_zs = np.concatenate(full_zs)
        full_k_kms = np.concatenate(k_kms)
        full_Pk_kms = np.concatenate(Pk_kms)

        # remove scales not used from cov matrix
        full_cov_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        full_cov_stat_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        for i0 in range(len(full_Pk_kms)):
            ind0 = np.argwhere(full_zs[i0] == data.full_zs)[:, 0]
            j0 = ind0[np.argmin(abs(full_k_kms[i0] - data.full_k_kms[ind0]))]
            for i1 in range(len(full_Pk_kms)):
                ind1 = np.argwhere(full_zs[i1] == data.full_zs)[:, 0]
                j1 = ind1[
                    np.argmin(abs(full_k_kms[i1] - data.full_k_kms[ind1]))
                ]
                full_cov_kms[i0, i1] = data.full_cov_Pk_kms[j0, j1]
                full_cov_stat_kms[i0, i1] = data.full_cov_stat_Pk_kms[j0, j1]

        return (
            zs,
            k_kms,
            Pk_kms,
            cov,
            cov_stat,
            full_zs,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
        )

    def _load_p1d_to_cov(self, theory, path_data=None):
        """Interpolate cov matrix to the redshifts of the data, data to scales of cov matrix"""
        # figure out dataset to mimic
        if self.data_cov_label == "Chabanier2019":
            data = data_Chabanier2019.P1D_Chabanier2019(add_syst=self.add_syst)
        elif self.data_cov_label == "PD2013":
            data = data_PD2013.P1D_PD2013(add_syst=self.add_syst)
        elif self.data_cov_label == "QMLE_Ohio":
            data = data_QMLE_Ohio.P1D_QMLE_Ohio()
        elif self.data_cov_label == "Karacayli2022":
            data = data_Karacayli2022.P1D_Karacayli2022()
        elif self.data_cov_label.startswith("DESIY1"):
            data = data_DESIY1.P1D_DESIY1(
                data_label=self.data_cov_label, path_data=path_data
            )
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
        cov_stat = []
        zs_native = []
        zs_cov = []

        for iz in range(len(z_sim)):
            z = z_sim[iz]
            iz_data = np.argmin(np.abs(data.z - z))
            # print(z, data.z[iz_data])

            # interpolate native k to cov k
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
            zs_native.append(z)
            zs_cov.append(data.z[iz_data])
            Pk_kms.append(sim_p1d_kms)
            # # Now get covariance from the nearest z bin in data
            cov.append(data.cov_Pk_kms[iz_data][Ncull:, Ncull:])
            cov_stat.append(data.covstat_Pk_kms[iz_data][Ncull:, Ncull:])

        full_k_kms = np.concatenate(k_kms)
        full_zs_native = []
        full_zs_cov = []
        for ii in range(len(k_kms)):
            full_zs_native.append(np.ones(len(k_kms[ii])) * zs_native[ii])
            full_zs_cov.append(np.ones(len(k_kms[ii])) * zs_cov[ii])
        full_zs_native = np.concatenate(full_zs_native)
        full_zs_cov = np.concatenate(full_zs_cov)
        full_Pk_kms = np.concatenate(Pk_kms)

        full_cov_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        full_cov_stat_kms = np.zeros((len(full_Pk_kms), len(full_Pk_kms)))
        for i0 in range(len(full_Pk_kms)):
            ind0 = np.argwhere(full_zs_cov[i0] == data.full_zs)[:, 0]
            j0 = ind0[np.argmin(abs(full_k_kms[i0] - data.full_k_kms[ind0]))]
            for i1 in range(len(full_Pk_kms)):
                ind1 = np.argwhere(full_zs_cov[i1] == data.full_zs)[:, 0]
                j1 = ind1[
                    np.argmin(abs(full_k_kms[i1] - data.full_k_kms[ind1]))
                ]
                full_cov_kms[i0, i1] = data.full_cov_Pk_kms[j0, j1]
                full_cov_stat_kms[i0, i1] = data.full_cov_stat_Pk_kms[j0, j1]

        return (
            zs_native,
            k_kms,
            Pk_kms,
            cov,
            cov_stat,
            full_zs_native,
            full_Pk_kms,
            full_cov_kms,
            full_cov_stat_kms,
        )

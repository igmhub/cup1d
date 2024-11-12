import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

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
            warn("Perturbing data by adding Gaussian noise")
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

    def plot_igm(self):
        """Plot IGM histories"""

        # true IGM parameters
        pars_true = {}
        pars_true["z"] = self.truth["igm"]["z"]
        pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
        pars_true["gamma"] = self.truth["igm"]["gamma"]
        pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
        pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

        fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
        ax = ax.reshape(-1)

        arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
        latex_labs = [
            r"$\tau_\mathrm{eff}$",
            r"$\gamma$",
            r"$\sigma_T$",
            r"$k_F$",
        ]

        for ii in range(len(arr_labs)):
            _ = pars_true[arr_labs[ii]] != 0
            ax[ii].plot(
                pars_true["z"][_],
                pars_true[arr_labs[ii]][_],
                "o:",
                label="true",
            )

            ax[ii].set_ylabel(latex_labs[ii])
            if ii == 0:
                ax[ii].set_yscale("log")

            if (ii == 2) | (ii == 3):
                ax[ii].set_xlabel(r"$z$")

        plt.tight_layout()

    # def _get_cosmo(self, nyx_version="Jul2024"):
    #     # get cosmology
    #     fname = os.environ["NYX_PATH"] + "nyx_emu_cosmo_" + nyx_version + ".npy"
    #     data_cosmo = np.load(fname, allow_pickle=True)

    #     true_cosmo = None
    #     for ii in range(len(data_cosmo)):
    #         if data_cosmo[ii]["sim_label"] == self.input_sim:
    #             true_cosmo = camb_cosmo.get_Nyx_cosmology(
    #                 data_cosmo[ii]["cosmo_params"]
    #             )
    #             break
    #     if true_cosmo is None:
    #         raise ValueError(f"Cosmo not found in {fname} for {self.input_sim}")

    #     return true_cosmo

    # def _get_igm(self):
    #     """Load IGM history"""
    #     fname = os.environ["NYX_PATH"] + "/IGM_histories.npy"
    #     igm_hist = np.load(fname, allow_pickle=True).item()
    #     if self.input_sim not in igm_hist:
    #         raise ValueError(
    #             self.input_sim
    #             + " not found in "
    #             + fname
    #             + r"\n Check out the LaCE script save_"
    #             + self.input_sim[:3]
    #             + "_IGM.py"
    #         )
    #     else:
    #         true_igm = igm_hist[self.input_sim]

    #     return true_igm

    # def set_truth(self, theory, zs):
    #     # setup fiducial cosmology
    #     self.truth = {}

    #     sim_cosmo = theory.cosmo_model_fid["cosmo"].cosmo

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
    #     blob = theory.cosmo_model_fid["cosmo"].get_linP_params()
    #     for ii in range(len(blob_params)):
    #         self.truth["linP"][blob_params[ii]] = blob[blob_params[ii]]

    #     self.truth["igm"] = {}
    #     zs = np.array(zs)
    #     self.truth["igm"]["label"] = self.input_sim
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
    #         ] = theory.model_cont.fid_SiIII[-1 - ii]
    #         self.truth["cont"][
    #             "ln_SiII_" + str(ii)
    #         ] = theory.model_cont.fid_SiII[-1 - ii]
    #         self.truth["cont"][
    #             "ln_A_damp_" + str(ii)
    #         ] = theory.model_cont.fid_HCD[-1 - ii]
    #         self.truth["cont"]["ln_SN_" + str(ii)] = theory.model_cont.fid_SN[
    #             -1 - ii
    #         ]

    # def plot_igm(self):
    #     """Plot IGM histories"""

    #     # true IGM parameters
    #     pars_true = {}
    #     pars_true["z"] = self.truth["igm"]["z"]
    #     pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
    #     pars_true["gamma"] = self.truth["igm"]["gamma"]
    #     pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
    #     pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

    #     fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
    #     ax = ax.reshape(-1)

    #     arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
    #     latex_labs = [
    #         r"$\tau_\mathrm{eff}$",
    #         r"$\gamma$",
    #         r"$\sigma_T$",
    #         r"$k_F$",
    #     ]

    #     for ii in range(len(arr_labs)):
    #         _ = pars_true[arr_labs[ii]] != 0
    #         ax[ii].plot(
    #             pars_true["z"][_],
    #             pars_true[arr_labs[ii]][_],
    #             "o:",
    #             label="true",
    #         )

    #         ax[ii].set_ylabel(latex_labs[ii])
    #         if ii == 0:
    #             ax[ii].set_yscale("log")

    #         if (ii == 2) | (ii == 3):
    #             ax[ii].set_xlabel(r"$z$")

    #     plt.tight_layout()

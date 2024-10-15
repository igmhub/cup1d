import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from cup1d.p1ds.base_p1d_data import BaseDataP1D, _drop_zbins

from lace.cosmo import camb_cosmo
from cup1d.likelihood import lya_theory
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_igm import IGM


class P1D_DESIY1(BaseDataP1D):
    def __init__(
        self,
        fname=None,
        full_cov=False,
        z_min=2,
        z_max=10,
        true_sim_label=None,
    ):
        """Read measured P1D from file.
        - full_cov: for now, no covariance between redshift bins
        - z_min: z=2.0 bin is not recommended by Karacayli2024
        - z_max: maximum redshift to include"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        zs, k_kms, Pk_kms, cov = read_from_file(fname=fname, full_cov=full_cov)

        # set truth if possible
        if true_sim_label is not None:
            self.input_sim = true_sim_label
            model_igm = IGM(np.array(zs), fid_sim_igm=true_sim_label)
            model_cont = Contaminants(
                fid_SiII=[0, -10],
                fid_SiIII=[0, -10],
                fid_HCD=[0, -6],
                fid_SN=[0, -10],
            )
            true_cosmo = self._get_cosmo()
            theory = lya_theory.Theory(
                zs=np.array(zs),
                emulator=None,
                fid_cosmo=true_cosmo,
                model_igm=model_igm,
                model_cont=model_cont,
            )
            self.set_truth(theory, zs)

        super().__init__(zs, k_kms, Pk_kms, cov, z_min=z_min, z_max=z_max)

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


def read_from_file(fname=None, full_cov=False, kmin=1e-3, nknyq=0.5):
    """Read file containing P1D"""

    # folder storing P1D measurement
    if fname is not None:
        fname = fname
    else:
        datadir = BaseDataP1D.BASEDIR + "/QMLE_DESIY1/"
        fname = (
            datadir + "/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
        )

    hdu = fits.open(fname)

    zs_raw = hdu[1].data["Z"]
    k_kms_raw = hdu[1].data["K"]
    Pk_kms_raw = hdu[1].data["PLYA"]
    cov_raw = hdu[2].data.copy()
    diag_cov_raw = np.diag(cov_raw)

    z_unique = np.unique(zs_raw)

    zs = []
    k_kms = []
    Pk_kms = []
    cov = []
    for z in z_unique:
        dv = 2.99792458e5 * 0.8 / 1215.67 / (1 + z)
        k_nyq = np.pi / dv
        zs.append(z)
        mask = np.argwhere(
            (zs_raw == z)
            & (diag_cov_raw > 0)
            & (k_kms_raw > kmin)
            & (k_kms_raw < k_nyq * nknyq)
        )[:, 0]
        slice_cov = slice(mask[0], mask[-1] + 1)
        k_kms.append(np.array(k_kms_raw[mask]))
        Pk_kms.append(np.array(Pk_kms_raw[mask]))
        cov.append(np.array(cov_raw[slice_cov, slice_cov]))

    return zs, k_kms, Pk_kms, cov

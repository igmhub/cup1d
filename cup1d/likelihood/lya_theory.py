import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import lace
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import gp_emulator
from cup1d.nuisance import mean_flux_model
from cup1d.nuisance import thermal_model
from cup1d.nuisance import pressure_model
from cup1d.nuisance import metal_model
from cup1d.nuisance import hcd_model_McDonald2005
from cup1d.likelihood import CAMB_model


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        zs,
        zs_hires=None,
        emulator=None,
        verbose=False,
        F_model=None,
        T_model=None,
        P_model=None,
        hcd_model=None,
        z_star=3.0,
        kp_kms=0.009,
        metal_models=[],
        cosmo_fid=None,
        free_param_names=None,
        fid_sim_igm="mpg_central",
        true_sim_igm=None,
    ):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - emulator: object to interpolate simulated p1d
            - verbose: print information, useful to debug
            - F_model: mean flux model
            - T_model: thermal model
            - P_model: pressure model
            - metal_models: list of metal models to include
            - hcd_model: model for HCD contamination
            - cosmo_fid: fiducial cosmology used for fixed parameters
            - fid_sim_igm: IGM model assumed
            - true_sim_igm: if not None, true IGM model of the mock
        """

        self.verbose = verbose
        self.zs = zs
        self.zs_hires = zs_hires

        # specify pivot point used in compressed parameters
        self.z_star = z_star
        self.kp_kms = kp_kms

        # setup emulator
        if emulator is None:
            self.emulator = gp_emulator.GPEmulator(training_set="Pedersen21")
            print("Using default emulator: Pedersen21")
        else:
            self.emulator = emulator

        self.emu_kp_Mpc = self.emulator.kp_Mpc

        # load fiducial IGM history (used for fitting)
        self.fid_sim_igm = fid_sim_igm
        self.fid_igm = self.get_igm(fid_sim_igm)

        # load true IGM history (if any)
        if true_sim_igm is not None:
            self.true_sim_igm = true_sim_igm
            self.true_igm = self.get_igm(true_sim_igm)
        else:
            self.true_sim_igm = None

        # setup fiducial cosmology
        if not cosmo_fid:
            cosmo_fid = camb_cosmo.get_cosmology()

        # setup CAMB object for the fiducial cosmology and precompute some things
        if self.zs_hires is not None:
            _zs = np.concatenate([self.zs, self.zs_hires, [self.z_star]])
        else:
            _zs = np.concatenate([self.zs, [self.z_star]])
        _zs = np.unique(_zs)

        self.cosmo_model_fid = {}
        self.cosmo_model_fid["zs"] = _zs
        self.cosmo_model_fid["cosmo"] = CAMB_model.CAMBModel(
            zs=_zs,
            cosmo=cosmo_fid,
            z_star=self.z_star,
            kp_kms=self.kp_kms,
        )
        self.cosmo_model_fid["linP_Mpc_params"] = self.cosmo_model_fid[
            "cosmo"
        ].get_linP_Mpc_params(kp_Mpc=self.emu_kp_Mpc)
        self.cosmo_model_fid["M_of_zs"] = self.cosmo_model_fid[
            "cosmo"
        ].get_M_of_zs()

        # setup fiducial IGM models (from Gadget sims if not specified)
        if F_model is not None:
            self.F_model = F_model
        else:
            self.F_model = mean_flux_model.MeanFluxModel(
                free_param_names=free_param_names,
                fid_igm=self.fid_igm,
            )
        if T_model:
            self.T_model = T_model
        else:
            self.T_model = thermal_model.ThermalModel(
                free_param_names=free_param_names,
                fid_igm=self.fid_igm,
            )
        if P_model:
            self.P_model = P_model
        else:
            self.P_model = pressure_model.PressureModel(
                free_param_names=free_param_names,
                fid_igm=self.fid_igm,
            )

        # check whether we want to include metal contamination models
        self.metal_models = []
        for model in metal_models:
            X_model = copy.deepcopy(model)
            self.metal_models.append(X_model)
        # temporary hack
        if free_param_names:
            metal_param_names = []
            for name in free_param_names:
                if "ln_Si" in name:
                    # for now we only know how to vary SiIII
                    if "ln_SiIII_" not in name:
                        raise ValueError("implement metal param", name)
                    metal_param_names.append(name)
            if len(metal_param_names) > 0:
                # you have at least one free parameter for metals
                if len(self.metal_models) > 0:
                    raise ValueError("either pass metal_models or free_params")
                X_model = metal_model.MetalModel(
                    metal_label="SiIII", free_param_names=free_param_names
                )
                self.metal_models.append(X_model)

        # setup HCD model
        if hcd_model:
            self.hcd_model = copy.deepcopy(hcd_model)
        else:
            # close to zero HCD contamination
            self.hcd_model = hcd_model_McDonald2005.HCD_Model_McDonald2005(
                free_param_names=free_param_names
            )

    def fixed_background(self, like_params):
        """Check if any of the input likelihood parameters would change
        the background expansion of the fiducial cosmology"""

        # look for parameters that would change background
        for par in like_params:
            if par.name in ["ombh2", "omch2", "H0", "mnu", "cosmomc_theta"]:
                return False

        return True

    def set_truth(self):
        # setup fiducial cosmology
        self.truth = {}

        sim_cosmo = self.cosmo_model_fid["cosmo"].cosmo

        self.truth["ombh2"] = sim_cosmo.ombh2
        self.truth["omch2"] = sim_cosmo.omch2
        self.truth["As"] = sim_cosmo.InitPower.As
        self.truth["ns"] = sim_cosmo.InitPower.ns
        self.truth["nrun"] = sim_cosmo.InitPower.nrun
        self.truth["H0"] = sim_cosmo.H0
        self.truth["mnu"] = camb_cosmo.get_mnu(sim_cosmo)

        blob_params = ["Delta2_star", "n_star", "alpha_star"]
        blob = self.cosmo_model_fid["cosmo"].get_linP_params()
        for ii in range(len(blob_params)):
            self.truth[blob_params[ii]] = blob[blob_params[ii]]

        self.truth["igm"] = self.true_igm

    def emulate_arr_p1d_Mpc(self, model, k_Mpc, return_covar=False, z=None):
        """Wrapper for emulator calls for GP emulator (workaroud should be move to LaCE)"""

        p1d = []
        cov_p1d = []
        for ii in range(len(model)):
            _res = self.emulator.emulate_p1d_Mpc(
                model[ii], k_Mpc[ii], z=z[ii], return_covar=return_covar
            )
            if return_covar:
                p1d.append(_res[0])
                cov_p1d.append(_res[1])
            else:
                p1d.append(_res)

        if return_covar:
            return p1d, cov_p1d
        else:
            return p1d

    def get_igm(self, sim_igm):
        """Load IGM history"""
        if sim_igm[:3] == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + "/data/sim_suites/Australia20/IGM_histories.npy"
        elif sim_igm[:3] == "nyx":
            fname = os.environ["NYX_PATH"] + "/IGM_histories.npy"
        else:
            raise ValueError("only mpg and nyx sim_igm implemented")

        try:
            igm_hist = np.load(fname, allow_pickle=True).item()
        except:
            raise ValueError(
                fname
                + "not found. You can produce it using LaCE"
                + r"\n script save_"
                + sim_igm[:3]
                + "_IGM.py"
            )
        else:
            if sim_igm not in igm_hist:
                raise ValueError(
                    sim_igm
                    + " not found in "
                    + fname
                    + r"\n Check out the LaCE script save_"
                    + sim_igm[:3]
                    + "_IGM.py"
                )
            else:
                fid_igm = igm_hist[sim_igm]

        return fid_igm

    def get_linP_Mpc_params_from_fiducial(self, zs, like_params):
        """Recycle linP_Mpc_params from fiducial model, when only varying
        primordial power spectrum (As, ns, nrun)"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        # differences in primordial power (at CMB pivot point)
        ratio_As = 1.0
        delta_ns = 0.0
        delta_nrun = 0.0
        for par in like_params:
            if par.name == "As":
                fid_As = self.cosmo_model_fid["cosmo"].cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.cosmo_model_fid["cosmo"].cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.cosmo_model_fid["cosmo"].cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale in primordial power
        ks_Mpc = self.cosmo_model_fid["cosmo"].cosmo.InitPower.pivot_scalar
        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(self.emu_kp_Mpc / ks_Mpc)

        # compute scalings
        delta_alpha_p = delta_nrun
        delta_n_p = delta_ns + delta_nrun * ln_kp_ks
        ln_ratio_A_p = (
            np.log(ratio_As)
            + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
        )

        # update values of linP_params at emulator pivot point, at each z
        linP_Mpc_params = []
        for z in zs:
            _ = np.argwhere(self.cosmo_model_fid["zs"] == z)[0, 0]
            zlinP = self.cosmo_model_fid["linP_Mpc_params"][_]
            linP_Mpc_params.append(
                {
                    "Delta2_p": zlinP["Delta2_p"] * np.exp(ln_ratio_A_p),
                    "n_p": zlinP["n_p"] + delta_n_p,
                    "alpha_p": zlinP["alpha_p"] + delta_alpha_p,
                }
            )

        return linP_Mpc_params

    def get_emulator_calls(
        self, zs, like_params=[], return_M_of_z=True, return_blob=False
    ):
        """Compute models that will be emulated, one per redshift bin.
        - like_params identify likelihood parameters to use.
        - return_M_of_z will also return conversion from Mpc to km/s
        - return_blob will return extra information about the call."""

        # compute linear power parameters at all redshifts, and H(z) / (1+z)
        if self.fixed_background(like_params):
            # use background and transfer functions from fiducial cosmology
            if self.verbose:
                print("recycle transfer function")
            linP_Mpc_params = self.get_linP_Mpc_params_from_fiducial(
                zs, like_params
            )
            M_of_zs = []
            for z in zs:
                _ = np.argwhere(self.cosmo_model_fid["zs"] == z)[0, 0]
                M_of_zs.append(self.cosmo_model_fid["M_of_zs"][_])
            M_of_zs = np.array(M_of_zs)

            if return_blob:
                blob = self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose:
                print("create new CAMB_model")
            camb_model = self.cosmo_model_fid["cosmo"].get_new_model(
                zs, like_params
            )
            linP_Mpc_params = camb_model.get_linP_Mpc_params(
                kp_Mpc=self.emu_kp_Mpc
            )
            M_of_zs = camb_model.get_M_of_zs()
            if return_blob:
                blob = self.get_blob(camb_model=camb_model)

        # loop over redshifts and store emulator calls
        emu_call = {}
        for key in self.emulator.emu_params:
            if (key == "Delta2_p") | (key == "n_p") | (key == "alpha_p"):
                emu_call[key] = np.zeros(len(zs))
                for ii in range(len(linP_Mpc_params)):
                    emu_call[key][ii] = linP_Mpc_params[ii][key]
            elif key == "mF":
                emu_call[key] = self.F_model.get_mean_flux(
                    zs, like_params=like_params
                )
            elif key == "gamma":
                emu_call[key] = self.T_model.get_gamma(
                    zs, like_params=like_params
                )
            elif key == "sigT_Mpc":
                emu_call[key] = (
                    self.T_model.get_sigT_kms(zs, like_params=like_params)
                    / M_of_zs
                )
            elif key == "kF_Mpc":
                emu_call[key] = (
                    self.P_model.get_kF_kms(zs, like_params=like_params)
                    * M_of_zs
                )
            elif key == "lambda_P":
                emu_call[key] = 1000 / (
                    self.P_model.get_kF_kms(zs, like_params=like_params)
                    * M_of_zs
                )
            else:
                raise ValueError(
                    "Not a theory model for emulator parameter", key
                )

        if return_M_of_z == True:
            if return_blob:
                return emu_call, M_of_zs, blob
            else:
                return emu_call, M_of_zs
        else:
            if return_blob:
                return emu_call, blob
            else:
                return emu_call

    def get_blobs_dtype(self):
        """Return the format of the extra information (blobs) returned
        by get_p1d_kms and used in emcee_sampler."""

        blobs_dtype = [
            ("Delta2_star", float),
            ("n_star", float),
            ("alpha_star", float),
            ("f_star", float),
            ("g_star", float),
            ("H0", float),
        ]
        return blobs_dtype

    def get_blob(self, camb_model=None):
        """Return extra information (blob) for the emcee_sampler."""

        if camb_model is None:
            Nblob = len(self.get_blobs_dtype())
            if Nblob == 1:
                return np.nan
            else:
                out = np.nan, *([np.nan] * (Nblob - 1))
                return out
        else:
            # compute linear power parameters for input cosmology
            params = self.cosmo_model_fid["cosmo"].get_linP_params()
            return (
                params["Delta2_star"],
                params["n_star"],
                params["alpha_star"],
                params["f_star"],
                params["g_star"],
                camb_model.cosmo.H0,
            )

    def get_blob_fixed_background(self, like_params):
        """Fast computation of blob when running with fixed background"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        # differences in primordial power (at CMB pivot point)
        ratio_As = 1.0
        delta_ns = 0.0
        delta_nrun = 0.0
        for par in like_params:
            if par.name == "As":
                fid_As = self.cosmo_model_fid["cosmo"].cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.cosmo_model_fid["cosmo"].cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.cosmo_model_fid["cosmo"].cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale of primordial power
        ks_Mpc = self.cosmo_model_fid["cosmo"].cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.cosmo_model_fid["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # get blob for fiducial cosmo
        fid_blob = self.get_blob(self.cosmo_model_fid["cosmo"])

        # rescale blobs
        delta_alpha_star = delta_nrun
        delta_n_star = delta_ns + delta_nrun * ln_kp_ks
        ln_ratio_A_star = (
            np.log(ratio_As)
            + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
        )

        alpha_star = fid_blob[2] + delta_alpha_star
        n_star = fid_blob[1] + delta_n_star
        Delta2_star = fid_blob[0] * np.exp(ln_ratio_A_star)

        return (Delta2_star, n_star, alpha_star) + fid_blob[3:]

    def get_p1d_kms(
        self,
        zs,
        k_kms,
        like_params=[],
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
    ):
        """Emulate P1D in velocity units, for all redshift bins,
        as a function of input likelihood parameters.
        It might also return a covariance from the emulator,
        or a blob with extra information for the emcee_sampler."""

        if self.emulator is None:
            raise ValueError("no emulator provided")

        # figure out emulator calls
        _res = self.get_emulator_calls(
            zs,
            like_params=like_params,
            return_M_of_z=True,
            return_blob=return_blob,
        )
        if return_blob:
            emu_call, M_of_z, blob = _res
        else:
            emu_call, M_of_z = _res

        # compute input k to emulator
        Nz = len(zs)
        length = 0
        for iz in range(Nz):
            if len(k_kms[iz]) > length:
                length = len(k_kms[iz])

        kin_Mpc = np.zeros((Nz, length))
        for iz in range(Nz):
            kin_Mpc[iz, : len(k_kms[iz])] = k_kms[iz] * M_of_z[iz]

        _res = self.emulator.emulate_p1d_Mpc(
            emu_call, kin_Mpc, return_covar=return_covar, z=self.zs
        )
        if return_covar:
            p1d_Mpc, cov_Mpc = _res
        else:
            p1d_Mpc = _res

        p1d_kms = []
        covars = []
        for iz in range(Nz):
            p1d_kms.append(p1d_Mpc[iz][: len(k_kms[iz])] * M_of_z[iz])
            if return_covar:
                if cov_Mpc is None:
                    covars.append(None)
                else:
                    covars.append(
                        cov_Mpc[iz][: len(k_kms[iz]), : len(k_kms[iz])]
                        * M_of_z[iz] ** 2
                    )

        # include multiplicative metal contamination
        # NEED TO BE UPDATED AS THE OTHER MODELS
        for X_model_fid in self.metal_models:
            X_model = X_model_fid.get_new_model(like_params)
            for iz, z in enumerate(zs):
                cont = X_model.get_contamination(
                    z=z, k_kms=k_kms[iz], mF=emu_call["mF"][iz]
                )
                p1d_kms[iz] *= cont

        # include HCD contamination
        # NEED TO BE UPDATED AS THE OTHER MODELS
        for iz, z in enumerate(zs):
            hcd_model = self.hcd_model.get_new_model(like_params)
            cont = hcd_model.get_contamination(z=z, k_kms=k_kms[iz])
            p1d_kms[iz] *= cont

        # decide what to return, and return it
        out = [p1d_kms]
        if return_covar:
            out.append(covars)
        if return_blob:
            out.append(blob)
        if return_emu_params:
            out.append(emu_call)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_parameters(self):
        """Return parameters in models, even if not free parameters"""

        # get parameters from CAMB model
        params = self.cosmo_model_fid["cosmo"].get_likelihood_parameters()

        # get parameters from nuisance IGM models
        for par in self.F_model.get_parameters():
            params.append(par)
        for par in self.T_model.get_sigT_kms_parameters():
            params.append(par)
        for par in self.T_model.get_gamma_parameters():
            params.append(par)
        for par in self.P_model.get_parameters():
            params.append(par)

        # get parameters from metal contamination models
        for metal in self.metal_models:
            for par in metal.get_parameters():
                params.append(par)

        # get parameters from HCD contamination models
        for par in self.hcd_model.get_parameters():
            params.append(par)

        if self.verbose:
            print("got parameters")
            for par in params:
                print(par.info_str())

        return params

    def plot_p1d(
        self, k_kms, like_params=[], plot_every_iz=1, k_kms_hires=None
    ):
        """Emulate and plot P1D in velocity units, for all redshift bins,
        as a function of input likelihood parameters"""

        if self.zs_hires is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            length = 1
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            length = 2

        for ii in range(length):
            if ii == 0:
                zs = self.zs
                k_kms_use = k_kms
            else:
                zs = self.zs_hires
                k_kms_use = k_kms_hires
            # ask emulator prediction for P1D in each bin
            emu_p1d = self.get_p1d_kms(zs, k_kms_use, like_params)

            # plot only few redshifts for clarity
            Nz = len(zs)
            for iz in range(0, Nz, plot_every_iz):
                col = plt.cm.jet(iz / (Nz - 1))
                ax[ii].plot(
                    k_kms_use[iz],
                    emu_p1d[iz] * k_kms_use[iz] / np.pi,
                    color=col,
                    label="z=%.1f" % zs[iz],
                )

            ax[ii].legend()
            ax[ii].set_ylabel(
                r"$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$"
            )
            ax[ii].set_yscale("log")
            ax[ii].set_xlabel(r"$k$ [s/km]")
        plt.show()

        return

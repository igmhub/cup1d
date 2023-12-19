import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import gp_emulator
from cup1d.nuisance import mean_flux_model
from cup1d.nuisance import thermal_model
from cup1d.nuisance import pressure_model
from cup1d.nuisance import metal_model
from cup1d.likelihood import CAMB_model


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        zs,
        emulator=None,
        verbose=False,
        F_model_fid=None,
        T_model_fid=None,
        P_model_fid=None,
        z_star=3.0,
        kp_kms=0.009,
        include_metals=[],
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
            - F_model_fid: fiducial mean flux model
            - T_model_fid: fiducial thermal model
            - P_model_fid: fiducial pressure model
            - include_metals: list of metal labels to include
            - cosmo_fid: fiducial cosmology used for fixed parameters
            - fid_sim_igm: IGM model assumed
            - true_sim_igm: if not None, true IGM model of the mock
        """

        self.verbose = verbose
        self.zs = zs

        # specify pivot point used in compressed parameters
        self.z_star = z_star
        self.kp_kms = kp_kms

        # setup emulator
        if emulator is None:
            self.emulator = gp_emulator.GPEmulator(training_set="Pedersen21")
            print("Using default emulator: Pedersen21")
        else:
            self.emulator = emulator

        # guess if emulator is GP or NN
        if hasattr(self.emulator, "nhidden"):
            self.get_emulator_calls = self.get_emulator_calls_nn
            self.call_emulator = self.emulator.emulate_arr_p1d_Mpc
        else:
            self.get_emulator_calls = self.get_emulator_calls_gp
            self.call_emulator = self.emulate_arr_p1d_Mpc_gp

        self.emu_kp_Mpc = self.emulator.kp_Mpc

        # load fiducial IGM history (used for fitting)
        self.fid_sim_igm = fid_sim_igm
        self.fid_igm = self.get_igm(fid_sim_igm)

        # load true IGM history
        if true_sim_igm is not None:
            self.true_sim_igm = true_sim_igm
            self.true_igm = self.get_igm(true_sim_igm)
        else:
            self.true_sim_igm = None

        # setup fiducial cosmology
        if not cosmo_fid:
            cosmo_fid = camb_cosmo.get_cosmology()

        # setup CAMB object for the fiducial cosmology and precompute some things
        self.cosmo_model_fid = CAMB_model.CAMBModel(
            zs=self.zs, cosmo=cosmo_fid, z_star=self.z_star, kp_kms=self.kp_kms
        )
        self.linP_Mpc_params_fid = self.cosmo_model_fid.get_linP_Mpc_params(
            kp_Mpc=self.emu_kp_Mpc
        )
        self.M_of_zs = self.cosmo_model_fid.get_M_of_zs()

        # setup fiducial IGM models (from Gadget sims if not specified)
        if F_model_fid:
            self.F_model_fid = F_model_fid
        else:
            self.F_model_fid = mean_flux_model.MeanFluxModel(
                free_param_names=free_param_names,
                fid_igm=self.fid_igm,
            )
        if T_model_fid:
            self.T_model_fid = T_model_fid
        else:
            self.T_model_fid = thermal_model.ThermalModel(
                free_param_names=free_param_names,
                fid_igm=self.fid_igm,
            )
        if P_model_fid:
            self.P_model_fid = P_model_fid
        else:
            self.P_model_fid = pressure_model.PressureModel(
                free_param_names=free_param_names,
                fid_igm=self.fid_igm,
            )
        self.F_model_emcee = copy.deepcopy(self.F_model_fid)
        self.T_model_emcee = copy.deepcopy(self.T_model_fid)
        self.P_model_emcee = copy.deepcopy(self.P_model_fid)

        # if self.true_sim_igm is not None:
        #     self.F_model_true = mean_flux_model.MeanFluxModel(
        #         free_param_names=free_param_names,
        #         fid_igm=self.true_igm,
        #     )
        #     self.T_model_true = thermal_model.ThermalModel(
        #         free_param_names=free_param_names,
        #         fid_igm=self.true_igm,
        #     )
        #     self.P_model_true = pressure_model.PressureModel(
        #         free_param_names=free_param_names,
        #         fid_igm=self.true_igm,
        #     )

        # check whether we want to include metal contamination models
        self.metal_models = []
        for metal_label in include_metals:
            X_model = metal_model.MetalModel(metal_label=metal_label)
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
                    raise ValueError(
                        "either pass include_metals or free_params"
                    )
                X_model = metal_model.MetalModel(
                    metal_label="SiIII", free_param_names=free_param_names
                )
                self.metal_models.append(X_model)

    def fixed_background(self, like_params):
        """Check if any of the input likelihood parameters would change
        the background expansion of the fiducial cosmology"""

        # look for parameters that would change background
        for par in like_params:
            if par.name in ["ombh2", "omch2", "H0", "mnu", "cosmomc_theta"]:
                return False

        return True

    def emulate_arr_p1d_Mpc_gp(
        self, model, logk_Mpc, return_covar=False, z=None
    ):
        """Wrapper for emulator calls for GP emulator (workaroud should be move to LaCE)"""

        p1d = []
        cov_p1d = []
        k_Mpc = 10**logk_Mpc
        for ii in range(len(model)):
            if return_covar:
                _p1d, _cov_p1d = self.emulator.emulate_p1d_Mpc(
                    model[ii], k_Mpc[ii], z=z[ii], return_covar=return_covar
                )
                p1d.append(_p1d)
                cov_p1d.append(_cov_p1d)
            else:
                p1d.append(
                    self.emulator.emulate_p1d_Mpc(
                        model[ii], k_Mpc[ii], z=z[ii], return_covar=return_covar
                    )
                )
        if return_covar:
            return p1d, cov_p1d
        else:
            return p1d

    def get_igm(self, sim_igm):
        """Load IGM history"""
        if sim_igm[:3] == "mpg":
            fname = (
                os.environ["LACE_REPO"]
                + "/src/lace/data/sim_suites/Australia20/IGM_histories.npy"
            )
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

    def get_linP_Mpc_params_from_fiducial(self, like_params):
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
                fid_As = self.cosmo_model_fid.cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.cosmo_model_fid.cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.cosmo_model_fid.cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale in primordial power
        ks_Mpc = self.cosmo_model_fid.cosmo.InitPower.pivot_scalar
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
        for zlinP in self.linP_Mpc_params_fid:
            linP_Mpc_params.append(
                {
                    "Delta2_p": zlinP["Delta2_p"] * np.exp(ln_ratio_A_p),
                    "n_p": zlinP["n_p"] + delta_n_p,
                    "alpha_p": zlinP["alpha_p"] + delta_alpha_p,
                }
            )

        return linP_Mpc_params

    def get_emulator_calls_nn(
        self, like_params=[], return_M_of_z=True, return_blob=False
    ):
        """Compute models that will be emulated, one per redshift bin.
        - like_params identify likelihood parameters to use.
        - return_M_of_z will also return conversion from Mpc to km/s
        - return_blob will return extra information about the call."""

        # useful while debugging Nyx emulator
        emu_params = self.emulator.emu_params
        if self.verbose:
            print("list of parameters expected by the emulator")
            print(emu_params)

        # setup IGM models using list of likelihood parameters
        igm_models = self.update_igm_models(like_params)
        F_model = igm_models["F_model"]
        T_model = igm_models["T_model"]
        P_model = igm_models["P_model"]

        # compute linear power parameters at all redshifts, and H(z) / (1+z)
        if self.fixed_background(like_params):
            # use background and transfer functions from fiducial cosmology
            if self.verbose:
                print("recycle transfer function")
            linP_Mpc_params = self.get_linP_Mpc_params_from_fiducial(
                like_params
            )
            M_of_zs = self.M_of_zs.copy()
            if return_blob:
                blob = self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose:
                print("create new CAMB_model")
            camb_model = self.cosmo_model_fid.get_new_model(like_params)
            linP_Mpc_params = camb_model.get_linP_Mpc_params(
                kp_Mpc=self.emu_kp_Mpc
            )
            M_of_zs = camb_model.get_M_of_zs()
            if return_blob:
                blob = self.get_blob(camb_model=camb_model)

        # loop over redshifts and store emulator calls
        emu_calls = []
        Nz = len(self.zs)
        emu_calls = np.zeros((Nz, len(self.emulator.emu_params)))
        for iz, z in enumerate(self.zs):
            for jj, param in enumerate(self.emulator.emu_params):
                if (
                    (param == "Delta2_p")
                    | (param == "n_p")
                    | (param == "alpha_p")
                ):
                    emu_calls[iz, jj] = linP_Mpc_params[iz][param]
                elif param == "mF":
                    emu_calls[iz, jj] = F_model.get_mean_flux(z)
                elif param == "gamma":
                    emu_calls[iz, jj] = T_model.get_gamma(z)
                elif param == "sigT_Mpc":
                    sigT_kms = T_model.get_sigT_kms(z)
                    emu_calls[iz, jj] = sigT_kms / M_of_zs[iz]
                elif param == "kF_Mpc":
                    kF_kms = P_model.get_kF_kms(z)
                    emu_calls[iz, jj] = kF_kms * M_of_zs[iz]
                elif param == "lambda_P":
                    kF_kms = P_model.get_kF_kms(z)
                    emu_calls[iz, jj] = 1000 / (kF_kms * M_of_zs[iz])

        if return_M_of_z == True:
            if return_blob:
                return emu_calls, M_of_zs, blob
            else:
                return emu_calls, M_of_zs
        else:
            if return_blob:
                return emu_calls, blob
            else:
                return emu_calls

    def get_emulator_calls_gp(
        self, like_params=[], return_M_of_z=True, return_blob=False
    ):
        """Compute models that will be emulated, one per redshift bin.
        - like_params identify likelihood parameters to use.
        - return_M_of_z will also return conversion from Mpc to km/s
        - return_blob will return extra information about the call."""

        # useful while debugging Nyx emulator
        emu_params = self.emulator.emu_params
        if self.verbose:
            print("list of parameters expected by the emulator")
            print(emu_params)

        # setup IGM models using list of likelihood parameters
        igm_models = self.get_igm_models(like_params)
        F_model = igm_models["F_model"]
        T_model = igm_models["T_model"]
        P_model = igm_models["P_model"]

        # compute linear power parameters at all redshifts, and H(z) / (1+z)
        if self.fixed_background(like_params):
            # use background and transfer functions from fiducial cosmology
            if self.verbose:
                print("recycle transfer function")
            linP_Mpc_params = self.get_linP_Mpc_params_from_fiducial(
                like_params
            )
            M_of_zs = self.cosmo_model_fid.get_M_of_zs()
            if return_blob:
                blob = self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose:
                print("create new CAMB_model")
            camb_model = self.cosmo_model_fid.get_new_model(like_params)
            linP_Mpc_params = camb_model.get_linP_Mpc_params(
                kp_Mpc=self.emu_kp_Mpc
            )
            M_of_zs = camb_model.get_M_of_zs()
            if return_blob:
                blob = self.get_blob(camb_model=camb_model)

        # loop over redshifts and store emulator calls
        emu_calls = []
        Nz = len(self.zs)
        for iz, z in enumerate(self.zs):
            # emulator parameters for linear power, at this redshift (in Mpc)
            model = linP_Mpc_params[iz]
            # emulator parameters for nuisance models, at this redshift
            model["mF"] = F_model.get_mean_flux(z)
            model["gamma"] = T_model.get_gamma(z)
            sigT_kms = T_model.get_sigT_kms(z)
            model["sigT_Mpc"] = sigT_kms / M_of_zs[iz]
            kF_kms = P_model.get_kF_kms(z)
            kF_Mpc = kF_kms * M_of_zs[iz]
            # figure out type of pressure emu_param being used
            if "kF_Mpc" in emu_params:
                model["kF_Mpc"] = kF_Mpc
            elif "lambda_P" in emu_params:
                lamP_kpc = 1000 / kF_Mpc
                model["lambda_P"] = lamP_kpc
            if self.verbose:
                print(iz, z, "model", model)
            emu_calls.append(model)

        if return_M_of_z == True:
            if return_blob:
                return emu_calls, M_of_zs, blob
            else:
                return emu_calls, M_of_zs
        else:
            if return_blob:
                return emu_calls, blob
            else:
                return emu_calls

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
            params = self.cosmo_model_fid.get_linP_params()
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
                fid_As = self.cosmo_model_fid.cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.cosmo_model_fid.cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.cosmo_model_fid.cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale of primordial power
        ks_Mpc = self.cosmo_model_fid.cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.cosmo_model_fid.dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # get blob for fiducial cosmo
        fid_blob = self.get_blob(self.cosmo_model_fid)

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
        self, k_kms, like_params=[], return_covar=False, return_blob=False
    ):
        """Emulate P1D in velocity units, for all redshift bins,
        as a function of input likelihood parameters.
        It might also return a covariance from the emulator,
        or a blob with extra information for the emcee_sampler."""

        if self.emulator is None:
            raise ValueError("no emulator provided")

        # figure out emulator calls, one per redshift
        if return_blob:
            emu_calls, M_of_z, blob = self.get_emulator_calls(
                like_params=like_params, return_M_of_z=True, return_blob=True
            )
        else:
            emu_calls, M_of_z = self.get_emulator_calls(
                like_params=like_params, return_M_of_z=True, return_blob=False
            )

        # compute input k to emulator
        Nz = len(self.zs)
        logk_Mpc = np.zeros((Nz, len(k_kms)))
        for iz in range(Nz):
            logk_Mpc[iz] = np.log10(k_kms * M_of_z[iz])

        if return_covar:
            p1d_Mpc, cov_Mpc = self.call_emulator(
                emu_calls, logk_Mpc, return_covar=True, z=self.zs
            )
        else:
            p1d_Mpc = self.call_emulator(
                emu_calls, logk_Mpc, return_covar=False, z=self.zs
            )

        p1d_kms = []
        covars = []
        for iz in range(Nz):
            p1d_kms.append(p1d_Mpc[iz] * M_of_z[iz])
            if return_covar:
                if cov_Mpc is None:
                    covars.append(None)
                else:
                    covars.append(cov_Mpc[iz] * M_of_z[iz] ** 2)

        ind = np.argwhere((np.array(self.emulator.emu_params) == "mF"))[0, 0]
        # include multiplicate metal contamination
        for X_model_fid in self.metal_models:
            X_model = X_model_fid.get_new_model(like_params)
            for iz, z in enumerate(self.zs):
                mF = emu_calls[iz][ind]
                cont = X_model.get_contamination(z=z, k_kms=k_kms, mF=mF)
                p1d_kms[iz] *= cont

        # decide what to return, and return it
        if return_covar:
            if return_blob:
                return p1d_kms, covars, blob
            else:
                return p1d_kms, covars
        else:
            if return_blob:
                return p1d_kms, blob
            else:
                return p1d_kms

    def get_parameters(self):
        """Return parameters in models, even if not free parameters"""

        # get parameters from CAMB model
        params = self.cosmo_model_fid.get_likelihood_parameters()

        # get parameters from nuisance IGM models
        for par in self.F_model_fid.get_parameters():
            params.append(par)
        for par in self.T_model_fid.get_sigT_kms_parameters():
            params.append(par)
        for par in self.T_model_fid.get_gamma_parameters():
            params.append(par)
        for par in self.P_model_fid.get_parameters():
            params.append(par)

        # get parameters from metal contamination models
        for metal in self.metal_models:
            for par in metal.get_parameters():
                params.append(par)

        if self.verbose:
            print("got parameters")
            for par in params:
                print(par.info_str())

        return params

    def get_igm_models(self, like_params=[]):
        """Setup IGM models from input list of likelihood parameters"""

        F_model = self.F_model_fid.get_new_model(like_params)
        T_model = self.T_model_fid.get_new_model(like_params)
        P_model = self.P_model_fid.get_new_model(like_params)

        models = {"F_model": F_model, "T_model": T_model, "P_model": P_model}

        return models

    def update_igm_models(self, like_params=[]):
        """Setup IGM models from input list of likelihood parameters"""

        self.F_model_emcee.update_parameters(like_params)
        self.T_model_emcee.update_parameters(like_params)
        self.P_model_emcee.update_parameters(like_params)

        models = {
            "F_model": self.F_model_emcee,
            "T_model": self.T_model_emcee,
            "P_model": self.P_model_emcee,
        }

        return models

    def plot_p1d(self, k_kms, like_params=[], plot_every_iz=1):
        """Emulate and plot P1D in velocity units, for all redshift bins,
        as a function of input likelihood parameters"""

        # ask emulator prediction for P1D in each bin
        emu_p1d = self.get_p1d_kms(k_kms, like_params)

        # plot only few redshifts for clarity
        Nz = len(self.zs)
        for iz in range(0, Nz, plot_every_iz):
            # acess data for this redshift
            z = self.zs[iz]
            p1d = emu_p1d[iz]
            # plot everything
            col = plt.cm.jet(iz / (Nz - 1))
            plt.plot(k_kms, p1d * k_kms / np.pi, color=col, label="z=%.1f" % z)
        plt.yscale("log")
        plt.legend()
        plt.xlabel("k [s/km]")
        plt.ylabel(r"$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$")
        plt.ylim(0.005, 0.6)
        plt.show()

        return

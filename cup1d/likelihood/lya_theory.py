import numpy as np
import copy
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import gp_emulator

from cup1d.likelihood import CAMB_model
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_systematics import Systematics
from cup1d.likelihood.model_igm import IGM
from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.utils.utils_sims import get_training_hc
from cup1d.utils.hull import Hull
from cup1d.utils.utils import is_number_string


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        emulator=None,
        model_igm=None,
        model_cont=None,
        model_syst=None,
        use_hull=True,
        verbose=False,
        z_star=3.0,
        kp_kms=0.009,
        use_star_priors=None,
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
            - fid_cosmo: fiducial cosmology used for fixed parameters
            - fid_sim_igm: IGM model assumed
            - true_sim_igm: if not None, true IGM model of the mock
        """

        self.verbose = verbose

        # specify pivot point used in compressed parameters
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.use_hull = use_hull
        self.use_star_priors = use_star_priors

        # setup emulator
        if emulator is None:
            raise ValueError("Emulator not specified")
        else:
            self.emulator = emulator
        self.emu_kp_Mpc = self.emulator.kp_Mpc
        res = get_training_hc(self.emulator.list_sim_cube[0][:3])
        self.emu_pars = res[0]
        self.hc_points = res[1]
        self.emu_cosmo_all = res[2]
        self.emu_igm_all = res[3]

        # setup model_igm
        if model_igm is None:
            self.model_igm = IGM(zs)
        else:
            self.model_igm = model_igm

        # setup model_cont
        if model_cont is None:
            self.model_cont = Contaminants()
        else:
            self.model_cont = model_cont

        # setup model_syst
        if model_syst is None:
            self.model_syst = Systematics()
        else:
            self.model_syst = model_syst

    def set_fid_cosmo(
        self, zs, zs_hires=None, input_cosmo=None, extra_factor=1.15
    ):
        """Setup fiducial cosmology"""

        self.zs = zs
        self.zs_hires = zs_hires

        if self.use_hull:
            self.hull = Hull(
                zs=zs,
                data_hull=self.hc_points,
                suite=self.emulator.list_sim_cube[0][:3],
                extra_factor=extra_factor,
            )
            if zs_hires is not None:
                if len(zs) == len(zs_hires):
                    self.hull_hires = self.hull
                else:
                    self.hull_hires = Hull(
                        zs=zs_hires,
                        data_hull=self.hc_points,
                        suite=self.emulator.list_sim_cube[0][:3],
                        extra_factor=extra_factor,
                    )

        # setup fiducial cosmology (used for fitting)
        if input_cosmo is None:
            input_cosmo = camb_cosmo.get_cosmology()

        # setup CAMB object for the fiducial cosmology and precompute some things
        if self.zs_hires is not None:
            _zs = np.concatenate([self.zs, self.zs_hires, [self.z_star]])
        else:
            _zs = np.concatenate([self.zs, [self.z_star]])
        _zs = np.unique(_zs)

        self.fid_cosmo = {}
        self.fid_cosmo["zs"] = _zs
        self.fid_cosmo["cosmo"] = CAMB_model.CAMBModel(
            zs=_zs,
            cosmo=input_cosmo,
            z_star=self.z_star,
            kp_kms=self.kp_kms,
        )
        self.fid_cosmo["linP_Mpc_params"] = self.fid_cosmo[
            "cosmo"
        ].get_linP_Mpc_params(kp_Mpc=self.emu_kp_Mpc)
        self.fid_cosmo["M_of_zs"] = self.fid_cosmo["cosmo"].get_M_of_zs()
        self.fid_cosmo["linP_params"] = self.fid_cosmo[
            "cosmo"
        ].get_linP_params()

        # when using a fiducial cosmology, easy to change in other cases (TODO)
        self.set_cosmo_priors()

    def rescale_fid_cosmo(self, target_params):
        dkms_dMpc = self.fid_cosmo["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar
        pstar = self.fid_cosmo["cosmo"].get_linP_params()

        fid_Ap = pstar["Delta2_star"]
        ratio_Ap = target_params["Delta2_star"] / fid_Ap

        fid_np = pstar["n_star"]
        delta_np = target_params["n_star"] - fid_np

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # compute scalings
        delta_ns = delta_np
        ln_ratio_As = np.log(ratio_Ap) - delta_np * ln_kp_ks

        new_As = (
            np.exp(ln_ratio_As) * self.fid_cosmo["cosmo"].cosmo.InitPower.As
        )
        new_ns = delta_ns + self.fid_cosmo["cosmo"].cosmo.InitPower.ns
        rescaled_cosmo = camb_cosmo.get_cosmology(
            H0=self.fid_cosmo["cosmo"].cosmo.H0,
            mnu=self.fid_cosmo["cosmo"].cosmo.omnuh2 * 93.14,
            omch2=self.fid_cosmo["cosmo"].cosmo.omch2,
            ombh2=self.fid_cosmo["cosmo"].cosmo.ombh2,
            omk=self.fid_cosmo["cosmo"].cosmo.omk,
            As=new_As,
            ns=new_ns,
            nrun=self.fid_cosmo["cosmo"].cosmo.InitPower.nrun,
            pivot_scalar=ks_Mpc,
            w=self.fid_cosmo["cosmo"].cosmo.DarkEnergy.w,
            wa=self.fid_cosmo["cosmo"].cosmo.DarkEnergy.wa,
        )

        self.set_fid_cosmo(
            self.zs, zs_hires=self.zs_hires, input_cosmo=rescaled_cosmo
        )

    def set_cosmo_priors(self, extra_factor=1.25):
        """Set priors for cosmological parameters

        We get the priors on As, ns, and nrun from differences in star parameters in the training set
        Only works when using a fiducial cosmology
        """

        # pivot scale of primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.fid_cosmo["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        fid_As = self.fid_cosmo["cosmo"].cosmo.InitPower.As
        fid_ns = self.fid_cosmo["cosmo"].cosmo.InitPower.ns
        fid_nrun = self.fid_cosmo["cosmo"].cosmo.InitPower.nrun

        fid_Astar = self.fid_cosmo["linP_params"]["Delta2_star"]
        fid_nstar = self.fid_cosmo["linP_params"]["n_star"]
        fid_alphastar = self.fid_cosmo["linP_params"]["alpha_star"]

        if self.use_star_priors is not None:
            self.star_priors = {}
            for key in self.use_star_priors:
                self.star_priors[key] = self.use_star_priors[key]
        else:
            self.star_priors = None

        hc_fid = {}
        hc_fid["As"] = []
        hc_fid["ns"] = []
        hc_fid["nrun"] = []

        for key in self.emu_cosmo_all:
            cos = self.emu_cosmo_all[key]
            if is_number_string(cos["sim_label"][-1]) == False:
                continue
            test_Astar = cos["star_params"]["Delta2_star"]
            test_nstar = cos["star_params"]["n_star"]
            test_alphastar = cos["star_params"]["alpha_star"]

            ln_ratio_Astar = np.log(test_Astar / fid_Astar)
            delta_nstar = test_nstar - fid_nstar
            delta_alphastar = test_alphastar - fid_alphastar

            delta_nrun = delta_alphastar
            delta_ns = delta_nstar - delta_nrun * ln_kp_ks
            ln_ratio_As = (
                ln_ratio_Astar
                - (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
            )
            hc_fid["nrun"].append(fid_nrun + delta_nrun)
            hc_fid["ns"].append(fid_ns + delta_ns)
            hc_fid["As"].append(fid_As * np.exp(ln_ratio_As))

        hc_fid["As"] = np.array(hc_fid["As"])
        hc_fid["ns"] = np.array(hc_fid["ns"])
        hc_fid["nrun"] = np.array(hc_fid["nrun"])

        self.cosmo_priors = {
            "As": np.array([hc_fid["As"].min(), hc_fid["As"].max()]),
            "ns": np.array([hc_fid["ns"].min(), hc_fid["ns"].max()]),
            "nrun": np.array([hc_fid["nrun"].min(), hc_fid["nrun"].max()]),
        }

        for par in self.cosmo_priors:
            for ii in range(2):
                if (ii == 0) and (self.cosmo_priors[par][ii] < 0):
                    self.cosmo_priors[par][ii] *= extra_factor
                elif (ii == 0) and (self.cosmo_priors[par][ii] >= 0):
                    self.cosmo_priors[par][ii] *= 1 - (extra_factor - 1)
                elif (ii == 1) and (self.cosmo_priors[par][ii] < 0):
                    self.cosmo_priors[par][ii] *= 1 - (extra_factor - 1)
                elif (ii == 1) and (self.cosmo_priors[par][ii] >= 0):
                    self.cosmo_priors[par][ii] *= extra_factor

    def fixed_background(self, like_params):
        """Check if any of the input likelihood parameters would change
        the background expansion of the fiducial cosmology"""

        # look for parameters that would change background
        for par in like_params:
            if par.name in ["ombh2", "omch2", "H0", "mnu", "cosmomc_theta"]:
                return False

        return True

    def get_linP_Mpc_params_from_fiducial(
        self, zs, like_params, return_derivs=False
    ):
        """Recycle linP_Mpc_params from fiducial model, when only varying
        primordial power spectrum (As, ns, nrun)"""

        # make sure you are not changing the background expansion
        assert self.fixed_background(like_params)

        zs = np.atleast_1d(zs)

        # differences in primordial power (at CMB pivot point)
        ratio_As = 1.0
        delta_ns = 0.0
        delta_nrun = 0.0
        for par in like_params:
            if par.name == "As":
                fid_As = self.fid_cosmo["cosmo"].cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.fid_cosmo["cosmo"].cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.fid_cosmo["cosmo"].cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale in primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar
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
            _ = np.argwhere(self.fid_cosmo["zs"] == z)[0, 0]
            zlinP = self.fid_cosmo["linP_Mpc_params"][_]
            linP_Mpc_params.append(
                {
                    "Delta2_p": zlinP["Delta2_p"] * np.exp(ln_ratio_A_p),
                    "n_p": zlinP["n_p"] + delta_n_p,
                    "alpha_p": zlinP["alpha_p"] + delta_alpha_p,
                }
            )

        if return_derivs:
            val_derivs = {}
            _ = np.argwhere(self.fid_cosmo["zs"] == self.z_star)[0, 0]
            zlinP = self.fid_cosmo["linP_Mpc_params"][_]

            val_derivs["Delta2star"] = zlinP["Delta2_p"] * np.exp(ln_ratio_A_p)
            val_derivs["nstar"] = zlinP["n_p"] + delta_n_p
            val_derivs["alphastar"] = zlinP["alpha_p"] + delta_alpha_p

            val_derivs["der_alphastar_nrun"] = 1
            val_derivs["der_alphastar_ns"] = 0
            val_derivs["der_alphastar_As"] = 0

            val_derivs["der_nstar_nrun"] = ln_kp_ks
            val_derivs["der_nstar_ns"] = 1
            val_derivs["der_nstar_As"] = 0

            val_derivs["der_Delta2star_nrun"] = (
                0.5 * val_derivs["Delta2star"] * ln_kp_ks**2
            )
            val_derivs["der_Delta2star_ns"] = (
                val_derivs["Delta2star"] * ln_kp_ks
            )
            val_derivs["der_Delta2star_As"] = val_derivs["Delta2star"] / (
                ratio_As * fid_As
            )

            return linP_Mpc_params, val_derivs
        else:
            return linP_Mpc_params

    def get_err_linP_Mpc_params(self, like_params, covar):
        """Get error on linP_Mpc_params"""

        res = {}

        _, der = self.get_blob_fixed_background(like_params, return_derivs=True)

        err_As = covar[0, 0]
        err_ns = covar[1, 1]
        err_ns_As = covar[0, 1]
        if covar.shape[0] == 3:
            err_nrun = covar[2, 2]
            err_nrun_ns = covar[1, 2]
            err_nrun_As = covar[0, 2]
        else:
            err_nrun = 0
            err_nrun_ns = 0
            err_nrun_As = 0

        err_alphastar = (
            der["der_alphastar_nrun"] ** 2 * err_nrun
            + der["der_alphastar_ns"] ** 2 * err_ns
            + der["der_alphastar_As"] ** 2 * err_As
            + der["der_alphastar_nrun"] * der["der_alphastar_ns"] * err_nrun_ns
            + der["der_alphastar_nrun"] * der["der_alphastar_As"] * err_nrun_As
            + der["der_alphastar_ns"] * der["der_alphastar_As"] * err_ns_As
        )
        err_nstar = (
            der["der_nstar_nrun"] ** 2 * err_nrun
            + der["der_nstar_ns"] ** 2 * err_ns
            + der["der_nstar_As"] ** 2 * err_As
            + der["der_nstar_nrun"] * der["der_nstar_ns"] * err_nrun_ns
            + der["der_nstar_nrun"] * der["der_nstar_As"] * err_nrun_As
            + der["der_nstar_ns"] * der["der_nstar_As"] * err_ns_As
        )
        err_Delta2star = (
            der["der_Delta2star_nrun"] ** 2 * err_nrun
            + der["der_Delta2star_ns"] ** 2 * err_ns
            + der["der_Delta2star_As"] ** 2 * err_As
            + der["der_Delta2star_nrun"]
            * der["der_Delta2star_ns"]
            * err_nrun_ns
            + der["der_Delta2star_nrun"]
            * der["der_Delta2star_As"]
            * err_nrun_As
            + der["der_Delta2star_ns"] * der["der_Delta2star_As"] * err_ns_As
        )

        res["Delta2_star"] = der["Delta2star"]
        res["n_star"] = der["nstar"]
        res["alpha_star"] = der["alphastar"]
        res["err_Delta2_star"] = np.sqrt(err_Delta2star)
        res["err_n_star"] = np.sqrt(err_nstar)
        res["err_alpha_star"] = np.sqrt(err_alphastar)

        return res

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
                _ = np.argwhere(self.fid_cosmo["zs"] == z)[0, 0]
                M_of_zs.append(self.fid_cosmo["M_of_zs"][_])
            M_of_zs = np.array(M_of_zs)
            if return_blob:
                blob = self.get_blob_fixed_background(like_params)
        else:
            # setup a new CAMB_model from like_params
            if self.verbose:
                print("create new CAMB_model")
            camb_model = self.fid_cosmo["cosmo"].get_new_model(zs, like_params)
            linP_Mpc_params = camb_model.get_linP_Mpc_params(
                kp_Mpc=self.emu_kp_Mpc
            )
            M_of_zs = camb_model.get_M_of_zs()
            if return_blob:
                blob = self.get_blob(camb_model=camb_model)

        # store emulator calls
        emu_call = {}
        for key in self.emulator.emu_params:
            if (key == "Delta2_p") | (key == "n_p") | (key == "alpha_p"):
                emu_call[key] = np.zeros(len(zs))
                for ii in range(len(linP_Mpc_params)):
                    emu_call[key][ii] = linP_Mpc_params[ii][key]
            elif key == "mF":
                emu_call[key] = self.model_igm.models["F_model"].get_mean_flux(
                    zs, like_params=like_params
                )
                emu_call["mF_fid"] = self.model_igm.models[
                    "F_model"
                ].get_mean_flux(zs)
            elif key == "gamma":
                emu_call[key] = self.model_igm.models["T_model"].get_gamma(
                    zs, like_params=like_params
                )
            elif key == "sigT_Mpc":
                emu_call[key] = (
                    self.model_igm.models["T_model"].get_sigT_kms(
                        zs, like_params=like_params
                    )
                    / M_of_zs
                )
            elif key == "kF_Mpc":
                emu_call[key] = (
                    self.model_igm.models["P_model"].get_kF_kms(
                        zs, like_params=like_params
                    )
                    * M_of_zs
                )
            elif key == "lambda_P":
                emu_call[key] = 1000 / (
                    self.model_igm.models["P_model"].get_kF_kms(
                        zs, like_params=like_params
                    )
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
        by get_p1d_kms and used in the fitter."""

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
        """Return extra information (blob) for the fitter."""

        if camb_model is None:
            Nblob = len(self.get_blobs_dtype())
            if Nblob == 1:
                return np.nan
            else:
                out = np.nan, *([np.nan] * (Nblob - 1))
                return out
        else:
            # compute linear power parameters for input cosmology
            params = self.fid_cosmo["cosmo"].get_linP_params()
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
                fid_As = self.fid_cosmo["cosmo"].cosmo.InitPower.As
                ratio_As = par.value / fid_As
            if par.name == "ns":
                fid_ns = self.fid_cosmo["cosmo"].cosmo.InitPower.ns
                delta_ns = par.value - fid_ns
            if par.name == "nrun":
                fid_nrun = self.fid_cosmo["cosmo"].cosmo.InitPower.nrun
                delta_nrun = par.value - fid_nrun

        # pivot scale of primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.fid_cosmo["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # get blob for fiducial cosmo
        fid_blob = self.get_blob(self.fid_cosmo["cosmo"])

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

        linP_Mpc_params = (Delta2_star, n_star, alpha_star) + fid_blob[3:]

        return linP_Mpc_params

    def err_star(self, cov_As_ns, like_params):
        D2star = self.get_blob_fixed_background(like_params)[0]
        for par in like_params:
            if par.name == "As":
                As = par.value

        # pivot scale of primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].cosmo.InitPower.pivot_scalar

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.fid_cosmo["cosmo"].dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        dD2star_dAs = D2star / As
        dD2star_dns = D2star * np.log(kp_Mpc / ks_Mpc)
        err_D2star = np.sqrt(
            dD2star_dAs**2 * cov_As_ns[0, 0]
            + dD2star_dns**2 * cov_As_ns[1, 1]
            + 2 * dD2star_dAs * dD2star_dns * cov_As_ns[1, 0]
        )
        err_nstar = np.sqrt(cov_As_ns[1, 1])

        return err_D2star, err_nstar

    def get_p1d_kms(
        self,
        zs,
        k_kms,
        like_params=[],
        return_covar=False,
        return_blob=True,
        return_emu_params=False,
        apply_hull=True,
        hires=False,
        remove=None,
    ):
        """Emulate P1D in velocity units, for all redshift bins,
        as a function of input likelihood parameters.
        It might also return a covariance from the emulator,
        or a blob with extra information for the fitter."""

        zs = np.atleast_1d(zs)

        # figure out emulator calls
        emu_call, M_of_z, blob = self.get_emulator_calls(
            zs,
            like_params=like_params,
            return_M_of_z=True,
            return_blob=True,
        )

        # np.save("emu_call_fiducial.npy", emu_call)

        # also apply priors on compressed parameters
        # temporary hack
        dict_trans = {
            "Delta2_star": 0,
            "n_star": 1,
            "alpha_star": 2,
        }
        if self.star_priors is not None:
            for key in self.star_priors:
                _ = np.argwhere(
                    (blob[dict_trans[key]] > self.star_priors[key][1])
                    | (blob[dict_trans[key]] < self.star_priors[key][0])
                )
                if len(_) > 0:
                    return None

        # check priors
        if self.use_hull & apply_hull:
            if hires == False:
                hull = self.hull
            else:
                hull = self.hull_hires

            p0 = np.zeros((len(zs), len(hull.params)))
            for jj, key in enumerate(hull.params):
                p0[:, jj] = emu_call[key]
                # print(key, emu_call[key])

            if hull.in_hulls(p0) == False:
                # print("Not in hull")
                return None

        # compute input k to emulator in Mpc
        Nz = len(zs)
        length = 0
        if Nz > 1:
            for iz in range(Nz):
                if len(k_kms[iz]) > length:
                    length = len(k_kms[iz])
        else:
            if len(k_kms) == 1:
                k_kms = k_kms[0]
            length = len(k_kms)
            k_kms = [k_kms]

        kin_Mpc = np.zeros((Nz, length))
        for iz in range(Nz):
            kin_Mpc[iz, : len(k_kms[iz])] = k_kms[iz] * M_of_z[iz]

        # call emulator
        _res = self.emulator.emulate_p1d_Mpc(emu_call, kin_Mpc)
        # if return_covar:
        #     p1d_Mpc, cov_Mpc = _res
        # else:
        #     p1d_Mpc = _res
        p1d_Mpc = _res

        # move from Mpc to kms
        p1d_kms = []
        covars = []
        for iz in range(Nz):
            p1d_kms.append(p1d_Mpc[iz][: len(k_kms[iz])] * M_of_z[iz])
            # if return_covar:
            #     if cov_Mpc is None:
            #         covars.append(None)
            #     else:
            #         covars.append(
            #             cov_Mpc[iz][: len(k_kms[iz]), : len(k_kms[iz])]
            #             * M_of_z[iz] ** 2
            #         )

        # check if need to apply systematics
        apply_syst = False
        for par in like_params:
            if par.name.startswith("R_coeff"):
                apply_syst = True

        if apply_syst:
            syst_total = self.model_syst.get_contamination(
                zs, k_kms, like_params=like_params
            )
        else:
            syst_total = np.ones(len(zs))

        cont_all = self.model_cont.get_contamination(
            zs,
            k_kms,
            emu_call["mF"],
            M_of_z,
            like_params=like_params,
            remove=remove,
        )
        # print(
        #     "mult_cont_total",
        #     np.concatenate(cont_all["cont_mul_metals"]).min(),
        #     np.concatenate(cont_all["cont_mul_metals"]).max(),
        # )
        # print(
        #     "add_cont_total",
        #     np.concatenate(cont_all["cont_add_metals"]).min(),
        #     np.concatenate(cont_all["cont_add_metals"]).max(),
        # )
        # print(
        #     "HCD",
        #     np.concatenate(cont_all["cont_HCD"]).min(),
        #     np.concatenate(cont_all["cont_HCD"]).max(),
        # )
        # print(
        #     "syst_total",
        #     np.concatenate(syst_total).min(),
        #     np.concatenate(syst_total).max(),
        # )

        # if len(cont_all["cont_HCD"]) != len(z):
        # print(len(zs))
        # print(len(cont_all["cont_HCD"]))
        # print(len(cont_all["cont_mul_metals"]))
        # # print(len(cont_all["IC_corr"]))
        # # print(len(p1d_kms))
        # print(len(cont_all["cont_add_metals"]))
        # print(len(syst_total))
        p1d_cont_kms = []
        for iz, z in enumerate(zs):
            # Pcont = (mul_metal * HCD * IC_corr * Pemu + add_metal) * syst
            _p1d_cont_kms = (
                cont_all["cont_HCD"][iz]
                * cont_all["cont_mul_metals"][iz]
                * cont_all["IC_corr"][iz]
                * p1d_kms[iz]
                + cont_all["cont_add_metals"][iz]
            ) * syst_total[iz]

            # essentially the same results as using the previous expression
            # Pcont = (mul_metal * IC_corr * Pemu + add_metal) * HCD * syst
            # _p1d_cont_kms = (
            #     (
            #         cont_all["cont_mul_metals"][iz]
            #         * cont_all["IC_corr"][iz]
            #         * p1d_kms[iz]
            #         + cont_all["cont_add_metals"][iz]
            #     )
            #     * cont_all["cont_HCD"][iz]
            #     * syst_total[iz]
            # )
            p1d_cont_kms.append(_p1d_cont_kms)
        # print(p1d_cont_kms[0])

        # decide what to return, and return it
        out = [p1d_cont_kms]
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
        # TODO (can we set the priors only once?)
        params = self.fid_cosmo["cosmo"].get_likelihood_parameters(
            cosmo_priors=self.cosmo_priors
        )

        # get parameters from nuisance IGM models
        for model in self.model_igm.models:
            for par in self.model_igm.models[model].get_parameters():
                params.append(self.model_igm.models[model].get_parameter(par))

        # TODO: get parameters from all contaminants together

        # get parameters from metal contamination models
        for model_name in self.model_cont.metal_models:
            metal = self.model_cont.metal_models[model_name]
            for key in metal.params:
                params.append(metal.params[key])

        # get parameters from HCD contamination model
        for key in self.model_cont.hcd_model.params:
            params.append(self.model_cont.hcd_model.params[key])

        # # get parameters from SN contamination model
        # for par in self.model_cont.sn_model.get_parameters():
        #     params.append(par)

        # # get parameters from AGN contamination model
        # for par in self.model_cont.agn_model.get_parameters():
        #     params.append(par)

        # get parameters from systematic model
        for key in self.model_syst.resolution_model.params:
            params.append(self.model_syst.resolution_model.params[key])

        # for par in params:
        #     print(par.name)

        if self.verbose:
            print("got parameters")
            for par in params:
                print(par.info_str())

        return params

    def plot_p1d(
        self,
        k_kms,
        like_params=[],
        plot_every_iz=1,
        k_kms_hires=None,
        zmask=None,
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

            if emu_p1d is None:
                return "out of prior range"

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

        return

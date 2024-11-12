import numpy as np
import copy
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import gp_emulator

from cup1d.likelihood import CAMB_model
from cup1d.likelihood.model_contaminants import Contaminants
from cup1d.likelihood.model_igm import IGM
from cup1d.likelihood.cosmologies import set_cosmo


def set_theory(
    zs,
    emulator,
    free_parameters=None,
    set_metric=True,
    zs_hires=None,
    cosmo_label="Planck18",
    sim_igm="mpg_central",
    igm_priors="hc",
    SiIII=None,
    SiII=None,
    HCD=None,
    SN=None,
    AGN=None,
    ic_correction=None,
):
    """Set theory"""

    # set fiducial cosmology
    fid_cosmo = set_cosmo(cosmo_label=cosmo_label)

    # set igm model
    model_igm = IGM(
        zs,
        free_param_names=free_parameters,
        fid_sim_igm=sim_igm,
        list_sim_cube=emulator.list_sim_cube,
        type_priors=igm_priors,
        set_metric=set_metric,
    )

    # set contaminants
    model_cont = Contaminants(
        free_param_names=free_parameters,
        fid_SiIII=SiIII,
        fid_SiII=SiII,
        fid_HCD=HCD,
        fid_SN=SN,
        fid_AGN=AGN,
        ic_correction=ic_correction,
    )

    # set theory
    theory = Theory(
        zs=zs,
        zs_hires=zs_hires,
        emulator=emulator,
        fid_cosmo=fid_cosmo,
        model_igm=model_igm,
        model_cont=model_cont,
    )

    return theory


class Theory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation"""

    def __init__(
        self,
        zs,
        zs_hires=None,
        emulator=None,
        model_igm=None,
        model_cont=None,
        verbose=False,
        z_star=3.0,
        kp_kms=0.009,
        fid_cosmo=None,
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
        self.zs = zs
        self.zs_hires = zs_hires

        # specify pivot point used in compressed parameters
        self.z_star = z_star
        self.kp_kms = kp_kms

        # setup emulator
        if emulator is None:
            raise ValueError("Emulator not specified")
        else:
            self.emulator = emulator
        self.emu_kp_Mpc = self.emulator.kp_Mpc

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

        # setup fiducial cosmology (used for fitting)
        if not fid_cosmo:
            fid_cosmo = camb_cosmo.get_cosmology()

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
            cosmo=fid_cosmo,
            z_star=self.z_star,
            kp_kms=self.kp_kms,
        )
        self.cosmo_model_fid["linP_Mpc_params"] = self.cosmo_model_fid[
            "cosmo"
        ].get_linP_Mpc_params(kp_Mpc=self.emu_kp_Mpc)
        self.cosmo_model_fid["M_of_zs"] = self.cosmo_model_fid[
            "cosmo"
        ].get_M_of_zs()

    def emu_cosmo_hc(self):
        """Return cosmological parameters of simulations used for training"""

        # name of simulations used for training
        list_sim_hc = self.emulator.list_sim_cube
        if list_sim_hc[0][:3] == "mpg":
            get_cosmo = camb_cosmo.get_cosmology_from_dictionary
        elif list_sim_hc[0][:3] == "nyx":
            get_cosmo = camb_cosmo.get_Nyx_cosmology
        else:
            raise ValueError("Simulation not recognised")

        # load the cosmology of these
        cosmo_all = set_cosmo(cosmo_label=list_sim_hc[0], return_all=True)

        linP_hc = np.zeros((len(list_sim_hc), 3))
        for ii in range(len(list_sim_hc)):
            if cosmo_all[ii]["sim_label"] in list_sim_hc:
                cosmo = get_cosmo(cosmo_all[ii]["cosmo_params"])
                _ = CAMB_model.CAMBModel(
                    zs=[3.0],
                    cosmo=cosmo,
                    z_star=self.z_star,
                    kp_kms=self.kp_kms,
                )
                linparams = _.get_linP_params()
                linP_hc[ii, 0] = linparams["Delta2_star"]
                linP_hc[ii, 1] = linparams["n_star"]
                linP_hc[ii, 2] = linparams["alpha_star"]

        self.linP_hc = linP_hc

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

        if return_derivs:
            val_derivs = {}
            _ = np.argwhere(self.cosmo_model_fid["zs"] == self.z_star)[0, 0]
            zlinP = self.cosmo_model_fid["linP_Mpc_params"][_]

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

        # store emulator calls
        emu_call = {}
        for key in self.emulator.emu_params:
            if (key == "Delta2_p") | (key == "n_p") | (key == "alpha_p"):
                emu_call[key] = np.zeros(len(zs))
                for ii in range(len(linP_Mpc_params)):
                    emu_call[key][ii] = linP_Mpc_params[ii][key]
            elif key == "mF":
                emu_call[key] = self.model_igm.F_model.get_mean_flux(
                    zs, like_params=like_params
                )
            elif key == "gamma":
                emu_call[key] = self.model_igm.T_model.get_gamma(
                    zs, like_params=like_params
                )
            elif key == "sigT_Mpc":
                emu_call[key] = (
                    self.model_igm.T_model.get_sigT_kms(
                        zs, like_params=like_params
                    )
                    / M_of_zs
                )
            elif key == "kF_Mpc":
                emu_call[key] = (
                    self.model_igm.P_model.get_kF_kms(
                        zs, like_params=like_params
                    )
                    * M_of_zs
                )
            elif key == "lambda_P":
                emu_call[key] = 1000 / (
                    self.model_igm.P_model.get_kF_kms(
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
            params = self.cosmo_model_fid["cosmo"].get_linP_params()
            return (
                params["Delta2_star"],
                params["n_star"],
                params["alpha_star"],
                params["f_star"],
                params["g_star"],
                camb_model.cosmo.H0,
            )

    def get_blob_fixed_background(self, like_params, return_derivs=False):
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
        ### TODO: make this more efficient! Maybe directly storing the params?
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

        linP_Mpc_params = (Delta2_star, n_star, alpha_star) + fid_blob[3:]

        if return_derivs:
            val_derivs = {}

            val_derivs["Delta2star"] = Delta2_star
            val_derivs["nstar"] = n_star
            val_derivs["alphastar"] = alpha_star

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
        or a blob with extra information for the fitter."""

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

        # check prior here
        if self.model_igm.metric is not None:
            dist_priors = np.zeros((len(zs)))
            for ii in range(len(zs)):
                p0 = {}
                for key in emu_call:
                    p0[key] = emu_call[key][ii]
                dist_priors[ii] = self.model_igm.metric(p0)
            if dist_priors.max() > 1:
                # we are out of the prior range
                return None

        # compute input k to emulator in Mpc
        Nz = len(zs)
        length = 0
        for iz in range(Nz):
            if len(k_kms[iz]) > length:
                length = len(k_kms[iz])
        kin_Mpc = np.zeros((Nz, length))
        for iz in range(Nz):
            kin_Mpc[iz, : len(k_kms[iz])] = k_kms[iz] * M_of_z[iz]

        # call emulator
        _res = self.emulator.emulate_p1d_Mpc(
            emu_call, kin_Mpc, return_covar=return_covar, z=zs
        )
        if return_covar:
            p1d_Mpc, cov_Mpc = _res
        else:
            p1d_Mpc = _res

        # move from Mpc to kms
        p1d_kms = []
        covars = []
        for iz in range(Nz):
            try:
                p1d_kms.append(p1d_Mpc[iz][: len(k_kms[iz])] * M_of_z[iz])
            except:
                # needed for one redshift at a time
                p1d_kms.append(p1d_Mpc[: len(k_kms[iz])] * M_of_z[iz])
            if return_covar:
                if cov_Mpc is None:
                    covars.append(None)
                else:
                    covars.append(
                        cov_Mpc[iz][: len(k_kms[iz]), : len(k_kms[iz])]
                        * M_of_z[iz] ** 2
                    )

        # apply contaminants
        for iz, z in enumerate(zs):
            cont_total = self.model_cont.get_contamination(
                z,
                k_kms[iz],
                emu_call["mF"][iz],
                M_of_z[iz],
                like_params=like_params,
            )
            if cont_total is None:
                return None
            else:
                p1d_kms[iz] *= cont_total

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
        for par in self.model_igm.F_model.get_parameters():
            params.append(par)
        for par in self.model_igm.T_model.get_sigT_kms_parameters():
            params.append(par)
        for par in self.model_igm.T_model.get_gamma_parameters():
            params.append(par)
        for par in self.model_igm.P_model.get_parameters():
            params.append(par)

        # get parameters from metal contamination models
        for metal in self.model_cont.metal_models:
            for par in metal.get_X_parameters():
                params.append(par)
            for par in metal.get_D_parameters():
                params.append(par)

        # get parameters from HCD contamination model
        for par in self.model_cont.hcd_model.get_parameters():
            params.append(par)

        # get parameters from SN contamination model
        for par in self.model_cont.sn_model.get_parameters():
            params.append(par)

        # get parameters from AGN contamination model
        for par in self.model_cont.agn_model.get_parameters():
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

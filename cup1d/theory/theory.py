import numpy as np

from lace.cosmo import base_cosmology
from lace.cosmo import cosmology
from lace.cosmo import rescale_cosmology

from cup1d.likelihood.parameter import make_parameter
from cup1d.models.contaminants.model_contaminants import Contaminants
from cup1d.models.contaminants.model_systematics import Systematics
from cup1d.models.igm.model_igm import IGM
from cup1d.utils.utils_sims import get_training_hc
from cup1d.utils.hull import Hull
from cup1d.utils.utils import is_number_string


class Theory:
    r"""Translate likelihood parameters into emulator P1D predictions.

    Cosmological calculations are delegated to LaCE. The class only combines
    their linear-power quantities with the IGM, contamination, and
    instrumental-systematics models used by cup1d.
    """

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
        cosmo_priors=None,
    ):
        """Initialize the theory with an emulator and optional model objects."""

        self.verbose = verbose

        self.z_star = z_star
        self.kp_kms = kp_kms
        self.use_hull = use_hull
        self.use_star_priors = use_star_priors
        self.input_cosmo_priors = cosmo_priors

        if emulator is None:
            raise ValueError("Emulator not specified")
        self.emulator = emulator
        self.emu_kp_Mpc = self.emulator.kp_Mpc
        res = get_training_hc(self.emulator.list_sim_cube[0][:3])
        self.emu_pars = res[0]
        self.hc_points = res[1]
        self.emu_cosmo_all = res[2]
        self.emu_igm_all = res[3]

        if model_igm is None:
            self.model_igm = IGM(zs)
        else:
            self.model_igm = model_igm

        if model_cont is None:
            self.model_cont = Contaminants()
        else:
            self.model_cont = model_cont

        if model_syst is None:
            self.model_syst = Systematics()
        else:
            self.model_syst = model_syst

    def set_fid_cosmo(self, zs, cosmo_label=None, cosmo_params_dict=None):
        """Set the fiducial LaCE cosmology and precompute its quantities."""

        zs = np.unique(np.concatenate([np.atleast_1d(zs), [self.z_star]]))
        cosmo = cosmology.Cosmology(
            cosmo_label=cosmo_label, cosmo_params_dict=cosmo_params_dict
        )
        self.fid_cosmo = {
            "zs": zs,
            "cosmo": cosmo,
            "linP_Mpc_params": [
                cosmo.get_linP_Mpc_params(z, self.emu_kp_Mpc) for z in zs
            ],
            "M_of_zs": cosmo.get_dkms_dMpc(zs),
            "linP_params": cosmo.get_linP_kms_params(self.z_star, self.kp_kms),
        }
        if self.use_hull:
            self.hull = Hull(
                zs=zs,
                data_hull=self.hc_points,
                suite=self.emulator.list_sim_cube[0][:3],
                extra_factor=1.15,
            )
            self.hull_hires = self.hull
        self.set_cosmo_priors()

    def set_cosmo_priors(self, extra_factor=1.25):
        """Resolve cosmological prior limits for the fiducial cosmology.

        Primordial limits are inferred from the emulator training set unless
        explicitly supplied through ``Args.cosmo_priors``. Background limits
        always come from that Args mapping.
        """

        # pivot scale of primordial power
        ks_Mpc = self.fid_cosmo["cosmo"].ks_Mpc

        # likelihood pivot point, in velocity units
        dkms_dMpc = self.fid_cosmo["cosmo"].get_dkms_dMpc(self.z_star)
        kp_Mpc = self.kp_kms * dkms_dMpc

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        primordial = self.fid_cosmo["cosmo"].get_primordial_params()
        fid_As = primordial["As"]
        fid_ns = primordial["ns"]
        fid_nrun = primordial["nrun"]

        fid_Astar = self.fid_cosmo["linP_params"]["Delta2_star"]
        fid_nstar = self.fid_cosmo["linP_params"]["n_star"]
        fid_alphastar = self.fid_cosmo["linP_params"]["alpha_star"]

        if self.use_star_priors is None:
            self.star_priors = None
        else:
            self.star_priors = {}
            for key in self.use_star_priors:
                self.star_priors[key] = self.use_star_priors[key]

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
                ln_ratio_Astar - (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
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

        if self.input_cosmo_priors is not None:
            for name, bounds in self.input_cosmo_priors.items():
                if bounds is not None:
                    self.cosmo_priors[name] = np.asarray(bounds, dtype=float)

    def get_cosmology(self, like_params=None):
        """Return the LaCE cosmology corresponding to likelihood parameters.

        ``RescaledCosmology`` validates whether the requested parameters
        preserve the background. If they do not, construct a full cosmology
        and let LaCE obtain a new CAMB result.
        """

        fiducial_cosmo = self.fid_cosmo["cosmo"]
        like_params = {} if like_params is None else like_params
        new_params_dict = {
            name: value
            for name, value in like_params.items()
            if name in fiducial_cosmo.input_cosmo_params_dict
        }
        try:
            return rescale_cosmology.RescaledCosmology(fiducial_cosmo, new_params_dict)
        except rescale_cosmology.IncompatibleBackgroundError:
            pass

        cosmo_params_dict = fiducial_cosmo.input_cosmo_params_dict.copy()
        cosmo_params_dict.update(new_params_dict)
        return cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)

    def get_linP_Mpc_params(self, zs, like_params=None):
        """Get emulator linear-power parameters directly from LaCE."""

        cosmo = self.get_cosmology(like_params)
        return [
            cosmo.get_linP_Mpc_params(z, self.emu_kp_Mpc) for z in np.atleast_1d(zs)
        ]

    @staticmethod
    def _is_columnar_parameter_mapping(like_params):
        """Return whether a parameter mapping carries a leading batch axis."""

        if not isinstance(like_params, dict) or not like_params:
            return False
        dimensions = {np.asarray(value).ndim for value in like_params.values()}
        if dimensions == {0}:
            return False
        if dimensions != {1}:
            raise ValueError(
                "likelihood parameters must be all scalars or all "
                "one-dimensional batch arrays"
            )
        lengths = {len(np.asarray(value)) for value in like_params.values()}
        if len(lengths) != 1:
            raise ValueError("all batched likelihood parameters must share n_batch")
        return True

    def get_emulator_calls(
        self, zs, like_params=None, return_M_of_z=True, return_blob=False
    ):
        """Build emulator inputs for scalar or columnar parameter mappings.

        Scalar values preserve the historical return shapes.  A mapping whose
        values all have shape ``(n_batch,)`` is dispatched to
        :meth:`get_emulator_calls_batch`, returning inputs and conversions with
        leading ``(n_batch, n_z)`` axes.
        """

        if self._is_columnar_parameter_mapping(like_params):
            emu_call, M_of_z, blobs = self.get_emulator_calls_batch(zs, like_params)
            if return_M_of_z:
                return (emu_call, M_of_z, blobs) if return_blob else (emu_call, M_of_z)
            return (emu_call, blobs) if return_blob else emu_call

        # LaCE handles both transfer-function rescaling and the new-CAMB case.
        cosmo = self.get_cosmology(like_params)
        linP_Mpc_params = [cosmo.get_linP_Mpc_params(z, self.emu_kp_Mpc) for z in zs]
        M_of_zs = cosmo.get_dkms_dMpc(zs)
        if return_blob:
            blob = self.get_blob(cosmo)

        emu_call = {}
        for key in self.emulator.emu_params:
            if key in {"Delta2_p", "n_p", "alpha_p"}:
                emu_call[key] = np.zeros(len(zs))
                for ii in range(len(linP_Mpc_params)):
                    emu_call[key][ii] = linP_Mpc_params[ii][key]
            elif key == "mF":
                emu_call[key] = self.model_igm.models["F_model"].get_mean_flux(
                    zs, like_params=like_params
                )
                emu_call["mF_fid"] = self.model_igm.models["F_model"].get_mean_flux(zs)
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
                raise ValueError("Not a theory model for emulator parameter", key)

        if return_M_of_z:
            if return_blob:
                return emu_call, M_of_zs, blob
            return emu_call, M_of_zs
        if return_blob:
            return emu_call, blob
        return emu_call

    def get_emulator_calls_batch(self, zs, like_params):
        """Build batched emulator inputs with shape ``(n_batch, n_z)``.

        The background/linear-power rescaling is intentionally evaluated once
        per point: cup1d analyses use LaCE's inexpensive
        ``RescaledCosmology`` path after setup, not repeated CAMB calls.  IGM
        histories are evaluated columnarly across the full batch.
        """

        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        if not isinstance(like_params, dict) or not like_params:
            raise ValueError("like_params must be a non-empty columnar mapping")
        n_batch = None
        for name, values in like_params.items():
            values = np.asarray(values, dtype=float)
            if values.ndim != 1:
                raise ValueError(
                    f"batched parameter {name} must have shape (n_batch,), got {values.shape}"
                )
            if n_batch is None:
                n_batch = len(values)
            elif len(values) != n_batch:
                raise ValueError(f"batched parameter {name} has inconsistent length")

        cosmologies = [
            self.get_cosmology({name: values[index] for name, values in like_params.items()})
            for index in range(n_batch)
        ]
        linear = base_cosmology.BaseCosmology.get_linP_Mpc_params_for_cosmologies(
            cosmologies, zs, self.emu_kp_Mpc
        )
        M_of_z = base_cosmology.BaseCosmology.get_dkms_dMpc_for_cosmologies(
            cosmologies, zs
        )
        emu_call = {}
        for key in self.emulator.emu_params:
            if key in {"Delta2_p", "n_p", "alpha_p"}:
                emu_call[key] = linear[key]
            elif key == "mF":
                tau = self.model_igm.models["F_model"].get_value_batch(
                    "tau_eff", zs, like_params
                )
                tau *= self.model_igm.models["F_model"].fid_interp["tau_eff"](zs)[None, :]
                emu_call[key] = np.exp(-tau)
                emu_call["mF_fid"] = self.model_igm.models["F_model"].get_mean_flux(zs)
            elif key == "gamma":
                gamma = self.model_igm.models["T_model"].get_value_batch(
                    "gamma", zs, like_params
                )
                emu_call[key] = gamma * self.model_igm.models["T_model"].fid_interp["gamma"](zs)[None, :]
            elif key == "sigT_Mpc":
                sigT = self.model_igm.models["T_model"].get_value_batch(
                    "sigT_kms", zs, like_params
                )
                sigT *= self.model_igm.models["T_model"].fid_interp["sigT_kms"](zs)[None, :]
                emu_call[key] = sigT / M_of_z
            elif key in {"kF_Mpc", "lambda_P"}:
                kF = self.model_igm.models["P_model"].get_value_batch(
                    "kF_kms", zs, like_params
                )
                kF *= self.model_igm.models["P_model"].fid_interp["kF_kms"](zs)[None, :]
                if key == "kF_Mpc":
                    emu_call[key] = kF * M_of_z
                else:
                    emu_call[key] = 1000 / (kF * M_of_z)
            else:
                raise ValueError("Not a theory model for emulator parameter", key)
        blobs = np.asarray([self.get_blob(cosmo) for cosmo in cosmologies])
        return emu_call, M_of_z, blobs

    def get_blobs_dtype(self):
        """Return the dtype of the cosmological summary returned by the fitter."""

        return [
            ("Delta2_star", float),
            ("n_star", float),
            ("alpha_star", float),
            ("f_star", float),
            ("g_star", float),
            ("H0", float),
        ]

    def get_blob(self, cosmo=None):
        """Return extra information (blob) for the fitter."""

        if cosmo is None:
            number_of_blobs = len(self.get_blobs_dtype())
            if number_of_blobs == 1:
                return np.nan
            return np.nan, *([np.nan] * (number_of_blobs - 1))

        params = cosmo.get_linP_kms_params(self.z_star, self.kp_kms)
        dz = self.z_star / 100.0
        hubble_minus = cosmo.compute_hubble_parameter(self.z_star - dz)
        hubble_plus = cosmo.compute_hubble_parameter(self.z_star + dz)
        hubble_star = cosmo.compute_hubble_parameter(self.z_star)
        g_star = (
            (hubble_plus - hubble_minus)
            / (2 * dz)
            / hubble_star
            * (1 + self.z_star)
            * 2
            / 3
        )
        return (
            params["Delta2_star"],
            params["n_star"],
            params["alpha_star"],
            cosmo.get_growth_rate(self.z_star),
            g_star,
            cosmo.get_H0(),
        )

    def get_blob_for_parameters(self, like_params):
        """Return a blob for likelihood parameters via the LaCE flow."""

        return self.get_blob(self.get_cosmology(like_params))

    def get_P1D_kms(
        self,
        zs,
        k_ikms,
        like_params=None,
        return_covar=False,
        return_blob=True,
        return_emu_params=False,
        apply_hull=True,
        hires=False,
        remove=None,
        return_contaminants=False,
    ):
        """Emulate P1D in km/s with explicitly named inverse-km/s input."""
        return self.get_p1d_kms(
            zs,
            k_ikms,
            like_params=like_params,
            return_covar=return_covar,
            return_blob=return_blob,
            return_emu_params=return_emu_params,
            apply_hull=apply_hull,
            hires=hires,
            remove=remove,
            return_contaminants=return_contaminants,
        )

    def get_p1d_kms(self, zs, k_kms, like_params=None, **kwargs):
        """Return scalar or batched P1D according to parameter-array shape.

        A scalar parameter mapping follows the historical API. A columnar
        mapping with values shaped ``(n_batch,)`` returns a list over redshift
        whose items have shape ``(n_batch, n_k_z)``.
        """
        if self._is_columnar_parameter_mapping(like_params):
            unsupported = set(kwargs) - {"remove"}
            if unsupported:
                raise ValueError(f"batched P1D does not support {sorted(unsupported)}")
            return self._get_p1d_kms_batch(zs, k_kms, like_params, **kwargs)
        return self._get_p1d_kms_scalar(zs, k_kms, like_params=like_params, **kwargs)

    def _get_p1d_kms_scalar(
        self,
        zs,
        k_kms,
        like_params=None,
        return_covar=False,
        return_blob=True,
        return_emu_params=False,
        apply_hull=True,
        hires=False,
        remove=None,
        return_contaminants=False,
    ):
        """Emulate the P1D in velocity units for the requested redshifts."""

        zs = np.atleast_1d(zs)
        like_params = {} if like_params is None else like_params

        emu_call, M_of_z, blob = self.get_emulator_calls(
            zs,
            like_params=like_params,
            return_M_of_z=True,
            return_blob=True,
        )

        blob_index = {
            "Delta2_star": 0,
            "n_star": 1,
            "alpha_star": 2,
        }
        if self.star_priors is not None:
            for key in self.star_priors:
                _ = np.argwhere(
                    (blob[blob_index[key]] > self.star_priors[key][1])
                    | (blob[blob_index[key]] < self.star_priors[key][0])
                )
                if len(_) > 0:
                    return None

        if self.use_hull and apply_hull:
            hull = self.hull_hires if hires else self.hull

            p0 = np.zeros((len(zs), len(hull.params)))
            for jj, key in enumerate(hull.params):
                p0[:, jj] = emu_call[key]
            if not hull.in_hulls(p0):
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

        if "forest" in self.emulator.emulator_label:
            new_cosmo_params = {}
            for name, value in like_params.items():
                if name in ["As", "ns", "nrun"]:
                    new_cosmo_params[name] = value
            self.emulator.set_linear_theory(zs, new_cosmo_params=new_cosmo_params)
            _res = self.emulator.emulate_P1D_Mpc(zs, kin_Mpc, emu_call)
        else:
            _res = self.emulator.emulate_p1d_Mpc(emu_call, kin_Mpc)
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
        for name in like_params:
            if name.startswith("R_coeff"):
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

        terms = []
        p1d_cont_kms = []
        for iz, z in enumerate(zs):
            terms.append(
                {
                    "z": z,
                    "k_kms": k_kms[iz],
                    "p1d_emu_kms": p1d_kms[iz],
                    "C_res": syst_total[iz],
                    "C_mul_metals": cont_all["cont_mul_metals"][iz],
                    "C_add_metals": cont_all["cont_add_metals"][iz],
                    "C_HCD": cont_all["cont_HCD"][iz],
                    "p1d_tot_kms": "[(C_mul_metals * C_HCD * p1d_emu_kms + C_add_metals) * C_res]",
                }
            )
            _p1d_cont_kms = (
                cont_all["cont_HCD"][iz]
                * cont_all["cont_mul_metals"][iz]
                * cont_all["IC_corr"][iz]
                * p1d_kms[iz]
                + cont_all["cont_add_metals"][iz]
            ) * syst_total[iz]

            p1d_cont_kms.append(_p1d_cont_kms)

        out = [p1d_cont_kms]
        if return_covar:
            out.append(covars)
        if return_blob:
            out.append(blob)
        if return_emu_params:
            out.append(emu_call)
        if return_contaminants:
            out.append(terms)

        return out[0] if len(out) == 1 else out

    def _get_p1d_kms_batch(self, zs, k_kms, like_params, remove=None):
        """Evaluate LaCE P1D for a columnar batch, returning ``[(batch, k_z)]``.

        This path flattens batch and redshift for the GP emulator; ragged data
        grids remain a list over redshift. ForestFlow keeps its dedicated
        latent-index batch path until its linear-theory wrapper accepts a
        cosmology batch.
        """
        forest = "forest" in self.emulator.emulator_label
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        emu_call, M_of_z, _ = self.get_emulator_calls(zs, like_params, return_M_of_z=True, return_blob=True)
        n_batch, n_z = M_of_z.shape
        n_k = max(len(values) for values in k_kms)
        kin = np.zeros((n_batch, n_z, n_k))
        for iz, values in enumerate(k_kms):
            kin[:, iz, :len(values)] = np.asarray(values)[None, :] * M_of_z[:, iz, None]
        flat_call = {name: np.asarray(values).reshape(-1) for name, values in emu_call.items() if name in self.emulator.emu_params}
        if forest:
            cosmology_parameters = [
                {name: np.asarray(values)[ib] for name, values in like_params.items() if name in {"As", "ns", "nrun"}}
                for ib in range(n_batch)
            ]
            p_mpc = self.emulator.emulate_p1d_Mpc_batch(zs, kin, emu_call, cosmology_parameters)
        else:
            p_mpc = self.emulator.emulate_p1d_Mpc(flat_call, kin.reshape(n_batch*n_z, n_k)).reshape(n_batch, n_z, n_k)
        p_kms = [p_mpc[:, iz, :len(k_kms[iz])] * M_of_z[:, iz, None] for iz in range(n_z)]
        cont = self.model_cont.get_contamination(zs, k_kms, emu_call["mF"], M_of_z, like_params, remove=remove)
        if any(name.startswith("R_coeff") for name in like_params):
            syst = self.model_syst.get_contamination(zs, k_kms, like_params)
        else:
            syst = [np.ones_like(item) for item in p_kms]
        return [(cont["cont_HCD"][iz] * cont["cont_mul_metals"][iz] * cont["IC_corr"][iz] * p_kms[iz] + cont["cont_add_metals"][iz]) * syst[iz] for iz in range(n_z)]

    def get_parameters(self):
        """Return all likelihood parameters, including fixed parameters."""

        # LaCE provides the fiducial values; Args provides the prior limits.
        cosmology = self.fid_cosmo["cosmo"]
        background = cosmology.get_background_params()
        primordial = cosmology.get_primordial_params()
        params = [
            make_parameter(
                "ombh2", *self.cosmo_priors["ombh2"], background["ombh2"]
            ),
            make_parameter(
                "omch2", *self.cosmo_priors["omch2"], background["omch2"]
            ),
            make_parameter(
                "As",
                self.cosmo_priors["As"][0],
                self.cosmo_priors["As"][1],
                primordial["As"],
            ),
            make_parameter(
                "ns",
                self.cosmo_priors["ns"][0],
                self.cosmo_priors["ns"][1],
                primordial["ns"],
            ),
            make_parameter(
                "mnu",
                *self.cosmo_priors["mnu"],
                self.fid_cosmo["cosmo"].background_params["mnu"],
            ),
            make_parameter(
                "nrun",
                self.cosmo_priors["nrun"][0],
                self.cosmo_priors["nrun"][1],
                primordial["nrun"],
            ),
            make_parameter("H0", *self.cosmo_priors["H0"], cosmology.get_H0()),
        ]

        for model in self.model_igm.models:
            for par in self.model_igm.models[model].get_parameters():
                params.append(self.model_igm.models[model].get_parameter(par))

        for model_name in self.model_cont.metal_models:
            metal = self.model_cont.metal_models[model_name]
            for key in metal.params:
                params.append(metal.params[key])

        for key in self.model_cont.hcd_model.params:
            params.append(self.model_cont.hcd_model.params[key])

        for key in self.model_syst.resolution_model.params:
            params.append(self.model_syst.resolution_model.params[key])

        parameters = {parameter["name"]: parameter for parameter in params}
        if len(parameters) != len(params):
            raise ValueError("Theory parameter names must be unique")

        if self.verbose:
            print("got parameters")
            for parameter in parameters.values():
                print(
                    f"{parameter['name']} = {parameter['value']}"
                )

        return parameters

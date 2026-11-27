from collections.abc import Mapping

import numpy as np
from lace.cosmo import base_cosmology
from lace.cosmo.parameters import COSMOLOGY_PARAMETERS, THETA_PARAMETERS
from cup1d.utils.warn_once import warn_once
from lace.cosmo import cosmology
from lace.cosmo import rescale_cosmology

from cup1d.likelihood.parameter import make_parameter
from cup1d.emulator.hull import Hull
from cup1d.utils.utils import is_number_string
from cup1d.emulator.utils_sims import get_training_hc
from cup1d.utils.batch import as_columns, as_redshift_rows, row
from cup1d.emulator.factory import as_emulator
from cup1d.theory.emulator_inputs import emulator_inputs
from cup1d.models.observation import apply_observation_model
from cup1d.theory.projection import interpolate_p1d, interpolate_signed





# half width of the nrun prior around the fiducial (not set by the training set)
NRUN_PRIOR_HALF_WIDTH = 0.05

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
        hull_margin=0.05,
        verbose=False,
        z_star=3.0,
        kp_kms=0.009,
        use_star_priors=None,
        cosmo_priors=None,
        cosmo_prior_nsigma=None,
    ):
        """Initialize theory components and emulator training metadata.

        Parameters
        ----------
        emulator : object
            Required P1D emulator exposing its parameter list and simulation
            training cube.
        model_igm, model_cont, model_syst : object
            IGM, contaminant, and systematic models (required).
        use_hull : bool, default: True
            Enable checks against the emulator's admitted training domain.
        hull_margin : float, default: 0.05
            Fractional expansion of the training hull about the mean of the
            training points (0 is the strict hull), in every 2D projection.
        verbose : bool, default: False
            Retained compatibility verbosity flag.
        z_star : float, default: 3.0
            Redshift of compressed linear-power quantities.
        kp_kms : float, default: 0.009
            Compressed-power pivot in ``s / km``.
        use_star_priors : mapping, optional
            Optional bounds on compressed linear-power parameters.
        cosmo_priors : mapping, optional
            Cosmological prior bounds overriding training-derived limits.
        cosmo_prior_nsigma : float, optional
            If given, the training-derived ``As`` and ``ns`` bounds are the
            mean +- this many standard deviations of the training
            simulations, instead of their range expanded by 25%.

        Raises
        ------
        ValueError
            If no emulator is supplied.
        """

        self.verbose = verbose

        self.z_star = z_star
        self.kp_kms = kp_kms
        self.use_hull = use_hull
        self.hull_margin = float(hull_margin)
        self.use_star_priors = use_star_priors
        self.input_cosmo_priors = cosmo_priors
        self.cosmo_prior_nsigma = cosmo_prior_nsigma

        if emulator is None:
            raise ValueError("Emulator not specified")
        # one interface to LaCE and ForestFlow (cup1d.emulator.factory)
        self.emulator = as_emulator(emulator)
        self.emu_kp_Mpc = self.emulator.kp_Mpc
        res = get_training_hc(self.emulator.suite)
        self.emu_pars = res[0]
        self.hc_points = res[1]
        self.emu_cosmo_all = res[2]
        self.emu_igm_all = res[3]

        if model_igm is None or model_cont is None or model_syst is None:
            raise ValueError(
                "Theory needs model_igm, model_cont and model_syst "
                "(cup1d.theory.factory.set_theory builds them from the configuration)"
            )
        self.model_igm = model_igm
        self.model_cont = model_cont
        self.model_syst = model_syst

    def set_fid_cosmo(self, zs, cosmo_label=None, cosmo_params_dict=None):
        """Set the fiducial cosmology and precompute redshift-dependent values.

        Parameters
        ----------
        zs : array_like
            Requested redshifts; ``z_star`` is added automatically.
        cosmo_label : str, optional
            LaCE cosmology label.
        cosmo_params_dict : mapping, optional
            Explicit cosmological parameters, overriding a label as defined by
            :class:`lace.cosmo.cosmology.Cosmology`. Without a label or
            parameters, an existing fiducial cosmology is kept (only the
            redshift-dependent values are recomputed).
        """

        zs = np.unique(np.concatenate([np.atleast_1d(zs), [self.z_star]]))
        if cosmo_label is None and cosmo_params_dict is None and getattr(self, "fid_cosmo", None):
            # new redshifts, same cosmology (e.g. a mock keeps its own truth)
            cosmo = self.fid_cosmo["cosmo"]
        else:
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
                suite=self.emulator.suite,
                extra_factor=1.0 + self.hull_margin,
            )
            self.hull_hires = self.hull
        self.set_cosmo_priors()

    @staticmethod
    def _parameter_values(like_params):
        """Unwrap a public parameter point into values used by theory models.

        Parameters
        ----------
        like_params : mapping, optional
            Named numerical values or public parameter dictionaries.

        Returns
        -------
        dict or None
            Values-only parameter mapping, preserving scalar or columnar
            arrays; ``None`` remains ``None``.

        Raises
        ------
        TypeError
            If a non-mapping, non-``None`` point is supplied.

        Notes
        -----
        Public cup1d calls accept a named parameter point, where each entry
        may include ``value``, prior limits, and other metadata. Theory and
        its component models only need the numerical values. Values-only
        scalar mappings and columnar batch mappings remain supported for
        internal/vectorized use.
        """

        if like_params is None:
            return None
        if not isinstance(like_params, Mapping):
            raise TypeError("like_params must be a mapping of parameter names")
        return {
            name: value["value"]
            if isinstance(value, Mapping) and "value" in value
            else value
            for name, value in like_params.items()
        }

    def set_cosmo_priors(self, extra_factor=1.25):
        """Resolve cosmological prior limits for the fiducial cosmology.

        Parameters
        ----------
        extra_factor : float, default: 1.25
            Factor widening the training-derived ``As`` and ``ns`` ranges
            about their centres, before explicit configuration overrides.

        Notes
        -----
        Primordial limits are inferred from the emulator training set unless
        explicitly supplied through ``Args.cosmo_priors``. Background limits
        always come from that Args mapping. With ``cosmo_prior_nsigma`` set,
        the ``As`` and ``ns`` limits are the training mean +- that many
        standard deviations. ``nrun`` (not varied in the training suites)
        gets ``fiducial +- NRUN_PRIOR_HALF_WIDTH``.
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

        # widen each training range about its centre
        for par in self.cosmo_priors:
            low, high = self.cosmo_priors[par]
            centre, half_width = 0.5 * (low + high), 0.5 * (high - low)
            self.cosmo_priors[par] = np.array(
                [centre - extra_factor * half_width, centre + extra_factor * half_width]
            )

        # the training suites do not vary the running: the spread of their
        # alpha_star is a transfer-function offset, not primordial running,
        # and excludes nrun = 0. A broad range around the fiducial instead
        # (it only matters when nrun is free).
        self.cosmo_priors["nrun"] = np.array(
            [fid_nrun - NRUN_PRIOR_HALF_WIDTH, fid_nrun + NRUN_PRIOR_HALF_WIDTH]
        )

        if self.cosmo_prior_nsigma is not None:
            # keep the emulator close to its training domain: n_p depends on
            # ns and Delta2_p on As and ns, so wide primordial priors drive
            # the emulator into extrapolation
            for par in ("As", "ns"):
                mean, std = hc_fid[par].mean(), hc_fid[par].std()
                self.cosmo_priors[par] = np.array(
                    [
                        mean - self.cosmo_prior_nsigma * std,
                        mean + self.cosmo_prior_nsigma * std,
                    ]
                )

        if self.input_cosmo_priors is not None:
            for name, bounds in self.input_cosmo_priors.items():
                if bounds is not None:
                    self.cosmo_priors[name] = np.asarray(bounds, dtype=float)

    def get_cosmology(self, like_params=None):
        """Return the LaCE cosmology corresponding to likelihood parameters.

        Parameters
        ----------
        like_params : mapping, optional
            Public or values-only cosmological parameter point.

        Returns
        -------
        lace.cosmo.base_cosmology.BaseCosmology
            Rescaled fiducial cosmology when the background is compatible, or
            a freshly constructed full LaCE cosmology otherwise.

        Notes
        -----
        ``RescaledCosmology`` validates whether the requested parameters
        preserve the background. If they do not, construct a full cosmology
        and let LaCE obtain a new CAMB result.
        """

        fiducial_cosmo = self.fid_cosmo["cosmo"]
        like_params = self._parameter_values(like_params)
        like_params = {} if like_params is None else like_params
        # every cosmological name (also those the fiducial left at CAMB's
        # defaults, e.g. nnu); the other parameters (IGM, nuisance) are not
        new_params_dict = {
            name: value
            for name, value in like_params.items()
            if name in COSMOLOGY_PARAMETERS
        }
        try:
            return rescale_cosmology.RescaledCosmology(fiducial_cosmo, new_params_dict)
        except rescale_cosmology.IncompatibleBackgroundError:
            pass

        cosmo_params_dict = fiducial_cosmo.input_cosmo_params_dict.copy()
        if any(name in THETA_PARAMETERS for name in new_params_dict):
            # theta replaces H0 (and any other angular-size parameter)
            for name in ("H0",) + THETA_PARAMETERS:
                cosmo_params_dict.pop(name, None)
        cosmo_params_dict.update(new_params_dict)
        return cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)

    def get_linP_Mpc_params(self, zs, like_params=None):
        """Get emulator linear-power parameters directly from LaCE.

        Parameters
        ----------
        zs : array_like
            Redshifts at which to evaluate compressed linear power.
        like_params : mapping, optional
            Cosmological parameter point.

        Returns
        -------
        list of dict
            Compressed linear-power quantities in redshift order.
        """

        cosmo = self.get_cosmology(like_params)
        return [
            cosmo.get_linP_Mpc_params(z, self.emu_kp_Mpc) for z in np.atleast_1d(zs)
        ]

    def get_emulator_calls(
        self, zs, like_params=None, return_M_of_z=True, return_blob=False
    ):
        """Emulator inputs for one parameter point or a batch.

        Parameters
        ----------
        zs : array_like
            Evaluation redshifts.
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.
        return_M_of_z : bool, default: True
            Include the ``dkms/dMpc`` conversions.
        return_blob : bool, default: False
            Include the compressed cosmology values (:attr:`blob_names`).

        Returns
        -------
        dict or tuple
            ``emu_call`` (optionally with ``M_of_z`` and the blob): arrays of
            shape ``(n_z,)`` and a blob tuple for one point;
            ``(n_batch, n_z)`` arrays and an ``(n_batch, n_blob)`` blob array
            for a batch.

        Notes
        -----
        The cosmology is evaluated per point: cup1d analyses rescale one CAMB
        run (LaCE ``RescaledCosmology``), which is inexpensive. The IGM
        histories are evaluated for the whole batch at once.
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        values = self._parameter_values(like_params) or {}
        columns, n_batch, scalar = as_columns(values)
        cosmologies = [self.get_cosmology(row(columns, index)) for index in range(n_batch)]
        linear = base_cosmology.BaseCosmology.get_linP_Mpc_params_for_cosmologies(
            cosmologies, zs, self.emu_kp_Mpc
        )
        M_of_z = base_cosmology.BaseCosmology.get_dkms_dMpc_for_cosmologies(cosmologies, zs)
        if scalar:
            linear = {name: value[0] for name, value in linear.items()}
            M_of_z = M_of_z[0]
        emu_call = emulator_inputs(
            self.emulator.emu_params, zs, linear, M_of_z, self.model_igm,
            like_params=values if scalar else columns,
        )
        out = [emu_call]
        if return_M_of_z:
            out.append(M_of_z)
        if return_blob:
            blobs = np.asarray([self.get_blob(cosmo) for cosmo in cosmologies])
            out.append(tuple(blobs[0]) if scalar else blobs)
        return out[0] if len(out) == 1 else tuple(out)

    # Default sampler blobs; ``alpha_star`` is added when the running is
    # free (see :meth:`set_blob_names`). Any layout is a prefix of the legacy
    # order (Delta2_star, n_star, alpha_star, f_star, g_star, H0), so
    # positional readers of old 6-column files remain valid.
    blob_names = ("Delta2_star", "n_star")

    def set_blob_names(self, free_param_names):
        """Choose the sampler blobs for the given free parameters.

        Parameters
        ----------
        free_param_names : sequence of str
            Free likelihood parameters. ``alpha_star`` is stored only when
            ``nrun`` is free; otherwise it is fixed by the fiducial cosmology.
        """
        names = ["Delta2_star", "n_star"]
        if "nrun" in free_param_names:
            names.append("alpha_star")
        self.blob_names = tuple(names)

    def get_blobs_dtype(self):
        """Return field specifications for fitter cosmology summary blobs.

        Returns
        -------
        list of tuple
            ``(field_name, float)`` for each name in :attr:`blob_names`.
        """

        return [(name, float) for name in self.blob_names]

    def get_star_params(self, cosmo):
        """Return ``(Delta2_star, n_star, alpha_star)`` for one cosmology.

        Parameters
        ----------
        cosmo : lace.cosmo.base_cosmology.BaseCosmology
            Cosmology to summarize at ``z_star`` and ``kp_kms``.

        Returns
        -------
        tuple of float
            Compressed linear-power parameters, independent of
            :attr:`blob_names`.
        """
        params = cosmo.get_linP_kms_params(self.z_star, self.kp_kms)
        return params["Delta2_star"], params["n_star"], params["alpha_star"]

    def get_star_params_for_parameters(self, like_params):
        """Return :meth:`get_star_params` for a parameter point."""
        return self.get_star_params(self.get_cosmology(like_params))

    def get_blob(self, cosmo=None):
        """Return compressed cosmology information for fitter storage.

        Parameters
        ----------
        cosmo : lace.cosmo.base_cosmology.BaseCosmology, optional
            Cosmology to summarize. Missing input returns all-NaN placeholders.

        Returns
        -------
        tuple
            Values of :attr:`blob_names`, in that order.
        """
        if cosmo is None:
            return tuple([np.nan] * len(self.blob_names))
        star = dict(zip(("Delta2_star", "n_star", "alpha_star"), self.get_star_params(cosmo)))
        return tuple(star[name] for name in self.blob_names)

    def get_blob_for_parameters(self, like_params):
        """Return a fitter blob for a public or values-only parameter point.

        Parameters
        ----------
        like_params : mapping
            Cosmological parameter point.

        Returns
        -------
        tuple
            Summary values returned by :meth:`get_blob`.
        """

        return self.get_blob(self.get_cosmology(like_params))

    def get_P1D_kms(
        self,
        zs,
        k_ikms,
        like_params=None,
        return_covar=False,
        return_blob=None,
        return_emu_params=False,
        apply_hull=True,
        hires=False,
        remove=None,
        return_contaminants=False,
        k_ikms_emu=None,
        emulator_inputs=None,
        emulator_p1d_Mpc=None,
    ):
        """Evaluate the P1D with an explicitly named velocity-k argument.

        Parameters
        ----------
        zs : array_like
            Redshifts corresponding to ``k_ikms``.
        k_ikms : array_like or sequence of array_like
            Velocity-space wavenumbers in ``s / km``.
        like_params : mapping, optional
            Scalar or batched parameter point.
        return_covar, return_blob, return_emu_params, apply_hull, hires,
        remove, return_contaminants
            Passed unchanged to :meth:`get_p1d_kms`.
        k_ikms_emu : sequence of array_like, optional
            Per-redshift emulator grids in ``s / km``; passed to
            :meth:`get_p1d_kms` as ``k_kms_emu``.
        emulator_inputs : tuple, optional
            Precomputed ``(emu_call, M_of_z, blob)`` for ``zs`` and
            ``like_params``; see :meth:`get_emulator_calls`.
        emulator_p1d_Mpc : tuple, optional
            Precomputed emulator output; see :meth:`get_p1d_kms`.

        Returns
        -------
        object
            P1D prediction or the requested augmented result from
            :meth:`get_p1d_kms`.
        """
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
            k_kms_emu=k_ikms_emu,
            emulator_inputs=emulator_inputs,
            emulator_p1d_Mpc=emulator_p1d_Mpc,
        )

    def get_p1d_kms(
        self,
        zs,
        k_kms,
        like_params=None,
        return_covar=False,
        return_blob=None,
        return_emu_params=False,
        apply_hull=True,
        hires=False,
        remove=None,
        return_contaminants=False,
        k_kms_emu=None,
        emulator_inputs=None,
        emulator_p1d_Mpc=None,
    ):
        """Observed-model P1D in velocity units, for one point or a batch.

        Parameters
        ----------
        zs : array_like
            Redshifts.
        k_kms : sequence of array_like
            Per-redshift wavenumbers in ``s / km``.
        like_params : mapping, optional
            Scalar values (one point) or columns of shape ``(n_batch,)``.
        return_covar : bool, default: False
            Append an (empty) emulator-covariance list (historical API).
        return_blob : bool, optional
            Append the blob; by default for one point only.
        return_emu_params : bool, default: False
            Append the emulator inputs.
        apply_hull, hires : bool
            For one point, return ``None`` if the star priors or the
            (high-resolution) training hull reject it. A batch is not checked
            here (:meth:`cup1d.likelihood.likelihood.Likelihood.log_prob_and_blobs_batch`
            checks it before the emulator call).
        remove : mapping, optional
            Contaminant switches (figures).
        return_contaminants : bool, default: False
            Append the term-by-term contamination, one dict per redshift.
        k_kms_emu : sequence of array_like, optional
            Per-redshift grids (``s / km``) on which the emulator is
            evaluated; its P1D is then interpolated to ``k_kms``
            (:func:`interpolate_p1d`). Contaminants always use ``k_kms``.
        emulator_inputs : tuple, optional
            Precomputed ``(emu_call, M_of_z, blob)`` for exactly ``zs`` and
            ``like_params`` (:meth:`get_emulator_calls`).
        emulator_p1d_Mpc : tuple, optional
            Precomputed ``(p1d_Mpc, moments_Mpc)`` on the emulator grid
            (:meth:`emulate_p1d_Mpc_rows`); skips the emulator call.

        Returns
        -------
        list or list
            P1D per redshift, ``(n_k,)`` for one point and ``(n_batch, n_k)``
            for a batch, followed by the requested extras; ``None`` for a
            rejected point.
        """
        values = self._parameter_values(like_params) or {}
        columns, n_batch, scalar = as_columns(values)
        if return_blob is None:
            return_blob = scalar
        zs, k_kms = as_redshift_rows(zs, k_kms)
        n_z = len(zs)

        if emulator_inputs is None:
            emulator_inputs = self.get_emulator_calls(
                zs, like_params=values, return_M_of_z=True, return_blob=True
            )
        emu_call, M_of_z, blob = emulator_inputs
        if scalar and not self.is_admissible(
            zs, emu_call, blob, values, apply_hull=apply_hull, hires=hires
        ):
            return None

        # one implementation: a single point is a batch of one
        batch_call = {name: np.broadcast_to(value, (n_batch, n_z)) for name, value in emu_call.items()}
        M_batch = np.broadcast_to(np.asarray(M_of_z, dtype=float), (n_batch, n_z))
        k_eval = k_kms if k_kms_emu is None else as_redshift_rows(zs, k_kms_emu)[1]
        if emulator_p1d_Mpc is None:
            n_k = max(len(values_k) for values_k in k_eval)
            kin = np.zeros((n_batch, n_z, n_k))
            for iz, values_k in enumerate(k_eval):
                kin[:, iz, : len(values_k)] = values_k[None, :] * M_batch[:, iz, None]
            p1d_Mpc, moments_Mpc = self.emulate_p1d_Mpc_rows(zs, kin, batch_call, columns)
        else:
            p1d_Mpc, moments_Mpc = emulator_p1d_Mpc

        p1d_kms = []
        metal_moments = None if moments_Mpc is None else []
        for iz in range(n_z):
            n_k = len(k_eval[iz])
            power = p1d_Mpc[:, iz, :n_k] * M_batch[:, iz, None]
            if k_kms_emu is not None:
                power = interpolate_p1d(k_eval[iz], power, k_kms[iz])
            p1d_kms.append(power)
            if moments_Mpc is not None:
                moment_row = {}
                for name, moment in moments_Mpc.items():
                    moment = moment[:, iz, :n_k] * M_batch[:, iz, None]
                    if k_kms_emu is not None:
                        moment = interpolate_signed(k_eval[iz], moment, k_kms[iz])
                    moment_row[name] = moment
                metal_moments.append(moment_row)

        powers, terms = apply_observation_model(
            self.model_cont, self.model_syst, zs, k_kms, p1d_kms, batch_call["mF"],
            M_batch, like_params=columns, remove=remove, metal_moments=metal_moments,
        )
        if scalar:
            powers = [power[0] for power in powers]
            terms = [
                {key: value[0] if isinstance(value, np.ndarray) and value.ndim == 2 else value
                 for key, value in term.items()}
                for term in terms
            ]

        out = [powers]
        if return_covar:
            out.append([])
        if return_blob:
            out.append(blob)
        if return_emu_params:
            out.append(emu_call)
        if return_contaminants:
            out.append(terms)
        return out[0] if len(out) == 1 else out

    def is_admissible(self, zs, emu_call, blob, like_params, apply_hull=True, hires=False):
        """Return whether a point passes the star priors and the emulator hull.

        Parameters
        ----------
        zs : array_like
            Redshifts of ``emu_call``.
        emu_call : mapping
            Emulator inputs with one value per redshift.
        blob : tuple
            Values of :attr:`blob_names` for the point.
        like_params : mapping
            Scalar parameter values (used for star priors on unstored blobs).
        apply_hull, hires : bool
            Apply the (high-resolution) training-hull check.

        Returns
        -------
        bool
            ``False`` if a star prior or the hull rejects the point; checked
            before any emulator evaluation.
        """
        if not hasattr(self, "hull_checks"):
            self.hull_checks = self.hull_rejections = 0
        if self.star_priors is not None:
            star = None
            for key in self.star_priors:
                if key in self.blob_names:
                    value = blob[self.blob_names.index(key)]
                else:
                    if star is None:
                        star = dict(zip(
                            ("Delta2_star", "n_star", "alpha_star"),
                            self.get_star_params(self.get_cosmology(like_params)),
                        ))
                    value = star[key]
                if (value > self.star_priors[key][1]) or (value < self.star_priors[key][0]):
                    return False

        if self.use_hull and apply_hull:
            hull = self.hull_hires if hires else self.hull

            p0 = np.zeros((len(zs), len(hull.params)))
            for jj, key in enumerate(hull.params):
                p0[:, jj] = emu_call[key]
            self.hull_checks += 1
            if not hull.in_hulls(p0):
                self.hull_rejections += 1
                if getattr(self, "hull_reject", True):
                    return False
                warn_once(
                    "a point has emulator inputs outside the training hull: kept "
                    "(hull_reject: false), the emulator extrapolates there; the "
                    "fraction is reported with the fit",
                    key="outside_training_hull",
                )
        return True

    def hull_violations(self, zs, emu_call):
        """Pairs of emulator inputs (and redshifts) outside the training hull.

        Returns
        -------
        list of tuple
            ``(input_1, input_2, redshifts)`` for every violated projection;
            empty when inside (or without a hull).
        """
        from cup1d.emulator.hull import in_hull

        hull = getattr(self, "hull", None) if self.use_hull else None
        if hull is None:
            return []
        zs = np.atleast_1d(zs)
        points = np.stack([np.asarray(emu_call[key], dtype=float) for key in hull.params], axis=1)
        out = []
        for projection in hull.hulls:
            inside = in_hull(projection, points[:, [projection.dim0, projection.dim1]])
            if not inside.all():
                out.append((hull.params[projection.dim0], hull.params[projection.dim1],
                            [round(float(z), 2) for z in zs[~inside]]))
        return out

    def emulate_p1d_Mpc_rows(self, zs, kin_Mpc, emu_call, like_params):
        """Emulator P1D (and Kaiser metal moments) in Mpc units for a batch.

        Parameters
        ----------
        zs : array_like
            Redshifts.
        kin_Mpc : ndarray, shape (n_batch, n_z, n_k)
            Parallel wavenumbers in ``1 / Mpc``; shorter rows are zero padded.
        emu_call : mapping
            Emulator inputs of shape ``(n_batch, n_z)``.
        like_params : mapping
            Parameter columns of shape ``(n_batch,)`` (ForestFlow reads the
            cosmology from them); empty for the fiducial point.

        Returns
        -------
        p1d_Mpc : ndarray, shape (n_batch, n_z, n_k)
            Emulated P1D in ``Mpc``.
        moments_Mpc : dict or None
            Kaiser metal moments, each ``(n_batch, n_z, n_k)``, when that
            metal model is active; otherwise ``None``.
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        n_batch = np.asarray(kin_Mpc).shape[0]
        cosmologies = None
        if self.emulator.forestflow:
            # the same LaCE cosmology as the emulator inputs and M(z), so any
            # change of the background reaches the linear power too
            cosmologies = [self.get_cosmology(row(like_params, index)) for index in range(n_batch)]
        if "Si_kaiser" in getattr(self.model_cont, "metal_models", {}):
            return self.emulator.p1d_and_metal_moments_Mpc(zs, kin_Mpc, emu_call, cosmologies)
        return self.emulator.p1d_Mpc(zs, kin_Mpc, emu_call, cosmologies=cosmologies), None

    def get_parameters(self):
        """Return all theory, IGM, contamination, and systematic parameters.

        Returns
        -------
        list of dict
            Parameter metadata including fixed and sampled parameters.
        """

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

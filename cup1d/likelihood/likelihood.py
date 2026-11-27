import numpy as np
import os
import math
import copy
from mpi4py import MPI
from scipy.linalg import block_diag, cho_factor, cho_solve

from lace.cosmo.thermal_broadening import thermal_broadening_kms
from cup1d.utils import rebinning

from cup1d.utils.utils import split_string
from cup1d.utils.utils import get_path_repo
from cup1d.utils import blinding
from cup1d.utils import ic_files
from cup1d.utils.warn_once import numpy_errors_once
from cup1d.likelihood import parameter as parameter_space
from cup1d.utils.otypes import convert_otype, history_otype, interp_ic_history, legacy_otype
from cup1d.utils.batch import as_columns, row


def emulator_covariance_path(emulator):
    """File with the relative (leave-one-out) covariance of an emulator.

    Parameters
    ----------
    emulator : object
        Emulator exposing ``emulator_label`` (the internal one, e.g.
        ``CH24_mpgcen_gpr`` for ``lace_mpg``).

    Returns
    -------
    str
        ``<ForestFlow or LaCE>/data/covariance/l1O_cov_<emulator_label>.npy``.
    """
    label = emulator.emulator_label
    if getattr(emulator, "forestflow", "forest" in label):
        import forestflow

        root = os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "covariance")
    else:
        root = os.path.join(get_path_repo("lace"), "data", "covariance")
    return os.path.join(root, f"l1O_cov_{label}.npy")


def emulator_interpolation_weights(z, k_Mpc, emu_z, emu_k_Mpc):
    """Weights mapping data points onto the emulator-covariance grid.

    Parameters
    ----------
    z, k_Mpc : array_like
        Redshift and wavenumber (1/Mpc) of each data point.
    emu_z, emu_k_Mpc : array_like
        Redshift and wavenumber of each entry of the emulator covariance
        (``zz_zk`` and ``k_Mpc_zk``).

    Returns
    -------
    ndarray
        Matrix ``W`` of shape ``(n_data, n_emu)``. Each data point is linearly
        interpolated in redshift between the two bracketing emulator
        redshifts (clamped at the ends), using the nearest emulator
        wavenumber at each of them, so the projected relative covariance is
        ``W @ cov_zk @ W.T``, positive semidefinite by construction.

    Notes
    -----
    Nearest-redshift matching assigns data at z=2.4 and 2.6 (and 3.4 and
    3.6) to the same emulator redshift (2.5, 3.5) on the 0.25-spaced grid,
    making their emulator errors fully correlated; interpolation avoids it.
    """

    z = np.atleast_1d(np.asarray(z, dtype=float))
    k_Mpc = np.atleast_1d(np.asarray(k_Mpc, dtype=float))
    emu_z = np.asarray(emu_z, dtype=float)
    emu_k_Mpc = np.asarray(emu_k_Mpc, dtype=float)
    grid_z = np.unique(emu_z)
    members = {zz: np.flatnonzero(emu_z == zz) for zz in grid_z}

    weights = np.zeros((z.size, emu_z.size))
    for index, (z_point, k_point) in enumerate(zip(z, k_Mpc)):
        upper = np.searchsorted(grid_z, z_point)
        if upper == 0:
            nodes = [(grid_z[0], 1.0)]
        elif upper == grid_z.size:
            nodes = [(grid_z[-1], 1.0)]
        else:
            z_lo, z_hi = grid_z[upper - 1], grid_z[upper]
            frac = (z_point - z_lo) / (z_hi - z_lo)
            nodes = [(z_lo, 1.0 - frac), (z_hi, frac)]
        for z_node, weight in nodes:
            if weight == 0:
                continue
            candidates = members[z_node]
            nearest = candidates[np.argmin(np.abs(emu_k_Mpc[candidates] - k_point))]
            weights[index, nearest] += weight
    return weights





def emulator_kmax_iMpc(emulator):
    """Largest parallel wavenumber (1/Mpc) the P1D emulator was trained on."""
    for name in ("kmax_1d_iMpc", "kmax_Mpc"):  # ForestFlow, LaCE
        value = getattr(emulator, name, None)
        if value is not None:
            return float(value)
    return np.inf


def check_emulator_k_range(data, theory):
    """Raise if any data set reaches beyond the emulator's small-scale limit.

    The largest data wavenumber of every redshift, converted with the
    fiducial ``M(z)``, must not exceed :func:`emulator_kmax_iMpc` (the rule of
    the Cobaya route, ``cup1d.likelihood.external``).

    Raises
    ------
    ValueError
        Listing the offending data sets and redshifts.
    """
    kmax = emulator_kmax_iMpc(theory.emulator)
    if not np.isfinite(kmax):
        return
    cosmo = theory.fid_cosmo["cosmo"]
    beyond = []
    for key, dataset in data.items():
        for z, k_kms in zip(dataset.z, dataset.k_kms):
            k_iMpc = float(np.max(k_kms)) * float(cosmo.get_dkms_dMpc(z))
            if k_iMpc > kmax:
                beyond.append(f"{key} z={z:.2f}: k = {k_iMpc:.2f}/Mpc")
    if beyond:
        raise ValueError(
            f"data beyond the emulator's range (kmax = {kmax}/Mpc): " + "; ".join(beyond)
            + ". Cut the data at smaller k (e.g. kmax_ikms) or use another emulator."
        )

def _log_det_cov(icov, chol, method):
    """``log det C`` from the Cholesky factor or from ``C^-1`` (no underflow)."""
    if method == "cholesky" and chol is not None:
        return 2 * np.log(np.diag(chol[0])).sum()
    sign, logdet = np.linalg.slogdet(icov)
    return -logdet

def _at_a_time_value(results, iz, name):
    """Best-fit value of ``name`` in redshift bin ``iz`` of an at-a-time file.

    Handles the formats written so far: ``mle`` keyed by parameter name,
    with plain values or ``{"value": ...}`` entries, and older files keyed by
    LaTeX labels in the order of ``pnames``.
    """
    mle = results["mle"][iz]
    if name in mle:
        value = mle[name]
    else:
        names = list(results["pnames"][iz])
        # LaTeX-keyed parameters in pnames order, then the star blobs
        labels = [key for key in mle if key not in ("Delta2_star", "n_star", "alpha_star")]
        if name not in names or len(names) != len(labels):
            raise KeyError(f"{name} not in the at-a-time fit of redshift bin {iz}")
        value = mle[labels[names.index(name)]]
    if isinstance(value, dict):
        value = value["value"]
    return float(value)

class Likelihood(object):
    """Evaluate P1D data likelihoods using a theory prediction and priors.

    The object builds effective data-plus-emulator covariance at construction,
    manages free physical parameters, and exposes prediction, chi-squared, and
    posterior interfaces for inference drivers.
    """

    def __init__(
        self,
        data,
        theory,
        free_param_names=None,
        free_param_limits=None,
        verbose=None,
        cov_factor=1.0,
        prior_Gauss_rms=None,
        emu_cov_type="block",
        covariance_method="inverse",
        min_log_like=-1e100,
        args=None,
        start_from_min=True,
    ):
        """Initialize the native P1D likelihood and fixed data covariance.

        Parameters
        ----------
        data : dict
            Selected P1D data sets keyed by label.
        theory : cup1d.theory.theory.Theory
            Forward model supplying uncontaminated P1D predictions.
        free_param_names : sequence of str, optional
            Free physical parameter names. ``None`` delegates selection to the
            subsequent parameter setup.
        free_param_limits : sequence of tuple, optional
            Lower and upper physical limits in the same order as
            ``free_param_names``.
        verbose : bool, optional
            Print setup diagnostics from the MPI root rank; default: the
            configuration's ``verbose`` (``False`` without ``args``).
        cov_factor : float or mapping, default=1.0
            Factors of the statistical, systematic, emulator and full
            covariance: ``{"z", "val_stat", "val_syst", "val_emu",
            "val_full"}`` per redshift, or one float for all of them.
        prior_Gauss_rms : float, optional
            Fraction of each parameter's uniform width used for Gaussian
            priors. ``None`` instead uses parameter-specific widths or no
            Gaussian prior.
        emu_cov_type : {"diagonal", "block", "full"}, default="block"
            Correlation structure retained from emulator covariance.
        covariance_method : {"inverse", "cholesky"}, default="inverse"
            Numerical representation used for Gaussian contractions.
        min_log_like : float, default=-1e100
            Finite floor returned for rejected likelihood evaluations.
        args : cup1d.configuration.args.Args
            Configuration providing rebinning, initial-condition, and related
            native analysis settings.
        start_from_min : bool, default=True
            Load configured initial conditions when their file is present.

        Notes
        -----
        The likelihood sets data covariance once at construction. Its raw
        chi-squared is a data diagnostic; priors are added separately when a
        posterior is evaluated.
        """

        self.rank = MPI.COMM_WORLD.Get_rank()

        # explicit argument, else the configuration
        if verbose is None:
            verbose = bool(getattr(args, "verbose", False)) if args is not None else False
        self.verbose = verbose
        # explicit argument, else the configuration (null: no Gaussian prior)
        if prior_Gauss_rms is None and args is not None:
            prior_Gauss_rms = getattr(args, "prior_Gauss_rms", None)
        self.prior_Gauss_rms = prior_Gauss_rms
        if np.isscalar(cov_factor):
            # one factor for every covariance term at every redshift
            cov_factor = {"z": np.array([0.0]), **{
                name: np.array([float(cov_factor)])
                for name in ("val_stat", "val_syst", "val_emu", "val_full")
            }}
        self.cov_factor = cov_factor
        self.emu_cov_type = emu_cov_type
        if covariance_method not in {"inverse", "cholesky"}:
            raise ValueError("covariance_method must be 'inverse' or 'cholesky'")
        self.covariance_method = covariance_method
        self.min_log_like = min_log_like
        self.data = data
        # we only do this for latter save all relevant after fitting the model
        self.args = args

        # set a class containing the rebinned data
        self.Rebin_data = rebinning.Rebinning(
            self.data,
            k_rebin_factor=args.k_rebin_factor,
            n_k_emu=getattr(args, "n_k_emu", None),
            shared_k_grid=getattr(args, "shared_k_grid", True),
        )

        self.theory = theory
        # stop if the data reach smaller scales than the emulator was trained
        # on (larger scales are fine); the same rule as the Cobaya route
        check_emulator_k_range(self.data, self.theory)
        # Set inverse covariance. We do it here so we can account for emulator error
        self.set_icov()

        # setup parameters
        self.free_param_names = free_param_names
        self.set_free_parameters(free_param_names, free_param_limits)
        # sampler blobs: Delta2_star, n_star (+ alpha_star if nrun is free)
        self.theory.set_blob_names(self.free_param_names)
        if verbose and (self.rank == 0):
            print(len(self.free_params), "free parameters")

        self.set_Gauss_priors()

        # TBD (should be hanging from mock), true model (when working with mocks)
        self.set_truth()

        # store model
        self.set_model()

        # blinding offsets of the first data set that requests blinding
        self.blind = blinding.Blinding.from_data(self.data.values())

        # set IC for likelihood: the variation's own file, else the baseline
        # of the same sample and emulator, else the configured file_ic
        if start_from_min and (args.file_ic is not None):
            ic_file = ic_files.resolve_ic_file(args)
            if ic_file is not None:
                if self.rank == 0:
                    tried = ic_files.ic_candidates(args)
                    if ic_file != tried[0]:
                        print(
                            f"No ICs at {tried[0]}; falling back to the next "
                            "available file",
                            flush=True,
                        )
                    print("Loading ICs from", ic_file)
                # At-a-time files store per-redshift fits under "pnames";
                # global files are keyed by parameter name.
                ic_content = np.load(ic_file, allow_pickle=True).item()
                if "pnames" in ic_content:
                    if self.rank == 0:
                        print("Setting ICs from at a time fit")
                    self.set_ic_from_z_at_time(ic_file, verbose=self.verbose)
                else:
                    if self.rank == 0:
                        print("Setting ICs from global fit")
                    self.set_ic_global(ic_file, verbose=self.verbose)
            else:
                if self.rank == 0:
                    tried = "\n".join(f"  {path}" for path in ic_files.ic_candidates(args))
                    print(
                        f"No initial-condition file found, tried:\n{tried}\n"
                        "Starting from the fiducial parameter values instead.\n"
                        "Generate initial conditions with:\n"
                        "  python scripts/create_global_initial_conditions.py <global_opt yaml>\n"
                        "  python scripts/create_at_a_time_initial_conditions.py <at_a_time_global yaml>\n"
                        "or save them from a minimizer run with Analysis.save_global_ic().",
                        flush=True,
                    )

    def set_Gauss_priors(self):
        """Build Gaussian priors aligned with the free-parameter order.

        Returns
        -------
        None
            Sets ``Gauss_priors`` (widths) and ``Gauss_priors_mean``
            (centres), or ``None`` for both when no Gaussian prior is active.

        Notes
        -----
        Widths, by precedence: ``prior_Gauss_rms`` times the flat-prior range
        (every parameter); the top-level ``Gauss_priors`` mapping of the
        configuration (``{parameter: width}``); the parameter's own
        ``Gauss_priors_width``. The centres are the parameter values when the
        priors are built, before any initial condition replaces them, so a
        previous fit to the same data never becomes the prior.

        Raises
        ------
        ValueError
            If the ``Gauss_priors`` mapping names a parameter that is not free.
        """
        by_name = getattr(self.args, "Gauss_priors", None) or {}
        unknown = set(by_name) - set(self.free_params)
        if unknown:
            raise ValueError(f"Gauss_priors for parameters that are not free: {sorted(unknown)}")
        widths = np.full(len(self.free_params), np.inf)
        for ii, (name, parameter) in enumerate(self.free_params.items()):
            if self.prior_Gauss_rms is not None:
                widths[ii] = self.prior_Gauss_rms * (parameter["max_value"] - parameter["min_value"])
            elif name in by_name and by_name[name] is not None:
                widths[ii] = by_name[name]
            elif parameter["Gauss_priors_width"] is not None:
                widths[ii] = parameter["Gauss_priors_width"]
        if np.all(np.isinf(widths)):
            self.Gauss_priors = None
            self.Gauss_priors_mean = None
        else:
            self.Gauss_priors = widths
            self.Gauss_priors_mean = np.asarray(
                [parameter["value"] for parameter in self.free_params.values()], dtype=float
            )

    def set_icov(self, emulator_covariance=None, fiducial_conversion=None):
        """Build effective P1D covariance, Cholesky factors, and inverses.

        Returns
        -------
        None
            Sets per-redshift and, where provided, full-vector covariance,
            inverse-covariance, Cholesky, and emulator-covariance dictionaries
            keyed by data-set label.

        Raises
        ------
        numpy.linalg.LinAlgError
            If a scaled data-plus-emulator covariance is not positive definite.

        Notes
        -----
        The emulator bundle is loaded from ForestFlow for labels containing
        ``'forest'`` and from LaCE otherwise. Its relative covariance is
        projected to each data grid using the fiducial velocity conversion;
        ``emu_cov_type`` selects diagonal, within-redshift block, or full
        redshift-and-wavenumber correlations.
        """

        if (emulator_covariance is None) != (fiducial_conversion is None):
            raise ValueError(
                "emulator_covariance and fiducial_conversion must be supplied together"
            )
        if emulator_covariance is None:
            full_path = emulator_covariance_path(self.theory.emulator)
            emu_cov = np.load(full_path, allow_pickle=True).item()
            fiducial_conversion = self.theory.fid_cosmo["cosmo"].get_dkms_dMpc
        else:
            emu_cov = emulator_covariance
        # contains:
        # dict_save["zz"] = zz
        # dict_save["k_Mpc"] = k_Mpc
        # cross-k
        # dict_save["k_Mpc_k"] = k_Mpc_k
        # dict_save["cov_k"] = cov
        # cross-zk
        # dict_save["zz_zk"] = zz_zk
        # dict_save["k_Mpc_zk"] = k_Mpc_k
        # dict_save["cov_zk"] = cov

        # split in redshifts
        self.icov_Pk_kms = {}
        self.chol_Pk_kms = {}
        self.cov_Pk_kms = {}
        self.cov_emu_Pk_kms = {}
        # all redshifts together, for full Pk
        self.full_icov_Pk_kms = {}
        self.full_chol_Pk_kms = {}
        self.full_cov_Pk_kms = {}
        self.emu_full_cov_Pk_kms = {}

        # Iterate over both datasets: main dataset (idata = 0) and additional dataset (idata = 1)
        for key in self.data:
            data = self.data[key]
            # initialize, to store the results for different redshifts
            icov_Pk_kms = []
            chol_Pk_kms = []
            cov_Pk_kms = []
            cov_emu_Pk_kms = []

            # TBD need to ensure that we have a Pksmooth_kms for the emulator covariance
            if data.Pksmooth_kms is not None:
                pksmooth = data.Pksmooth_kms
            else:
                pksmooth = data.Pk_kms

            # Total number of k values across all redshifts
            nks = 0
            for ii in range(len(data.z)):
                nks += len(data.Pk_kms[ii])

            # Process each redshift bin
            emu_cov_blocks = []
            for ii in range(len(data.z)):
                # Copy the covariance matrix for the current redshift bin
                cov_stat = data.covstat_Pk_kms[ii].copy()
                cov_syst = data.cov_Pk_kms[ii] - cov_stat

                # inflate errors
                ind = np.argmin(np.abs(self.cov_factor["z"] - data.z[ii]))
                # inflate errors stat
                cov_stat *= self.cov_factor["val_stat"][ind] ** 2
                # inflate errors syst
                cov_syst *= self.cov_factor["val_syst"][ind] ** 2
                emu_cov_factor = self.cov_factor["val_emu"][ind] ** 2

                # Full covariance after inflating errors
                cov = cov_stat + cov_syst
                # Set emulator covariance, we stored relative difference
                # also add emulator covariance to stat + syst covariance

                # data k_kms to Mpc
                dkms_dMpc = fiducial_conversion(data.z[ii])
                k_Mpc = data.k_kms[ii] * dkms_dMpc

                # relative emulator covariance interpolated in z to this
                # redshift, rescaled by the smooth P1D since it stores the
                # relative error
                weights = emulator_interpolation_weights(
                    np.full(k_Mpc.shape[0], data.z[ii]),
                    k_Mpc,
                    emu_cov["zz_zk"],
                    emu_cov["k_Mpc_zk"],
                )
                rel_emu_cov = weights @ emu_cov["cov_zk"] @ weights.T
                add_emu_cov_kms = (
                    rel_emu_cov
                    * np.outer(pksmooth[ii], pksmooth[ii])
                    * emu_cov_factor
                )
                if self.emu_cov_type == "diagonal":
                    add_emu_cov_kms = np.diag(np.diag(add_emu_cov_kms))
                cov += add_emu_cov_kms

                emu_cov_blocks.append(add_emu_cov_kms)

                # inflate errors full
                ind = np.argmin(np.abs(self.cov_factor["z"] - data.z[ii]))
                cov *= self.cov_factor["val_full"][ind] ** 2

                # Factor once. Cholesky is used directly by the likelihood when
                # requested; keep the inverse for established diagnostic APIs.
                try:
                    chol_Pk_kms.append(cho_factor(cov, lower=True, check_finite=False))
                except np.linalg.LinAlgError as error:
                    raise np.linalg.LinAlgError(
                        f"Covariance for data set {key!r}, z={data.z[ii]:.3f} is not positive definite"
                    ) from error
                icov_Pk_kms.append(np.linalg.inv(cov))
                cov_Pk_kms.append(cov)
                cov_emu_Pk_kms.append(add_emu_cov_kms)

            self.icov_Pk_kms[key] = icov_Pk_kms
            self.chol_Pk_kms[key] = chol_Pk_kms
            self.cov_Pk_kms[key] = cov_Pk_kms
            self.cov_emu_Pk_kms[key] = cov_emu_Pk_kms

            # Process the full power spectrum data if available
            if data.full_Pk_kms is not None:
                # inflate errors
                cov_stat = data.full_cov_stat_Pk_kms.copy()
                cov_syst = data.full_cov_Pk_kms - cov_stat
                for i0 in range(cov_stat.shape[0]):
                    ind0 = np.argmin(np.abs(self.cov_factor["z"] - data.full_zs[i0]))
                    for i1 in range(cov_stat.shape[0]):
                        ind1 = np.argmin(
                            np.abs(self.cov_factor["z"] - data.full_zs[i1])
                        )
                        cov_stat[i0, i1] = (
                            cov_stat[i0, i1]
                            * self.cov_factor["val_stat"][ind0]
                            * self.cov_factor["val_stat"][ind1]
                        )
                        cov_syst[i0, i1] = (
                            cov_syst[i0, i1]
                            * self.cov_factor["val_syst"][ind0]
                            * self.cov_factor["val_syst"][ind1]
                        )
                cov = cov_stat + cov_syst
                # diagonal emu, already inflated
                if self.emu_cov_type == "diagonal":
                    diag_emu_cov = []
                    for ii in range(len(emu_cov_blocks)):
                        diag_emu_cov.append(np.diag(emu_cov_blocks[ii]))
                    full_emu_cov = np.concatenate(diag_emu_cov)
                    ind = np.diag_indices_from(cov)
                    cov[ind] += full_emu_cov
                # block emu, already inflated
                elif self.emu_cov_type == "block":
                    full_emu_cov = block_diag(*emu_cov_blocks)
                    cov += full_emu_cov
                # full emu
                else:
                    # The emulator covariance stores errors relative to the
                    # smooth P1D, so rescale by the smooth data P1D (as in the
                    # per-redshift blocks), not by the noisy measurement.
                    full_pksmooth = np.concatenate(pksmooth)
                    full_k_check = np.concatenate(data.k_kms)
                    full_z_check = np.concatenate(
                        [np.full(len(k), z) for k, z in zip(data.k_kms, data.z)]
                    )
                    if (
                        full_pksmooth.shape != data.full_Pk_kms.shape
                        or not np.allclose(full_k_check, data.full_k_kms)
                        or not np.allclose(full_z_check, data.full_zs)
                    ):
                        raise ValueError(
                            "Per-redshift smooth P1D does not align with the full "
                            "data vector; cannot build the full emulator covariance"
                        )
                    dkms_dMpc = {
                        z: fiducial_conversion(z) for z in np.unique(data.full_zs)
                    }
                    full_k_Mpc = np.asarray(
                        [
                            k_kms * dkms_dMpc[z]
                            for k_kms, z in zip(data.full_k_kms, data.full_zs)
                        ]
                    )
                    weights = emulator_interpolation_weights(
                        data.full_zs,
                        full_k_Mpc,
                        emu_cov["zz_zk"],
                        emu_cov["k_Mpc_zk"],
                    )
                    infl_index = np.argmin(
                        np.abs(
                            np.asarray(self.cov_factor["z"])[None, :]
                            - np.asarray(data.full_zs)[:, None]
                        ),
                        axis=1,
                    )
                    scale = (
                        full_pksmooth * self.cov_factor["val_emu"][infl_index]
                    )
                    full_emu_cov = (
                        weights @ emu_cov["cov_zk"] @ weights.T
                    ) * np.outer(scale, scale)

                    cov += full_emu_cov

                # inflate errors full
                for i0 in range(cov.shape[0]):
                    ind0 = np.argmin(np.abs(self.cov_factor["z"] - data.full_zs[i0]))
                    fact0 = self.cov_factor["val_full"][ind0]

                    for i1 in range(cov.shape[0]):
                        ind1 = np.argmin(
                            np.abs(self.cov_factor["z"] - data.full_zs[i1])
                        )
                        fact1 = self.cov_factor["val_full"][ind1]

                        cov[i0, i1] = cov[i0, i1] * fact0 * fact1

                try:
                    self.full_chol_Pk_kms[key] = cho_factor(cov, lower=True, check_finite=False)
                except np.linalg.LinAlgError as error:
                    raise np.linalg.LinAlgError(
                        f"Full covariance for data set {key!r} is not positive definite"
                    ) from error
                self.full_icov_Pk_kms[key] = np.linalg.inv(cov)
                self.full_cov_Pk_kms[key] = cov
                self.emu_full_cov_Pk_kms[key] = full_emu_cov

    def set_free_parameters(self, free_param_names, free_param_limits):
        """Select and optionally limit theory parameters for likelihood fitting.

        Parameters
        ----------
        free_param_names : sequence of str
            Theory parameter names retained as free likelihood coordinates.
        free_param_limits : sequence of tuple, optional
            Physical ``(min_value, max_value)`` overrides in matching order.

        Returns
        -------
        None
            Sets ordered ``free_params`` and ``free_param_names`` attributes.

        Raises
        ------
        ValueError
            If limits have the wrong length or a requested name is absent from
            the theory parameter definitions.
        """

        if free_param_limits is not None and len(free_param_limits) != len(
            free_param_names
        ):
            raise ValueError("wrong number of parameter limits")

        parameters = self.theory.get_parameters()
        self.free_params = {}
        for index, name in enumerate(free_param_names):
            if name not in parameters:
                raise ValueError(f"Could not find free parameter {name} in theory")
            parameter = copy.deepcopy(parameters[name])
            if free_param_limits is not None:
                parameter["min_value"], parameter["max_value"] = (
                    free_param_limits[index]
                )
            self.free_params[name] = parameter

        self.free_param_names = list(self.free_params)
        if self.verbose and self.rank == 0:
            print(f"likelihood setup with {len(self.free_params)} free parameters")

    def set_truth(self):
        """Store compatible mock-data truth in physical and unit-cube forms.

        Returns
        -------
        None
            Sets ``truth`` to ``None`` for observations or to simulation truth
            plus compatible free-parameter and compressed-cosmology values.

        Notes
        -----
        IGM truth coefficients are only assigned when the simulation IGM grid
        matches the fiducial model grid; otherwise their stored values are
        ``np.inf`` to indicate incompatibility.
        """

        # access true cosmology used in mock data
        primary_data = next(iter(self.data.values()))
        if not hasattr(primary_data, "truth"):
            if self.rank == 0:
                print("will not store truth, working with real data")
            self.truth = None
            return

        self.truth = {}
        for par in primary_data.truth:
            self.truth[par] = primary_data.truth[par]

        # make sure that we compare the correct zs
        ztruth = primary_data.truth["igm"]["z"]
        zfid = self.theory.model_igm.fid_igm["z"]
        mask_z = np.zeros(len(ztruth), dtype=int) - 1
        for ii in range(len(mask_z)):
            ind = np.argwhere(ztruth[ii] == zfid)[:, 0]
            if len(ind) != 0:
                mask_z[ii] = ind[0]

        ind = np.argwhere(mask_z != -1)[:, 0]
        mask_z = mask_z[ind]

        # equal_IGM for each IGM differently!!!
        equal_IGM = True
        for key in primary_data.truth["igm"]:
            if key not in self.theory.model_igm.fid_igm:
                continue
            lenz = self.theory.model_igm.fid_igm[key].shape[0]
            if (
                np.allclose(
                    np.array(primary_data.truth["igm"][key])[mask_z],
                    self.theory.model_igm.fid_igm[key],
                )
                == False
            ):
                equal_IGM = False
                break

        self.truth["like_params"] = {}
        self.truth["like_params_cube"] = {}
        pname2 = {"As": "Delta2_star", "ns": "n_star", "nrun": "alpha_star"}
        for name, parameter in self.free_params.items():
            if (
                ("tau" in name)
                | ("sigT" in name)
                | ("gamma" in name)
                | ("kF" in name)
            ):
                if equal_IGM:
                    if "tau" in name:
                        self.truth["like_params"][name] = 1
                        self.truth["like_params_cube"][name] = (
                            parameter_space.value_in_cube(self.free_params, name, self.truth["like_params"][name])
                        )
                    else:
                        self.truth["like_params"][name] = 0
                        self.truth["like_params_cube"][name] = (
                            parameter_space.value_in_cube(self.free_params, name, self.truth["like_params"][name])
                        )
                else:
                    self.truth["like_params"][name] = np.infty
                    self.truth["like_params_cube"][name] = np.infty
            elif (name == "As") | (name == "ns") | (name == "nrun"):
                self.truth["like_params"][name] = self.truth["cosmo"][name]
                self.truth["like_params_cube"][name] = parameter_space.value_in_cube(self.free_params, name, self.truth["like_params"][name])
                self.truth["like_params"][pname2[name]] = self.truth["linP"][
                    pname2[name]
                ]
            # else:
            #     if name not in self.truth["cont"]:
            #         print("could not find {} in truth".format(name))
            #         continue
            #     self.truth["like_params"][name] = self.truth["cont"][
            #         name
            #     ]
            #     self.truth["like_params_cube"][
            #         name
            #     ] = parameter_space.value_in_cube(self.free_params, name, self.truth["cont"][name])

    def set_model(self):
        """Store fiducial cosmology, IGM, free-parameter, and linP metadata.

        Returns
        -------
        None
            Builds the ``fid`` dictionary used by result and diagnostic code.
        """

        self.fid = {}

        sim_cosmo = self.theory.fid_cosmo["cosmo"]
        background = sim_cosmo.get_background_params()
        primordial = sim_cosmo.get_primordial_params()

        self.fid["cosmo"] = {
            "ombh2": background["ombh2"],
            "omch2": background["omch2"],
            "As": primordial["As"],
            "ns": primordial["ns"],
            "nrun": primordial["nrun"],
            "H0": sim_cosmo.get_H0(),
            "mnu": sim_cosmo.get_mnu(),
        }

        blob_params = ["Delta2_star", "n_star", "alpha_star"]
        blob = self.theory.fid_cosmo["linP_params"]

        self.fid["igm"] = self.theory.model_igm.fid_igm
        self.fid["fit"] = {}
        self.fid["fit_cube"] = {}
        self.fid["linP"] = {}

        pname2 = {"As": "Delta2_star", "ns": "n_star", "nrun": "alpha_star"}
        for name, parameter in self.free_params.items():
            self.fid["fit"][name] = parameter["value"]
            self.fid["fit_cube"][name] = parameter_space.value_in_cube(self.free_params, name, parameter["value"])
            if (name == "As") | (name == "ns") | (name == "nrun"):
                self.fid["fit"][pname2[name]] = blob[pname2[name]]
                self.fid["linP"][pname2[name]] = blob[pname2[name]]

    def get_P1D_kms(
        self,
        parameters=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
        apply_hull=True,
        remove=None,
    ):
        """Compute theoretical P1D in km/s using the canonical public name.

        Parameters
        ----------
        parameters : mapping, optional
            Physical free-parameter values.
        return_covar, return_blob, return_emu_params : bool, default=False
            Forward requested auxiliary theory products.
        apply_hull : bool, default=True
            Reject predictions outside the emulator admission hull.
        remove : sequence of str, optional
            Contaminant contributions omitted from the prediction.

        Returns
        -------
        tuple or None
            Same result as :meth:`get_p1d_kms`.
        """
        return self.get_p1d_kms(
            parameters=parameters,
            return_covar=return_covar,
            return_blob=return_blob,
            return_emu_params=return_emu_params,
            apply_hull=apply_hull,
            remove=remove,
        )

    def _dataset_params(self, key, like_params):
        """Drop resolution-systematic parameters for data sets without them.

        Parameters
        ----------
        key : str
            Data label of the data set being predicted.
        like_params : mapping
            Scalar or columnar physical parameter values.

        Returns
        -------
        mapping
            ``like_params`` unchanged when ``args.syst_data_labels`` is
            ``None`` or contains ``key``; otherwise a copy without
            ``R_coeff_*`` entries, so the theory applies no resolution
            correction to this data set.
        """
        from cup1d.models.observation import dataset_nuisance

        return dataset_nuisance(key, like_params, getattr(self.args, "syst_data_labels", None))

    def _group_params(self, keys, like_params):
        """Parameters for the contaminated P1D of a group of data sets.

        A single data set keeps its own parameters. A group sharing a fine
        grid is evaluated without ``R_coeff_*``; each data set then applies
        its own resolution correction (:meth:`_dataset_fine_p1d`).
        """
        from cup1d.models.observation import without_resolution

        if len(keys) == 1:
            return self._dataset_params(keys[0], like_params)
        return without_resolution(like_params)

    def _dataset_fine_p1d(self, keys, key, p1ds, like_params):
        """Fine-grid P1D of data set ``key`` from its group's contaminated P1D.

        Parameters
        ----------
        keys : list
            The group (:attr:`Rebinning.groups`).
        key : hashable
            Data set within it.
        p1ds : list of ndarray
            Group P1D on the shared fine grid, scalar or columnar rows.
        like_params : mapping
            Scalar values or columns.

        Returns
        -------
        list of ndarray
            ``p1ds`` times the data set's resolution correction (the last
            factor of the observation model), or ``p1ds`` for one data set.
        """
        from cup1d.models.observation import apply_resolution

        if len(keys) == 1:
            return p1ds
        return apply_resolution(
            self.theory.model_syst, np.atleast_1d(self.Rebin_data.zs[key]),
            self.Rebin_data.k_kms[key], p1ds, self._dataset_params(key, like_params))

    def _emulator_inputs_per_dataset(self, like_params):
        """Evaluate emulator inputs once and slice them for every data set.

        Parameters
        ----------
        like_params : mapping
            Scalar physical parameter values, or columns of shape
            ``(n_batch,)``.

        Returns
        -------
        dict
            ``{key: (emu_call, M_of_z, blob)}`` for each data set (blobs are
            ``(n_batch, n_blob)`` for columnar input), as
            :meth:`Theory.get_emulator_calls` would return for that data
            set's redshifts. The union inputs are kept in
            ``self._union_emulator_call`` and ``self._union_M_of_z`` for
            :meth:`_emulator_p1d_per_dataset`.

        Notes
        -----
        Exact: every emulator input is a pointwise function of redshift for
        a given parameter point, and the blob depends on cosmology only.
        """
        z_union = np.unique(np.concatenate(
            [np.atleast_1d(zs) for zs in self.Rebin_data.zs.values()]
        ))
        emu_call, M_of_z, blob = self.theory.get_emulator_calls(
            z_union, like_params=like_params, return_M_of_z=True, return_blob=True
        )
        self._union_emulator_call = emu_call
        M_of_z = np.asarray(M_of_z)
        self._union_M_of_z = M_of_z
        out = {}
        for key, zs in self.Rebin_data.zs.items():
            zs = np.atleast_1d(zs)
            index = np.searchsorted(z_union, zs)
            # the last axis is redshift for scalar (n_z,) and batched
            # (n_batch, n_z) inputs alike
            out[key] = (
                {name: np.asarray(values)[..., index] for name, values in emu_call.items()},
                M_of_z[..., index],
                blob,
            )
        return out

    def _union_emulator_grid(self):
        """Return the union redshift/k grid on which the emulator is evaluated.

        Returns
        -------
        z_union : ndarray
            Sorted unique redshifts of all data sets.
        k_union : list of ndarray
            Per union redshift, the concatenation of every data set's emulator
            grid in ``s / km`` (``k_kms_emu`` when set, else ``k_kms``).
        slices : dict
            ``{key: [(iu, start, stop), ...]}``, one entry per data-set
            redshift, locating its grid inside ``k_union[iu]``.

        Notes
        -----
        The data grids are fixed, so the result is computed once and cached.
        """
        if getattr(self, "_union_grid", None) is not None:
            return self._union_grid
        z_union = np.unique(np.concatenate(
            [np.atleast_1d(zs) for zs in self.Rebin_data.zs.values()]
        ))
        rows = [[] for _ in z_union]
        lengths = np.zeros(len(z_union), dtype=int)
        slices = {}
        # an identical row (shared fine grid) is evaluated once
        seen = [{} for _ in z_union]
        for key, zs in self.Rebin_data.zs.items():
            k_emu = self.Rebin_data.k_kms_emu[key]
            k_list = self.Rebin_data.k_kms[key] if k_emu is None else k_emu
            slices[key] = []
            for iz, z in enumerate(np.atleast_1d(zs)):
                iu = int(np.searchsorted(z_union, z))
                k_row = np.asarray(k_list[iz], dtype=float).reshape(-1)
                fingerprint = k_row.tobytes()
                if fingerprint in seen[iu]:
                    slices[key].append(seen[iu][fingerprint])
                    continue
                rows[iu].append(k_row)
                seen[iu][fingerprint] = (iu, lengths[iu], lengths[iu] + len(k_row))
                slices[key].append(seen[iu][fingerprint])
                lengths[iu] += len(k_row)
        k_union = [np.concatenate(row) for row in rows]
        self._union_grid = (z_union, k_union, slices)
        return self._union_grid

    def _emulator_p1d_per_dataset(self, like_params):
        """Evaluate the emulator once on the union grid and split it.

        Call after :meth:`_emulator_inputs_per_dataset` for the same columns.

        Parameters
        ----------
        like_params : mapping
            Columns of shape ``(n_batch,)``.

        Returns
        -------
        dict
            ``{key: (p1d, moments)}``: the emulator P1D (``Mpc``) as an
            ``(n_batch, n_z, n_k)`` array (zero padded) and the Kaiser metal
            moments in the same layout or ``None``; the ``emulator_p1d_Mpc``
            argument of :meth:`Theory.get_p1d_kms`.
        """
        z_union, k_union, slices = self._union_emulator_grid()
        M_of_z = np.atleast_2d(self._union_M_of_z)
        n_batch = M_of_z.shape[0]
        emu_call = {name: np.broadcast_to(values, M_of_z.shape)
                    for name, values in self._union_emulator_call.items()}
        n_k = max(len(k_row) for k_row in k_union)
        kin = np.zeros((n_batch, len(z_union), n_k))
        for iu, k_row in enumerate(k_union):
            kin[:, iu, :len(k_row)] = k_row[None, :] * M_of_z[:, iu, None]
        p1d, moments = self.theory.emulate_p1d_Mpc_rows(z_union, kin, emu_call, like_params)

        def split(values, items):
            n_k_key = max(b - a for _, a, b in items)
            array = np.zeros((n_batch, len(items), n_k_key))
            for iz, (iu, a, b) in enumerate(items):
                array[:, iz, :b - a] = values[:, iu, a:b]
            return array

        out = {}
        for key, items in slices.items():
            key_moments = None
            if moments is not None:
                key_moments = {name: split(values, items) for name, values in moments.items()}
            out[key] = (split(p1d, items), key_moments)
        return out

    def predict_batch(self, columns, remove=None, apply_hull=True, return_emu_params=False):
        """Rebinned model P1D of every data set for a batch of points.

        The single forward model of the likelihood: cosmology and IGM
        (emulator inputs) once for all data sets, star priors and training
        hull before the emulator, one emulator call on the union of the data
        grids, then the observation model per group of data sets and the
        rebinning. One point is a batch of one.

        Parameters
        ----------
        columns : mapping of str to ndarray
            Physical parameter columns of shape ``(n_batch,)``.
        remove : mapping, optional
            Contaminant switches (figures).
        apply_hull : bool, default=True
            Apply the training-hull check (the star priors always apply).
        return_emu_params : bool, default=False
            Also return the emulator inputs of each data set.

        Returns
        -------
        predictions : dict
            ``{key: [ndarray (n_valid, n_k_data) per redshift]}`` for the
            accepted points.
        valid : ndarray of bool, shape (n_batch,)
            Points accepted by the star priors and the training hull.
        blobs : ndarray, shape (n_batch, n_blob)
            Blobs of every point.
        emu_params : dict, optional
            ``{key: emulator inputs (n_valid, n_z)}`` when requested.
        """
        emulator_inputs = self._emulator_inputs_per_dataset(columns)
        blobs = np.asarray(next(iter(emulator_inputs.values()))[2])
        valid = self._admissible_rows(columns, emulator_inputs, apply_hull=apply_hull)
        predictions, emu_params = {}, {}
        if not np.any(valid):
            return (predictions, valid, blobs, emu_params) if return_emu_params else (predictions, valid, blobs)
        if not np.all(valid):
            columns = {name: values[valid] for name, values in columns.items()}
            emulator_inputs = self._emulator_inputs_per_dataset(columns)
        emulator_p1d = self._emulator_p1d_per_dataset(columns)
        for keys in self.Rebin_data.groups:
            # data sets of a group share redshifts and the fine grid: one
            # contaminated P1D, then each data set's resolution correction
            lead = keys[0]
            fine = self.theory.get_p1d_kms(
                self.Rebin_data.zs[lead],
                self.Rebin_data.k_kms[lead],
                self._group_params(keys, columns),
                k_kms_emu=self.Rebin_data.k_kms_emu[lead],
                remove=remove,
                emulator_inputs=emulator_inputs[lead],
                emulator_p1d_Mpc=emulator_p1d[lead],
            )
            for key in keys:
                own = self._dataset_fine_p1d(keys, key, fine, columns)
                predictions[key] = self.Rebin_data.rebinning(key, own)
                emu_params[key] = emulator_inputs[key][0]
        if return_emu_params:
            return predictions, valid, blobs, emu_params
        return predictions, valid, blobs

    def get_p1d_kms(
        self,
        parameters=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
        apply_hull=True,
        remove=None,
    ):
        """Rebinned model P1D of every data set at one point.

        Parameters
        ----------
        parameters : mapping, optional
            Full or partial named physical point; omitted values use defaults.
        return_covar, return_blob, return_emu_params : bool, default=False
            Request the corresponding extras (historical API: they are
            returned only with ``return_blob`` or ``return_emu_params``).
        apply_hull : bool, default=True
            Apply the training-hull check.
        remove : mapping, optional
            Contaminant switches (figures).

        Returns
        -------
        tuple or None
            ``(predictions, extras)``: P1D per data set and redshift, and per
            data set the list ``[covariances], [blob], [emulator inputs]`` as
            requested; ``None`` for a point rejected by the star priors or the
            training hull. See :meth:`predict_batch` (one point is a batch of
            one).
        """
        like_params = (
            {} if parameters is None
            else parameter_space.values_from_point(self.free_params, parameters)
        )
        columns = {name: np.array([value], dtype=float) for name, value in like_params.items()}
        predictions, valid, blobs, emu_params = self.predict_batch(
            columns, remove=remove, apply_hull=apply_hull, return_emu_params=True)
        if not valid[0]:
            return None
        blob = tuple(blobs[0])
        all_p1ds, other_stuff = {}, {}
        for key, rows in predictions.items():
            all_p1ds[key] = [values[0] for values in rows]
            extras = []
            if return_blob or return_emu_params:
                if return_covar:
                    extras.append([])
                if return_blob:
                    extras.append(blob)
                if return_emu_params:
                    extras.append({name: values[0] for name, values in emu_params[key].items()})
            other_stuff[key] = extras
        return all_p1ds, other_stuff

    def get_chi2(self, parameters=None, return_all=False, zmask=None):
        """Compute chi2 using data and theory (data plus emulator covariance).

        Parameters
        ----------
        parameters : mapping, optional
            Physical free-parameter values.
        return_all : bool, default=False
            Also return per-data-set, per-redshift chi-squared arrays.
        zmask : float or array-like, optional
            Exactly one diagnostic redshift, or ``None`` for a joint fit.

        Returns
        -------
        float or tuple
            Total chi-squared, optionally with per-redshift contributions.

        Notes
        -----
        ``zmask`` is a diagnostic single-redshift fit only. It intentionally
        uses that redshift's covariance block and cannot retain cross-redshift
        covariance terms. Use ``zmask=None`` for a joint fit.
        """

        log_like, log_like_all = self.get_log_like(
            parameters, ignore_log_det_cov=True, zmask=zmask
        )

        # rejected prediction (out of bounds or non-finite model)
        if not isinstance(log_like_all, dict):
            if return_all:
                return np.inf, {}
            return np.inf

        chi2_eachz = {}
        for key in log_like_all:
            chi2_eachz[key] = -2.0 * log_like_all[key]

        chi2_total = -2.0 * log_like

        if return_all:
            return chi2_total, chi2_eachz
        else:
            return chi2_total

    def log_like_rows(self, predictions, zmask=None, ignore_log_det_cov=True, per_redshift=False):
        """Gaussian log likelihood of a batch of predictions.

        Parameters
        ----------
        predictions : dict
            ``{key: [ndarray (n, n_k_data) per redshift]}`` (:meth:`predict_batch`).
        zmask : float, optional
            One diagnostic redshift (its covariance block only).
        ignore_log_det_cov : bool, default=True
            Omit the covariance normalization.
        per_redshift : bool, default=False
            Also return the per-redshift contributions (from the redshift
            blocks of the covariance, also when the full covariance is used
            for the total).

        Returns
        -------
        log_like : ndarray, shape (n,)
        per_z : dict
            ``{key: ndarray (n, n_z)}`` with ``per_redshift`` (else empty).
        """
        n = len(next(iter(predictions.values()))[0])
        log_like = np.zeros(n)
        per_z = {}
        for key in self.Rebin_data.zs:
            data = self.data[key]
            full_icov = self.full_icov_Pk_kms[key]
            use_full = full_icov is not None and zmask is None
            if per_redshift or not use_full:
                blocks = np.zeros((n, len(data.z)))
                for iz, redshift in enumerate(data.z):
                    if zmask is not None and not np.any(np.abs(zmask - redshift) < 1.0e-3):
                        continue
                    residual = np.asarray(data.Pk_kms[iz])[None, :] - predictions[key][iz]
                    blocks[:, iz] = -0.5 * self._chi2_rows(
                        residual, self.icov_Pk_kms[key][iz], self.chol_Pk_kms[key][iz],
                        ignore_log_det_cov)
                if per_redshift:
                    per_z[key] = blocks
                if not use_full:
                    log_like += blocks.sum(axis=1)
            if use_full:
                residual = data.full_Pk_kms[None, :] - np.concatenate(predictions[key], axis=1)
                log_like -= 0.5 * self._chi2_rows(
                    residual, full_icov, self.full_chol_Pk_kms.get(key), ignore_log_det_cov)
        return log_like, per_z

    def _chi2_rows(self, residual, icov, chol, ignore_log_det_cov=True):
        """``r C^-1 r`` (plus ``ln det C`` unless ignored) of residual rows ``(n, m)``."""
        # extreme emulator outputs (e.g. near prior edges) can overflow; the
        # resulting inf/nan rejects the point
        with numpy_errors_once("likelihood chi2"):
            if self.covariance_method == "cholesky":
                chi2 = np.sum(residual * cho_solve(chol, residual.T, check_finite=False).T, axis=1)
            else:
                chi2 = np.sum((residual @ icov) * residual, axis=1)
        if not ignore_log_det_cov:
            chi2 = chi2 + _log_det_cov(icov, chol, self.covariance_method)
        return chi2

    def get_log_like(
        self,
        parameters=None,
        ignore_log_det_cov=True,
        return_blob=False,
        zmask=None,
    ):
        """Gaussian log likelihood of one point and its per-redshift contributions.

        Parameters
        ----------
        parameters : mapping, optional
            Physical free-parameter values.
        ignore_log_det_cov : bool, default=True
            Omit covariance normalization determinants.
        return_blob : bool, default=False
            Include the blob of the point.
        zmask : float or array-like, optional
            Exactly one diagnostic redshift, or ``None`` for the joint fit.

        Returns
        -------
        list
            ``[log_like, log_like_all]`` and, when requested,
            ``[log_like, log_like_all, blob]``. Rejected or non-finite
            predictions return negative infinities (and a zero blob).

        A non-null ``zmask`` may select exactly one redshift. This is for
        one-redshift diagnostic fits: selecting a subset omits cross-redshift
        covariance, so multi-redshift masks are rejected.
        """
        zmask = self._validate_single_redshift_mask(zmask)
        null_out = [-np.inf, -np.inf]
        if return_blob:
            null_out.append(tuple([0] * len(self.theory.blob_names)))
        like_params = (
            {} if parameters is None
            else parameter_space.values_from_point(self.free_params, parameters)
        )
        columns = {name: np.array([value], dtype=float) for name, value in like_params.items()}
        predictions, valid, blobs = self.predict_batch(columns)
        if not valid[0]:
            return null_out
        log_like, per_z = self.log_like_rows(
            predictions, zmask=zmask, ignore_log_det_cov=ignore_log_det_cov, per_redshift=True)
        log_like = log_like[0]
        # something went wrong (nan, or +inf from overflow)
        if not np.isfinite(log_like):
            return null_out
        out = [log_like, {key: values[0] for key, values in per_z.items()}]
        if return_blob:
            out.append(tuple(blobs[0]))
        return out

    @staticmethod
    def _validate_single_redshift_mask(zmask):
        """Normalize a diagnostic redshift mask and reject unsafe subsets.

        Parameters
        ----------
        zmask : float or array-like, optional
            One finite redshift, or ``None`` for a full joint fit.

        Returns
        -------
        ndarray or None
            One-element floating-point array or ``None``.

        Raises
        ------
        ValueError
            If more than one redshift, a non-vector value, or a non-finite
            redshift is supplied.
        """

        if zmask is None:
            return None
        zmask = np.asarray(zmask, dtype=float)
        if zmask.ndim == 0:
            zmask = zmask.reshape(1)
        elif zmask.ndim != 1:
            raise ValueError("zmask must be one redshift value or None")
        if zmask.size != 1:
            raise ValueError(
                "zmask supports exactly one redshift. A multi-redshift subset "
                "would drop retained cross-redshift covariance; use zmask=None "
                "for a joint fit."
            )
        if not np.isfinite(zmask[0]):
            raise ValueError("zmask must contain one finite redshift")
        return zmask

    def regulate_log_like(self, log_like):
        """Replace invalid or excessively small likelihood values with a floor.

        Parameters
        ----------
        log_like : float or None
            Candidate log likelihood.

        Returns
        -------
        float
            ``min_log_like`` for ``None``/NaN values, otherwise the maximum
            of the candidate and ``min_log_like``.
        """

        if (log_like is None) or math.isnan(log_like):
            return self.min_log_like

        return max(self.min_log_like, log_like)

    def parameters_in_bounds(self, parameters):
        """Return whether named physical values satisfy uniform prior bounds.

        Parameters
        ----------
        parameters : mapping
            Full or partial named physical parameter values.

        Returns
        -------
        bool
            True only when every free parameter lies within its inclusive
            configured physical interval.
        """

        parameters = parameter_space.values_from_point(self.free_params, parameters)
        return all(
            parameter["min_value"] <= parameters[name] <= parameter["max_value"]
            for name, parameter in self.free_params.items()
        )

    def get_log_prior(self, parameters):
        """Compute uniform-bound and optional Gaussian log prior.

        Parameters
        ----------
        parameters : mapping
            Full or partial named physical parameter values.

        Returns
        -------
        float
            ``min_log_like`` outside uniform bounds, zero with no active
            Gaussian priors, or the summed Gaussian log prior.
        """

        parameters = parameter_space.values_from_point(self.free_params, parameters)
        if not self.parameters_in_bounds(parameters):
            return self.min_log_like
        if self.Gauss_priors is None:
            return 0.0
        values = np.asarray([parameters[name] for name in self.free_params])
        return -np.sum(
            (self.Gauss_priors_mean - values) ** 2 / (2 * self.Gauss_priors**2)
        )

    def compute_log_prob(
        self, parameters, return_blob=False, ignore_log_det_cov=True, zmask=None
    ):
        """Compute posterior from physical values, likelihood, and priors.

        Parameters
        ----------
        parameters : mapping
            Full or partial named physical parameter values.
        return_blob : bool, default=False
            Return the theory blob with the posterior.
        ignore_log_det_cov : bool, default=True
            Forward covariance-normalization choice to likelihood evaluation.
        zmask : float or array-like, optional
            One diagnostic redshift or ``None``.

        Returns
        -------
        float or tuple
            Posterior log probability, optionally paired with theory blob.
        """

        parameters = parameter_space.values_from_point(self.free_params, parameters)
        if not self.parameters_in_bounds(parameters):
            if return_blob:
                return self.min_log_like, self.theory.get_blob()
            return self.min_log_like

        log_prior = self.get_log_prior(parameters)
        if return_blob:
            log_like, _, blob = self.get_log_like(
                parameters,
                ignore_log_det_cov=ignore_log_det_cov,
                return_blob=True,
                zmask=zmask,
            )
        else:
            log_like, _ = self.get_log_like(
                parameters,
                ignore_log_det_cov=ignore_log_det_cov,
                return_blob=False,
                zmask=zmask,
            )
        log_like = self.regulate_log_like(log_like)
        if return_blob:
            return log_like + log_prior, blob
        return log_like + log_prior

    def log_prob(self, parameters, ignore_log_det_cov=True, zmask=None):
        """Return posterior log probability for physical parameter values.

        Parameters
        ----------
        parameters : mapping
            Full or partial named physical parameter values.
        ignore_log_det_cov : bool, default=True
            Omit covariance normalization determinants.
        zmask : float or array-like, optional
            One diagnostic redshift or ``None``.

        Returns
        -------
        float
            Posterior log probability.
        """

        return self.compute_log_prob(
            parameters,
            return_blob=False,
            ignore_log_det_cov=ignore_log_det_cov,
            zmask=zmask,
        )

    def log_prob_and_blobs(
        self, parameters, ignore_log_det_cov=True, zmask=None
    ):
        """Return posterior log probability and flattened theory blobs.

        Parameters
        ----------
        parameters : mapping
            Full or partial named physical parameter values.
        ignore_log_det_cov : bool, default=True
            Omit covariance normalization determinants.
        zmask : float or array-like, optional
            One diagnostic redshift or ``None``.

        Returns
        -------
        tuple
            Posterior log probability followed by theory blob values.
        """

        lnprob, blob = self.compute_log_prob(
            parameters,
            return_blob=True,
            ignore_log_det_cov=ignore_log_det_cov,
            zmask=zmask,
        )
        return lnprob, *blob

    def _parameter_columns(self, parameters_batch):
        """Physical parameter columns of every free parameter.

        Parameters
        ----------
        parameters_batch : mapping or iterable of mapping
            Columns of shape ``(n_batch,)``, or one mapping per point.

        Returns
        -------
        dict of str to ndarray
            ``(n_batch,)`` per free parameter (absent ones at their value).

        Raises
        ------
        ValueError
            If a column is not one-dimensional or columns differ in length.
        """
        if not isinstance(parameters_batch, dict):
            points = [parameter_space.values_from_point(self.free_params, point)
                      for point in parameters_batch]
            parameters_batch = {name: np.array([point[name] for point in points], dtype=float)
                                for name in self.free_params}
        columns, n_batch, scalar = as_columns(parameters_batch)
        if scalar and parameters_batch:
            raise ValueError("batched parameters must have shape (n_batch,)")
        return {
            name: columns[name] if name in columns else np.full(n_batch, parameter["value"], dtype=float)
            for name, parameter in self.free_params.items()
        } | {name: values for name, values in columns.items() if name not in self.free_params}

    def _admissible_rows(self, columns, emulator_inputs, apply_hull=True):
        """Star-prior and training-hull check of every row of a batch.

        Parameters
        ----------
        columns : mapping
            Physical parameter columns of shape ``(n_batch,)``.
        emulator_inputs : dict
            Output of :meth:`_emulator_inputs_per_dataset` for ``columns``.
        apply_hull : bool, default=True
            Apply the training-hull check.

        Returns
        -------
        ndarray of bool
            ``True`` for rows that :meth:`Theory.is_admissible` accepts.
        """
        n_batch = len(next(iter(columns.values()))) if columns else 1
        if self.theory.star_priors is None and not (self.theory.use_hull and apply_hull):
            return np.ones(n_batch, dtype=bool)
        z_union = self._union_emulator_grid()[0]
        blobs = next(iter(emulator_inputs.values()))[2]
        union_call = self._union_emulator_call
        return np.asarray([
            self.theory.is_admissible(
                z_union,
                {name: np.atleast_2d(values)[index] for name, values in union_call.items()},
                blobs[index],
                row(columns, index),
                apply_hull=apply_hull,
            )
            for index in range(n_batch)
        ])

    def log_prob_and_blobs_batch(
        self, parameters_batch, ignore_log_det_cov=True, zmask=None
    ):
        """Posterior and blobs of a batch of physical parameter points.

        Parameters
        ----------
        parameters_batch : mapping or iterable of mapping
            Columns of shape ``(n_batch,)``, or one mapping per point.
        ignore_log_det_cov : bool, default=True
            Omit covariance normalization determinants.
        zmask : float or array-like, optional
            One diagnostic redshift or ``None`` for joint fitting.

        Returns
        -------
        list of tuple
            One tuple per point: the log posterior followed by the blob
            values. Points out of bounds, rejected by the star priors or the
            training hull, or with non-finite predictions get the posterior
            floor ``min_log_like``.
        """
        zmask = self._validate_single_redshift_mask(zmask)
        columns = self._parameter_columns(parameters_batch)
        names = list(self.free_params)
        values = np.stack([columns[name] for name in names], axis=1)
        low = np.array([self.free_params[name]["min_value"] for name in names])
        high = np.array([self.free_params[name]["max_value"] for name in names])
        in_bounds = np.all((values >= low) & (values <= high), axis=1)
        n_points = len(values)
        log_post = np.full(n_points, self.min_log_like, dtype=float)
        blobs = np.tile(np.asarray(self.theory.get_blob(), dtype=float), (n_points, 1))
        if np.any(in_bounds):
            inside = {name: column[in_bounds] for name, column in columns.items()}
            predictions, valid, inside_blobs = self.predict_batch(inside)
            index = np.flatnonzero(in_bounds)
            blobs[index] = inside_blobs
            if np.any(valid):
                log_like, _ = self.log_like_rows(
                    predictions, zmask=zmask, ignore_log_det_cov=ignore_log_det_cov)
                log_prior = 0.0
                if self.Gauss_priors is not None:
                    log_prior = -np.sum(
                        (self.Gauss_priors_mean - values[index[valid]]) ** 2
                        / (2 * self.Gauss_priors**2), axis=1)
                # fmax maps NaN (non-finite predictions) to the floor
                log_post[index[valid]] = np.fmax(log_like, self.min_log_like) + log_prior
        return [(log_post[i], *blobs[i]) for i in range(n_points)]

    def old_plot_p1d(
        self,
        values=None,
        plot_every_iz=1,
        residuals=False,
        plot_fname=None,
        rand_posterior=None,
        show=True,
        return_covar=False,
        print_ratio=False,
        print_chi2=True,
        return_all=False,
        collapse=False,
        plot_realizations=True,
        zmask=None,
        n_perturb=0,
        plot_panels=False,
        z_at_time=False,
        fontsize=20,
        glob_full=False,
        fix_cosmo=False,
        n_param_glob_full=16,
        chi2_nozcov=False,
        ylims=None,
        store_data=False,
    ):
        """Delegate to :func:`cup1d.postprocessing.likelihood.old_plot_p1d`."""
        from cup1d.postprocessing.likelihood import old_plot_p1d as _plot

        return _plot(
            self,
            values,
            plot_every_iz,
            residuals,
            plot_fname,
            rand_posterior,
            show,
            return_covar,
            print_ratio,
            print_chi2,
            return_all,
            collapse,
            plot_realizations,
            zmask,
            n_perturb,
            plot_panels,
            z_at_time,
            fontsize,
            glob_full,
            fix_cosmo,
            n_param_glob_full,
            chi2_nozcov,
            ylims,
            store_data,
        )

    def plot_p1d(
        self,
        values=None,
        plot_every_iz=1,
        residuals=False,
        plot_fname=None,
        rand_posterior=None,
        show=True,
        return_covar=False,
        print_ratio=False,
        print_chi2=True,
        return_all=False,
        collapse=False,
        plot_realizations=True,
        zmask=None,
        n_perturb=0,
        plot_panels=False,
        z_at_time=False,
        fontsize=20,
        glob_full=False,
        fix_cosmo=False,
        n_param_glob_full=16,
        chi2_nozcov=False,
        ylims=None,
        store_data=False,
    ):
        """Render model/data P1D diagnostics through the maintained plotter.

        Parameters
        ----------
        values : array-like or mapping, optional
            Sampling coordinates or physical parameter values to plot.
        plot_every_iz : int, default=1
            Redshift-bin stride.
        residuals, print_ratio, print_chi2, return_all, collapse, plot_panels
            Diagnostic-layout and reporting switches forwarded unchanged.
        plot_fname : path-like, optional
            Optional figure destination.
        rand_posterior : ndarray, optional
            Posterior samples used for uncertainty realizations.
        show, return_covar, plot_realizations, z_at_time, glob_full, fix_cosmo,
        chi2_nozcov, store_data : bool
            Rendering and diagnostic options owned by the postprocessing API.
        zmask : float or array-like, optional
            One-redshift diagnostic selection.
        n_perturb, fontsize, n_param_glob_full : int
            Plotting/perturbation controls.
        ylims : tuple, optional
            Vertical limits.

        Returns
        -------
        object
            Exact product returned by the maintained postprocessing renderer.
        """
        from cup1d.postprocessing.likelihood import plot_p1d as _plot

        return _plot(
            self,
            values,
            plot_every_iz,
            residuals,
            plot_fname,
            rand_posterior,
            show,
            return_covar,
            print_ratio,
            print_chi2,
            return_all,
            collapse,
            plot_realizations,
            zmask,
            n_perturb,
            plot_panels,
            z_at_time,
            fontsize,
            glob_full,
            fix_cosmo,
            n_param_glob_full,
            chi2_nozcov,
            ylims,
            store_data,
        )

    def plot_p1d_errors(
        self,
        values=None,
        plot_fname=None,
        show=True,
        zmask=None,
        z_at_time=False,
        fontsize=16,
    ):
        """Render P1D covariance/error diagnostics through the plotter.

        Parameters
        ----------
        values : array-like or mapping, optional
            Point whose model/error state is shown.
        plot_fname : path-like, optional
            Optional figure destination.
        show : bool, default=True
            Display the figure interactively.
        zmask : float or array-like, optional
            One-redshift diagnostic selection.
        z_at_time : bool, default=False
            Use at-a-time redshift organization.
        fontsize : int, default=16
            Base text size.

        Returns
        -------
        object
            Exact product returned by the postprocessing renderer.
        """
        from cup1d.postprocessing.likelihood import plot_p1d_errors as _plot

        return _plot(self, values, plot_fname, show, zmask, z_at_time, fontsize)

    # def overplot_emulator_calls(
    #     self,
    #     param_1,
    #     param_2,
    #     values=None,
    #     tau_scalings=True,
    #     temp_scalings=True,
    # ):
    #     """For parameter pair (param1,param2), overplot emulator calls
    #     with values stored in archive, color coded by redshift"""

    #     # mask post-process scalings (optional)
    #     emu_data = self.theory.emulator.archive.data
    #     Nemu = len(emu_data)
    #     if not tau_scalings:
    #         mask_tau = [x["scale_tau"] == 1.0 for x in emu_data]
    #     else:
    #         mask_tau = [True] * Nemu
    #     if not temp_scalings:
    #         mask_temp = [
    #             (x["scale_T0"] == 1.0) & (x["scale_gamma"] == 1.0)
    #             for x in emu_data
    #         ]
    #     else:
    #         mask_temp = [True] * Nemu

    #     # figure out values of param_1,param_2 in archive
    #     emu_1 = np.array(
    #         [
    #             emu_data[i][param_1]
    #             for i in range(Nemu)
    #             if (mask_tau[i] & mask_temp[i])
    #         ]
    #     )
    #     emu_2 = np.array(
    #         [
    #             emu_data[i][param_2]
    #             for i in range(Nemu)
    #             if (mask_tau[i] & mask_temp[i])
    #         ]
    #     )

    #     # translate sampling point (in unit cube) to parameter values
    #     if values is not None:
    #         like_params = self.parameters_from_sampling_point(values)
    #     else:
    #         like_params = []
    #     emu_calls = self.theory.get_emulator_calls(like_params=like_params)
    #     # figure out values of param_1,param_2 called
    #     call_1 = [emu_call[param_1] for emu_call in emu_calls]
    #     call_2 = [emu_call[param_2] for emu_call in emu_calls]

    #     # overplot
    #     zs = self.data.z
    #     emu_z = np.array(
    #         [
    #             emu_data[i]["z"]
    #             for i in range(Nemu)
    #             if (mask_tau[i] & mask_temp[i])
    #         ]
    #     )
    #     zmin = min(min(emu_z), min(zs))
    #     zmax = max(max(emu_z), max(zs))
    #     plt.scatter(emu_1, emu_2, c=emu_z, s=1, vmin=zmin, vmax=zmax)
    #     plt.scatter(call_1, call_2, c=zs, s=50, vmin=zmin, vmax=zmax)
    #     cbar = plt.colorbar()
    #     cbar.set_label("Redshift", labelpad=+1)
    #     plt.xlabel(param_1)
    #     plt.ylabel(param_2)
    #     plt.show()

    #     return

    def plot_hcd_cont(
        self,
        zstar=3,
        p0=None,
        chain=None,
        save_directory=None,
        ftsize=24,
        nelem=5000,
        store_data=False,
    ):
        """Render HCD-contaminant posterior diagnostics.

        Parameters are forwarded to
        :func:`cup1d.postprocessing.contaminants.plot_hcd_cont`; ``zstar`` is
        the velocity-pivot redshift and ``chain`` provides posterior samples.
        """
        from cup1d.postprocessing.contaminants import plot_hcd_cont as _plot

        return _plot(self, zstar, p0, chain, save_directory, ftsize, nelem, store_data)

    def plot_metal_cont_add(
        self,
        free_params=None,
        chain=None,
        save_directory=None,
        ftsize=24,
        nelem=5000,
        store_data=False,
    ):
        """Render additive-metal contamination diagnostics from a fit chain."""
        from cup1d.postprocessing.contaminants import plot_metal_cont_add as _plot

        return _plot(
            self,
            free_params,
            chain,
            save_directory,
            ftsize,
            nelem,
            store_data,
        )

    def plot_metal_cont_mult(
        self,
        free_params=None,
        chain=None,
        zstar=3,
        save_directory=None,
        ftsize=24,
        nelem=5000,
        store_data=False,
    ):
        """Render multiplicative-metal contamination diagnostics from a fit chain."""
        from cup1d.postprocessing.contaminants import plot_metal_cont_mult as _plot

        return _plot(
            self,
            free_params,
            chain,
            zstar,
            save_directory,
            ftsize,
            nelem,
            store_data,
        )

    def plot_igm(
        self,
        cloud=False,
        chain_uformat=None,
        free_params=None,
        save_directory=None,
        zmask=None,
        plot_type="all",
        plot_fid=True,
        lab_fid="mpg-central",
        ftsize=18,
        nelem=20000,
        title="",
        pre_xylims=True,
        plot_more_igm=False,
        variation_label="baseline",
        store_data=False,
        plot_external_data=True,
        plot_truth=False,
    ):
        """Render IGM-history diagnostics through the maintained plotter.

        The options select posterior source, redshift masking, fiducial/truth
        overlays, external data, layout, and optional data persistence.
        """
        from cup1d.postprocessing.igm import plot_likelihood_igm as _plot

        return _plot(
            self,
            cloud,
            chain_uformat,
            free_params,
            save_directory,
            zmask,
            plot_type,
            plot_fid,
            lab_fid,
            ftsize,
            nelem,
            title,
            pre_xylims,
            plot_more_igm,
            variation_label,
            store_data,
            plot_external_data,
            plot_truth,
        )

    def plot_cov_terms(self, save_directory=None):
        """Render data, systematic, and emulator covariance contributions."""
        from cup1d.postprocessing.likelihood import plot_cov_terms as _plot

        return _plot(self, save_directory)

    def plot_cov_to_pk(
        self, use_pk_smooth=True, fname=None, ftsize=18, store_data=False
    ):
        """Render covariance relative to P1D, optionally using smooth P1D."""
        from cup1d.postprocessing.likelihood import plot_cov_to_pk as _plot

        return _plot(self, use_pk_smooth, fname, ftsize, store_data)

    def plot_correlation_matrix(self, save_directory=None):
        """Render the fitted data-vector correlation matrix."""
        from cup1d.postprocessing.likelihood import plot_correlation_matrix as _plot

        return _plot(self, save_directory)

    def plot_hull_fid(self, like_params=None):
        """Render the fiducial point against the emulator admission hull."""
        from cup1d.postprocessing.likelihood import plot_hull_fid as _plot

        return _plot(self, like_params)

    def set_ic_from_z_at_time(self, fname, verbose=True):
        """Initialize free coefficients from independent redshift-bin fits.

        Parameters
        ----------
        fname : path-like
            NumPy result dictionary from an at-a-time fit, with best-fit
            parameter values indexed by redshift.
        verbose : bool, default=True
            Print assigned physical values on the MPI root rank.

        Returns
        -------
        None
            Updates free-parameter values and resets IGM/HCD/metal model
            coefficients consistently.

        Raises
        ------
        ValueError
            If a free coefficient has no matching configured redshift node.
        """

        dir_out = np.load(fname, allow_pickle=True).item()

        # Update the physical fiducial values from the saved best fit.
        for name, parameter in self.free_params.items():
            if name in ["As", "ns"]:
                continue
            pname, iistr = split_string(name)
            ii = int(iistr)

            if (pname + "_znodes") in self.args.fid_igm:
                znode = self.args.fid_igm[pname + "_znodes"][ii]
            elif (pname + "_znodes") in self.args.fid_cont:
                znode = self.args.fid_cont[pname + "_znodes"][ii]
            elif (pname + "_znodes") in self.args.fid_syst:
                znode = self.args.fid_syst[pname + "_znodes"][ii]
            else:
                raise ValueError("Could not find znode for " + name)

            iz = np.argmin(np.abs(dir_out["z"] - znode))
            # print(iz, znode, dir_out["z"][iz])
            # print(dir_out["pnames"][iz], pname + "_0")
            # convert from the otype the file was written with (files that
            # do not record it used the baseline otypes)
            saved_otype = dir_out.get("otypes", {}).get(pname + "_0") or legacy_otype(pname)
            parameter["value"] = convert_otype(
                _at_a_time_value(dir_out, iz, pname + "_0"),
                saved_otype,
                history_otype(self.args, pname),
                floor=parameter["min_value"],
            )

            if verbose and (self.rank == 0):
                print(
                    name,
                    "\t",
                    np.round(parameter["value"], 3),
                    "\t",
                    np.round(parameter["min_value"], 3),
                    "\t",
                    np.round(parameter["max_value"], 3),
                    "\t",
                    parameter["Gauss_priors_width"],
                    parameter["fixed"],
                )

        # reset the coefficients of the models
        parameter_values = {
            name: parameter["value"]
            for name, parameter in self.free_params.items()
        }
        self.theory.model_igm.models["F_model"].reset_coeffs(
            parameter_values, rank=self.rank
        )
        self.theory.model_igm.models["T_model"].reset_coeffs(
            parameter_values, rank=self.rank
        )
        self.theory.model_cont.hcd_model.reset_coeffs(parameter_values, rank=self.rank)
        self.theory.model_cont.metal_models["Si_mult"].reset_coeffs(
            parameter_values, rank=self.rank
        )
        self.theory.model_cont.metal_models["Si_add"].reset_coeffs(
            parameter_values, rank=self.rank
        )

    def set_ic_global(self, fname, verbose=True):
        """Initialize free coefficients from a global redshift-dependent fit.

        Parameters
        ----------
        fname : path-like
            NumPy result dictionary containing fitted coefficient histories,
            and optionally scalar ``As`` and ``ns`` entries (``{'val': x}``).
        verbose : bool, default=True
            Print assigned physical values on the MPI root rank.

        Returns
        -------
        None
            Updates configured fixed flags, interpolates coefficient values at
            model nodes, resets contaminant/IGM coefficient models, and sets
            free ``As``/``ns`` when the file stores them (otherwise they keep
            their fiducial values). Coefficient families absent from the file
            (e.g. a metal model newer than it) also keep their fiducial values.

        Raises
        ------
        ValueError
            If a requested parameter has no matching model-node configuration.
        """
        dir_out = np.load(fname, allow_pickle=True).item()

        # Update the physical fiducial values from the saved best fit.
        for name, parameter in self.free_params.items():
            if name in ["As", "ns"]:
                # Older files do not store cosmology; keep the fiducial value.
                if name in dir_out:
                    parameter["value"] = float(dir_out[name]["val"])
                    if verbose and (self.rank == 0):
                        print(name, "\t", parameter["value"])
                continue
            pname, iistr = split_string(name)
            ii = int(iistr)

            if (pname + "_znodes") in self.args.fid_igm:
                znode = self.args.fid_igm[pname + "_znodes"][ii]
                isfixed = self.args.fid_igm[pname + "_fixed"]
            elif (pname + "_znodes") in self.args.fid_cont:
                znode = self.args.fid_cont[pname + "_znodes"][ii]
                isfixed = self.args.fid_cont[pname + "_fixed"]
            elif (pname + "_znodes") in self.args.fid_syst:
                znode = self.args.fid_syst[pname + "_znodes"][ii]
                isfixed = self.args.fid_syst[pname + "_fixed"]
            else:
                raise ValueError(
                    pname + "_znodes not found in either fid_igm, fid_cont, or fid_syst"
                )

            parameter["fixed"] = isfixed

            if (pname not in dir_out) and (pname == "HCD_const"):
                parameter["value"] = 0
            elif pname not in dir_out:
                # e.g. a metal model newer than the file: keep the fiducial value
                if verbose and (self.rank == 0):
                    print(f"{name}: not in the IC file, keeping its fiducial value")
            else:
                # interpolated in the otype the file was written with, then
                # converted to the configured one
                parameter["value"] = interp_ic_history(
                    dir_out[pname],
                    pname,
                    znode,
                    history_otype(self.args, pname),
                    floor=parameter["min_value"],
                )

            if verbose and (self.rank == 0):
                print(
                    name,
                    "\t",
                    np.round(parameter["value"], 3),
                    "\t",
                    np.round(parameter["min_value"], 3),
                    "\t",
                    np.round(parameter["max_value"], 3),
                    "\t",
                    parameter["Gauss_priors_width"],
                    parameter["fixed"],
                )

        # reset the coefficients of the models
        parameter_values = {
            name: parameter["value"]
            for name, parameter in self.free_params.items()
        }
        # self.theory.model_igm.models["F_model"].reset_coeffs(
        #     free_params, rank=self.rank
        # )
        # self.theory.model_igm.models["T_model"].reset_coeffs(
        #     free_params, rank=self.rank
        # )
        for model in self.theory.model_igm.models:
            self.theory.model_igm.models[model].reset_coeffs(
                parameter_values, rank=self.rank
            )

        self.theory.model_cont.hcd_model.reset_coeffs(parameter_values, rank=self.rank)

        # self.theory.model_cont.metal_models["Si_mult"].reset_coeffs(
        #     free_params, rank=self.rank
        # )
        # self.theory.model_cont.metal_models["Si_add"].reset_coeffs(
        #     free_params, rank=self.rank
        # )

        for model in self.theory.model_cont.metal_models:
            self.theory.model_cont.metal_models[model].reset_coeffs(
                parameter_values, rank=self.rank
            )


def others_igm():
    # Galdwick 2021
    """Return external IGM measurements in cup1d model conventions.

    Returns
    -------
    dict
        Literature measurements keyed by source, with redshift, mean flux,
        temperature, thermal-width, slope, and propagated one-sigma errors.
        Temperatures are converted to the IGM-model convention where needed.
    """
    z = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8])

    # mean transmitted flux
    Fmean = np.array(
        [
            0.8690,
            0.8261,
            0.7919,
            0.7665,
            0.7398,
            0.7105,
            0.6731,
            0.5927,
            0.5320,
            0.4695,
        ]
    )

    # uncertainty on mean transmitted flux
    dFmean = np.array(
        [
            0.0214,
            0.0206,
            0.0210,
            0.0216,
            0.0212,
            0.0213,
            0.0223,
            0.0247,
            0.0280,
            0.0278,
        ]
    )

    T0 = (
        np.array([9500, 11000, 12750, 13500, 14750, 14750, 12750, 11250, 10250, 9250])
        / 1e4
    )
    dT0 = np.array([1393, 1028, 1132, 1390, 1341, 1322, 1493, 1125, 1070, 876]) / 1e4

    # gamma and its uncertainty
    gamma = np.array(
        [1.500, 1.425, 1.325, 1.275, 1.250, 1.225, 1.275, 1.350, 1.400, 1.525]
    )
    dgamma = np.array(
        [0.096, 0.133, 0.122, 0.122, 0.109, 0.120, 0.129, 0.108, 0.101, 0.140]
    )

    gal21 = {
        "z": z,
        "mF": Fmean,
        "mF_err": dFmean,
        "T0": T0,
        "T0_err": dT0,
        "gamma": gamma,
        "gamma_err": dgamma,
    }
    # Derived quantities use the same conventions as the IGM model. Error
    # propagation is first order: tau_eff = -ln(mean flux), and thermal
    # broadening is proportional to sqrt(T0).
    gal21["tau_eff"] = -np.log(gal21["mF"])
    gal21["tau_eff_err"] = gal21["mF_err"] / gal21["mF"]
    gal21["sigT_kms"] = thermal_broadening_kms(gal21["T0"] * 1e4)
    gal21["sigT_kms_err"] = (
        gal21["sigT_kms"] * gal21["T0_err"] / (2 * gal21["T0"])
    )

    # Turner 2024
    z_tu24 = np.array(
        [
            2.05,
            2.15,
            2.25,
            2.35,
            2.45,
            2.55,
            2.65,
            2.75,
            2.85,
            2.95,
            3.05,
            3.15,
            3.25,
            3.35,
            3.45,
            3.55,
            3.65,
            3.75,
            3.85,
            3.95,
            4.05,
            4.15,
        ]
    )

    mF_tu24 = np.exp(
        -np.array(
            [
                0.147,
                0.158,
                0.179,
                0.200,
                0.226,
                0.235,
                0.268,
                0.292,
                0.316,
                0.342,
                0.373,
                0.410,
                0.455,
                0.498,
                0.527,
                0.579,
                0.638,
                0.694,
                0.770,
                0.830,
                0.854,
                0.928,
            ]
        )
    )
    emF_tu24 = np.array(
        [
            0.012,
            0.012,
            0.015,
            0.016,
            0.016,
            0.018,
            0.019,
            0.020,
            0.021,
            0.022,
            0.023,
            0.023,
            0.022,
            0.025,
            0.030,
            0.032,
            0.031,
            0.032,
            0.033,
            0.034,
            0.036,
            0.039,
        ]
    )

    tu24 = {"z": z_tu24, "mF": mF_tu24, "mF_err": emF_tu24}
    tu24["tau_eff"] = -np.log(tu24["mF"])
    tu24["tau_eff_err"] = tu24["mF_err"] / tu24["mF"]

    return gal21, tu24

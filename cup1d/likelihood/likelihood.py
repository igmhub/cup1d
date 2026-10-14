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
from cup1d.likelihood import parameter as parameter_space





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
        verbose=False,
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
        verbose : bool, default=False
            Print setup diagnostics from the MPI root rank.
        cov_factor : float or mapping, default=1.0
            Data, systematic, and emulator covariance scaling accepted by the
            configured covariance builder.
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

        self.verbose = verbose
        self.prior_Gauss_rms = prior_Gauss_rms
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
            self.data, k_rebin_factor=args.k_rebin_factor
        )

        self.theory = theory
        # Set inverse covariance. We do it here so we can account for emulator error
        self.set_icov()

        # setup parameters
        self.free_param_names = free_param_names
        self.set_free_parameters(free_param_names, free_param_limits)
        if verbose and (self.rank == 0):
            print(len(self.free_params), "free parameters")

        self.set_Gauss_priors()

        # TBD (should be hanging from mock), true model (when working with mocks)
        self.set_truth()

        # store model
        self.set_model()

        # set blinding
        apply_blinding = False
        seed = 0
        # apply blinding if any of the data sets has apply_blinding set to True
        for key in self.data:
            if self.data[key].apply_blinding:
                apply_blinding = True
                seed = int.from_bytes(
                    self.data[key].blinding.encode("utf-8"), byteorder="big"
                )
                break
        self.blind = blinding.set_blinding(apply_blinding, seed)

        # set IC for likelihood
        if start_from_min and (args.file_ic is not None):
            if os.path.isfile(args.file_ic):
                if self.rank == 0:
                    print("Loading ICs from", args.file_ic)
                if "ic_global" in args.file_ic:
                    if self.rank == 0:
                        print("Setting ICs from global fit")
                    self.set_ic_global(args.file_ic, verbose=True)
                else:
                    if self.rank == 0:
                        print("Setting ICs from at a time fit")
                    self.set_ic_from_z_at_time(args.file_ic, verbose=True)
            else:
                if self.rank == 0:
                    print(
                        f"Initial-condition file not found: {args.file_ic}\n"
                        "Generate at-a-time initial conditions with:\n"
                        "  python scripts/create_at_a_time_initial_conditions.py "
                        "configs/cm2026/variations/at_a_time_global_QMLE3.yaml",
                        flush=True,
                    )

    def set_Gauss_priors(self):
        """Build Gaussian-prior widths aligned with free-parameter order.

        Returns
        -------
        None
            Sets ``Gauss_priors`` to finite widths or ``None`` when no
            Gaussian prior is active.

        Notes
        -----
        ``prior_Gauss_rms`` takes precedence over parameter-specific
        ``Gauss_priors_width`` values. A width of ``1e4`` represents an
        effectively inactive prior and is removed when all widths are inactive.
        """

        self.Gauss_priors = np.ones((len(self.free_params)))
        for ii, (name, parameter) in enumerate(self.free_params.items()):
            if self.prior_Gauss_rms is not None:
                width = parameter["max_value"] - parameter["min_value"]
                _prior = self.prior_Gauss_rms * width
            elif parameter["Gauss_priors_width"] is not None:
                _prior = parameter["Gauss_priors_width"]
            else:
                _prior = 1e4  # so we get zero

            self.Gauss_priors[ii] = _prior

        if np.any(self.Gauss_priors != 1e4):
            pass
        else:
            self.Gauss_priors = None

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
            filename = "l1O_cov_" + self.theory.emulator.emulator_label + ".npy"
            if "forest" in self.theory.emulator.emulator_label:
                import forestflow

                covariance_root = os.path.join(
                    os.path.dirname(forestflow.__path__[0]), "data", "covariance"
                )
            else:
                covariance_root = os.path.join(
                    get_path_repo("lace"), "data", "covariance"
                )
            full_path = os.path.join(covariance_root, filename)
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

                # initialize emulator covariance
                add_emu_cov_kms = np.zeros((k_Mpc.shape[0], k_Mpc.shape[0]))

                # find closest z in cov
                ind0 = np.argmin(np.abs(emu_cov["zz_zk"] - data.z[ii]))
                # get cov from closest z
                ind = np.argwhere(emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0])[:, 0]
                # block diagonal for that redshift
                _emu_cov = emu_cov["cov_zk"][ind, :][:, ind]
                _k_Mpc = emu_cov["k_Mpc_zk"][ind]

                # rescale covariance matrix by power spectrum,
                # since the covariance matrix stores the relative error
                for i0 in range(k_Mpc.shape[0]):
                    # get closest k in emu cov matrix
                    j0 = np.argmin(np.abs(k_Mpc[i0] - _k_Mpc))
                    for i1 in range(k_Mpc.shape[0]):
                        # skip if diagonal and i0 != i1
                        if (self.emu_cov_type == "diagonal") and (i0 != i1):
                            continue
                        # get closest k in emu cov matrix
                        j1 = np.argmin(np.abs(k_Mpc[i1] - _k_Mpc))
                        add_emu_cov_kms[i0, i1] = (
                            _emu_cov[j0, j1]
                            * pksmooth[ii][i0]
                            * pksmooth[ii][i1]
                            * emu_cov_factor
                        )

                        cov[i0, i1] += add_emu_cov_kms[i0, i1]

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
                    full_emu_cov = np.zeros_like(cov)
                    for i0 in range(cov.shape[0]):
                        dkms_dMpc = fiducial_conversion(data.full_zs[i0])
                        full_k_kms0 = data.full_k_kms[i0] * dkms_dMpc

                        # find closest z in cov
                        ind0 = np.argmin(np.abs(emu_cov["zz_zk"] - data.full_zs[i0]))
                        ind = np.argwhere(emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0])[
                            :, 0
                        ]
                        # find closest k for such z
                        ind1 = np.argmin(
                            np.abs(emu_cov["k_Mpc_zk"][ind] - full_k_kms0)
                        )
                        # closest index in z and k
                        j0 = ind[ind1]

                        # index to inflate
                        ind0_infl = np.argmin(
                            np.abs(self.cov_factor["z"] - data.full_zs[i0])
                        )

                        for i1 in range(cov.shape[0]):
                            dkms_dMpc = fiducial_conversion(data.full_zs[i1])
                            full_k_kms1 = data.full_k_kms[i1] * dkms_dMpc

                            # find closest z in cov
                            ind0 = np.argmin(
                                np.abs(emu_cov["zz_zk"] - data.full_zs[i1])
                            )
                            ind = np.argwhere(
                                emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0]
                            )[:, 0]
                            # find closest k for such z
                            ind1 = np.argmin(
                                np.abs(emu_cov["k_Mpc_zk"][ind] - full_k_kms1)
                            )
                            # closest index in z and k
                            j1 = ind[ind1]

                            # index to inflate
                            ind1_infl = np.argmin(
                                np.abs(self.cov_factor["z"] - data.full_zs[i1])
                            )

                            full_emu_cov[i0, i1] = (
                                emu_cov["cov_zk"][j0, j1]
                                * data.full_Pk_kms[i0]
                                * data.full_Pk_kms[i1]
                                * self.cov_factor["val_emu"][ind0_infl]
                                * self.cov_factor["val_emu"][ind1_infl]
                            )

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

    def cosmology_params(self, parameters):
        """Extract supported cosmological values from a physical parameter map.

        Parameters
        ----------
        parameters : mapping
            Physical values keyed by parameter name.

        Returns
        -------
        dict
            Present values among ``ombh2``, ``omch2``, ``cosmomc_theta``,
            ``As``, ``ns``, ``mnu``, and ``nrun``.

        Raises
        ------
        ValueError
            If none of the supported cosmology names is present.
        """

        names = {"ombh2", "omch2", "cosmomc_theta", "As", "ns", "mnu", "nrun"}
        cosmo_dict = {
            name: value for name, value in parameters.items() if name in names
        }
        if not cosmo_dict:
            raise ValueError("No cosmology parameters found")
        return cosmo_dict

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

    def get_p1d_kms(
        self,
        parameters=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
        apply_hull=True,
        remove=None,
    ):
        """Compute rebinned theoretical P1D predictions for every data set.

        Parameters
        ----------
        parameters : mapping, optional
            Full or partial named physical point; omitted values use defaults.
        return_covar, return_blob, return_emu_params : bool, default=False
            Request corresponding auxiliary theory outputs.
        apply_hull : bool, default=True
            Apply emulator hull/domain admission checks.
        remove : sequence of str, optional
            Contaminant contributions omitted by theory.

        Returns
        -------
        predictions, auxiliary : tuple or None
            Re-binned P1D rows keyed by dataset label and requested auxiliary
            outputs. ``None`` signals an inadmissible theory prediction.

        Notes
        -----
        ForestFlow calls are primed across the union of requested redshifts
        and its cache is cleared before returning, including rejection paths.
        """

        # Public callers supply a full named point; theory receives scalar
        # values through this private boundary adapter.
        like_params = (
            {} if parameters is None
            else parameter_space.values_from_point(self.free_params, parameters)
        )

        all_p1ds = {}
        other_stuff = {}
        forest_emulator = self.theory.emulator
        use_forest_cache = "forest" in forest_emulator.emulator_label
        if use_forest_cache:
            # Several data sets commonly contain identical redshifts. Prime
            # ForestFlow once for their union instead of repeating cINN calls.
            emulator_calls = [
                self.theory.get_emulator_calls(
                    self.Rebin_data.zs[key], like_params=like_params
                )[0]
                for key in self.Rebin_data.zs
            ]
            forest_emulator.prime_prediction_cache(emulator_calls)
        for key in self.Rebin_data.zs:
            _results = self.theory.get_P1D_kms(
                self.Rebin_data.zs[key],
                self.Rebin_data.k_kms[key],
                like_params=like_params,
                return_covar=return_covar,
                return_blob=return_blob,
                return_emu_params=return_emu_params,
                apply_hull=apply_hull,
                remove=remove,
            )
            if _results is None:
                if use_forest_cache:
                    forest_emulator.clear_prediction_cache()
                return None

            if return_blob | return_emu_params:
                p1ds = _results[0]
            else:
                p1ds = _results

            all_p1ds[key] = self.Rebin_data.rebinning(key, p1ds)

            other_stuff[key] = []
            if return_blob | return_emu_params:
                for ii in range(1, len(_results)):
                    other_stuff[key].append(_results[ii])

        if use_forest_cache:
            forest_emulator.clear_prediction_cache()
        return all_p1ds, other_stuff

    def get_chi2(self, parameters=None, return_all=False, zmask=None):
        """Compute chi2 using data and theory, without emulator covariance.

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

        chi2_eachz = {}
        for key in log_like_all:
            chi2_eachz[key] = -2.0 * log_like_all[key]

        chi2_total = -2.0 * log_like

        if return_all:
            return chi2_total, chi2_eachz
        else:
            return chi2_total

    def get_log_like(
        self,
        parameters=None,
        ignore_log_det_cov=True,
        return_blob=False,
        zmask=None,
    ):
        """Compute P1D Gaussian log likelihood and per-redshift contributions.

        Parameters
        ----------
        parameters : mapping, optional
            Physical free-parameter values.
        ignore_log_det_cov : bool, default=True
            Omit covariance normalization determinants.
        return_blob : bool, default=False
            Include the theory blob associated with the prediction.
        zmask : float or array-like, optional
            Exactly one diagnostic redshift, or ``None`` for the joint fit.

        Returns
        -------
        list
            ``[log_like, log_like_all]`` and, when requested,
            ``[log_like, log_like_all, blob]``. Rejected predictions return
            negative infinities (and a zero blob when requested).

        A non-null ``zmask`` may select exactly one redshift. This is for
        one-redshift diagnostic fits: selecting a subset omits cross-redshift
        covariance, so multi-redshift masks are rejected.
        """

        zmask = self._validate_single_redshift_mask(zmask)

        # what to return if we are out of priors
        null_out = [-np.inf, -np.inf]
        if return_blob:
            blob = (0, 0, 0, 0, 0, 0)
            null_out.append(blob)

        # evaluate model in physical parameter space
        _res = self.get_p1d_kms(parameters, return_blob=return_blob)
        if _res is None:
            return null_out
        else:
            if return_blob:
                emu_p1d, extra = _res
                blob = next(iter(extra.values()))[0]
            else:
                emu_p1d = _res[0]

        # compute log like contribution from each sample and redshift bin
        log_like_all = {}
        log_like = 0
        for key in self.Rebin_data.zs:

            log_like_all[key] = np.zeros((self.Rebin_data.zs[key].shape[0]))

            emu_p1d_use = emu_p1d[key]
            data = self.data[key]
            icov_Pk_kms = self.icov_Pk_kms[key]
            chol_Pk_kms = self.chol_Pk_kms[key]
            full_icov_Pk_kms = self.full_icov_Pk_kms[key]
            full_chol_Pk_kms = self.full_chol_Pk_kms.get(key)

            # loop over redshift bins
            for iz in range(len(data.z)):
                if zmask is not None:
                    ind = np.argwhere(np.abs(zmask - data.z[iz]) < 1e-3)
                    if len(ind) == 0:
                        continue
                # compute chi2 for this redshift bin
                diff = data.Pk_kms[iz] - np.array(emu_p1d_use[iz]).reshape(-1)
                if self.covariance_method == "cholesky":
                    solved = cho_solve(chol_Pk_kms[iz], diff, check_finite=False)
                    chi2_z = np.dot(diff, solved)
                else:
                    chi2_z = np.dot(np.dot(icov_Pk_kms[iz], diff), diff)
                # print(iz, chi2_z, np.mean(icov_Pk_kms[iz]), np.mean(diff))
                # print(
                #     np.dot(icov_Pk_kms[iz], diff),
                # )
                # print(iz, chi2_z)
                # check whether to add determinant of covariance as well
                if ignore_log_det_cov:
                    log_like_all[key][iz] = -0.5 * chi2_z
                else:
                    log_det_cov = (2 * np.log(np.diag(chol_Pk_kms[iz][0])).sum()
                                   if self.covariance_method == "cholesky"
                                   else np.log(np.abs(1 / np.linalg.det(icov_Pk_kms[iz]))))
                    log_like_all[key][iz] = -0.5 * (chi2_z + log_det_cov)

            if (full_icov_Pk_kms is None) | (zmask is not None):
                log_like += np.sum(log_like_all[key])
            else:
                # compute chi2 using full cov
                diff = data.full_Pk_kms - np.concatenate(emu_p1d_use)
                if self.covariance_method == "cholesky":
                    solved = cho_solve(full_chol_Pk_kms, diff, check_finite=False)
                    chi2_all = np.dot(diff, solved)
                else:
                    chi2_all = np.dot(np.dot(full_icov_Pk_kms, diff), diff)
                if ignore_log_det_cov:
                    log_like += -0.5 * chi2_all
                else:
                    log_det_cov = (2 * np.log(np.diag(full_chol_Pk_kms[0])).sum()
                                   if self.covariance_method == "cholesky"
                                   else np.log(np.abs(1 / np.linalg.det(full_icov_Pk_kms))) )
                    log_like += -0.5 * (chi2_all + log_det_cov)

        # something went wrong
        if np.isnan(log_like):
            return null_out

        out = [log_like, log_like_all]
        if return_blob:
            out.append(blob)
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
        fiducial = np.asarray(
            [parameter["value"] for parameter in self.free_params.values()]
        )
        values = np.asarray([parameters[name] for name in self.free_params])
        return -np.sum(
            (fiducial - values) ** 2 / (2 * self.Gauss_priors**2)
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

    def _parameter_batch_rows(self, parameters_batch):
        """Validate a parameter batch and expose scalar compatibility rows.

        Parameters
        ----------
        parameters_batch : mapping or iterable of mapping
            Preferred columnar mapping of one-dimensional arrays, or legacy
            iterable of scalar physical parameter mappings.

        Returns
        -------
        list of dict
            Scalar parameter mappings in batch order.

        Raises
        ------
        ValueError
            If a column is not one-dimensional or columns have different
            batch lengths.

        New callers should provide ``{name: array(n_batch)}``, which avoids
        creating parameter dictionaries in the sampler/fitter layer.  The
        scalar rows are retained temporarily because cosmology and several
        legacy contaminant models have scalar-only APIs.
        """

        if isinstance(parameters_batch, dict):
            if not parameters_batch:
                return []
            arrays = {}
            n_batch = None
            for name, values in parameters_batch.items():
                values = np.asarray(values, dtype=float)
                if values.ndim != 1:
                    raise ValueError(
                        f"batched parameter {name} must have shape (n_batch,), "
                        f"got {values.shape}"
                    )
                if n_batch is None:
                    n_batch = len(values)
                elif len(values) != n_batch:
                    raise ValueError(
                        f"batched parameter {name} has length {len(values)}, "
                        f"expected {n_batch}"
                    )
                arrays[name] = values
            return [
                {name: values[index] for name, values in arrays.items()}
                for index in range(n_batch)
            ]
        return list(parameters_batch)

    def log_prob_and_blobs_batch(
        self, parameters_batch, ignore_log_det_cov=True, zmask=None
    ):
        """Evaluate posterior values and blobs for a physical-parameter batch.

        Parameters
        ----------
        parameters_batch : mapping or iterable of mapping
            Preferred columnar mapping with arrays of shape ``(n_batch,)``, or
            legacy scalar parameter mappings.
        ignore_log_det_cov : bool, default=True
            Omit covariance normalization determinants.
        zmask : float or array-like, optional
            One diagnostic redshift or ``None`` for joint fitting.

        Returns
        -------
        list of tuple
            One tuple per input point: posterior log probability followed by
            theory blob values. Out-of-bounds or rejected points receive the
            configured posterior floor.

        Emulator calls are coalesced before model evaluation. Predictions are
        then stacked so covariance contractions are evaluated over the entire
        batch with NumPy rather than one point at a time.
        """

        zmask = self._validate_single_redshift_mask(zmask)
        parameter_columns = parameters_batch if isinstance(parameters_batch, dict) else None
        parameters_batch = self._parameter_batch_rows(parameters_batch)
        n_points = len(parameters_batch)
        in_bounds = np.asarray(
            [self.parameters_in_bounds(parameters) for parameters in parameters_batch]
        )
        valid_indices = np.flatnonzero(in_bounds)
        blobs = [self.theory.get_blob()] * n_points
        batched_predictions = None
        # Columnar callers use one complete forward-model evaluation per data
        # group. Legacy list-of-dictionaries callers retain the compatibility
        # path below.
        if parameter_columns is not None and len(valid_indices):
            valid_columns = {name: np.asarray(values)[in_bounds] for name, values in parameter_columns.items()}
            batched_predictions = {}
            for key, redshifts in self.Rebin_data.zs.items():
                fine_prediction = self.theory.get_p1d_kms(
                    redshifts, self.Rebin_data.k_kms[key], valid_columns
                )
                batched_predictions[key] = self.Rebin_data.rebinning_batch(key, fine_prediction)
            first_redshifts = next(iter(self.Rebin_data.zs.values()))
            _, _, valid_blobs = self.theory.get_emulator_calls(
                first_redshifts, valid_columns, return_M_of_z=True, return_blob=True
            )
            for local_index, index in enumerate(valid_indices):
                blobs[index] = tuple(valid_blobs[local_index])
        else:
            predictions = [None] * n_points
            for index, parameters in enumerate(parameters_batch):
                if not in_bounds[index]:
                    continue
                result = self.get_p1d_kms(parameters, return_blob=True)
                if result is None:
                    continue
                predictions[index] = result[0]
                extra = result[1]
                blobs[index] = next(iter(extra.values()))[0]
            valid_indices = np.flatnonzero([prediction is not None for prediction in predictions])

        log_like = np.full(n_points, self.min_log_like, dtype=float)
        if len(valid_indices):
            batch_log_like = np.zeros(len(valid_indices))
            for key in self.Rebin_data.zs:
                data = self.data[key]
                inverse_covariance = self.icov_Pk_kms[key]
                chol_covariance = self.chol_Pk_kms[key]
                full_inverse_covariance = self.full_icov_Pk_kms[key]
                full_chol_covariance = self.full_chol_Pk_kms.get(key)
                if full_inverse_covariance is not None and zmask is None:
                    model = (
                        np.concatenate(batched_predictions[key], axis=1)
                        if batched_predictions is not None
                        else np.stack([np.concatenate(predictions[index][key]) for index in valid_indices])
                    )
                    residual = data.full_Pk_kms[None, :] - model
                    chi2 = (np.sum(residual * cho_solve(full_chol_covariance, residual.T, check_finite=False).T, axis=1)
                            if self.covariance_method == "cholesky" else np.einsum("bi,ij,bj->b", residual, full_inverse_covariance, residual, optimize=True))
                    batch_log_like -= 0.5 * chi2
                    if not ignore_log_det_cov:
                        batch_log_like -= 0.5 * np.log(
                            np.abs(1 / np.linalg.det(full_inverse_covariance))
                        )
                    continue

                for redshift_index, redshift in enumerate(data.z):
                    if zmask is not None and not np.any(
                        np.abs(zmask - redshift) < 1.0e-3
                    ):
                        continue
                    model = (
                        batched_predictions[key][redshift_index]
                        if batched_predictions is not None
                        else np.stack([predictions[index][key][redshift_index] for index in valid_indices])
                    )
                    residual = data.Pk_kms[redshift_index][None, :] - model
                    chi2 = (np.sum(residual * cho_solve(chol_covariance[redshift_index], residual.T, check_finite=False).T, axis=1)
                            if self.covariance_method == "cholesky" else np.einsum("bi,ij,bj->b", residual, inverse_covariance[redshift_index], residual, optimize=True))
                    batch_log_like -= 0.5 * chi2
                    if not ignore_log_det_cov:
                        batch_log_like -= 0.5 * np.log(
                            np.abs(
                                1
                                / np.linalg.det(
                                    inverse_covariance[redshift_index]
                                )
                            )
                        )
            log_like[valid_indices] = np.maximum(
                batch_log_like, self.min_log_like
            )

        log_prior = np.asarray(
            [self.get_log_prior(parameters) for parameters in parameters_batch]
        )
        posterior = log_like + log_prior
        posterior[~in_bounds] = self.min_log_like
        return [
            (posterior[index], *blobs[index]) for index in range(n_points)
        ]

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
            iname = np.argwhere(np.array(dir_out["pnames"][iz]) == (pname + "_0"))[0, 0]
            parameter["value"] = dir_out["mle"][iz][pname + "_0"]

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
            NumPy result dictionary containing fitted coefficient histories.
        verbose : bool, default=True
            Print assigned physical values on the MPI root rank.

        Returns
        -------
        None
            Updates configured fixed flags, interpolates coefficient values at
            model nodes, and resets contaminant/IGM coefficient models.

        Raises
        ------
        ValueError
            If a requested parameter has no matching model-node configuration.
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
            else:
                _z = dir_out[pname]["z"]
                _val = dir_out[pname]["val"]
                parameter["value"] = np.interp(znode, _z, _val)

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

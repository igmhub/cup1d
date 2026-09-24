import numpy as np
import os
import math
import copy
from mpi4py import MPI
from scipy.stats.distributions import chi2 as chi2_scipy
from scipy.optimize import minimize
from scipy.linalg import block_diag

from lace.cosmo.thermal_broadening import thermal_broadening_kms
from cup1d.utils.utils import is_number_string
from cup1d.utils import rebinning

from cup1d.utils.utils import split_string
from cup1d.utils.utils import get_path_repo
from cup1d.utils import blinding





class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

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
        min_log_like=-1e100,
        args=None,
        start_from_min=True,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - free_param_names is a list of param names, in any order
        - free_param_limits list of tuples, same order than free_param_names
        - if prior_Gauss_rms is None it will use uniform priors
        - ignore k-bins with k > kmin_kms
        - cov_factor adjusts the contribution from data covariance
        - emu_cov_factor adjusts the contribution from emulator covariance
        set between 0 and 1.
        - extra_p1d_data: extra P1D data, e.g., from HIRES
        - min_log_like: use this instead of - infinity"""

        self.rank = MPI.COMM_WORLD.Get_rank()

        self.verbose = verbose
        self.prior_Gauss_rms = prior_Gauss_rms
        self.cov_factor = cov_factor
        self.emu_cov_type = emu_cov_type
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
        """
        Sets Gaussian priors on the parameters
        """

        self.Gauss_priors = np.ones((len(self.free_params)))
        for ii, par_like in enumerate(self.free_params):
            if self.prior_Gauss_rms is not None:
                _prior = self.prior_Gauss_rms
            elif par_like.Gauss_priors_width is not None:
                _fid = par_like.value
                _width = par_like.Gauss_priors_width
                _low = par_like.get_value_in_cube(_fid - 0.5 * _width)
                _high = par_like.get_value_in_cube(_fid + 0.5 * _width)
                _prior = _high - _low
            else:
                _prior = 1e4  # so we get zero

            self.Gauss_priors[ii] = _prior

        if np.any(self.Gauss_priors != 1e4):
            pass
        else:
            self.Gauss_priors = None

    def set_icov(self):
        """
        Computes and sets the inverse covariance matrix for the P1 power spectrum data and full power spectrum data.

        This method processes the main dataset (`data`) and any additional dataset (`extra_data`) associated
        with the object. For each dataset:
        - It computes the inverse covariance matrices for the power spectrum (`Pk_kms`) at different redshifts,
          incorporating an emulator error factor.
        - It computes the inverse covariance matrix for the full power spectrum data, if available.

        The resulting inverse covariance matrices are stored in instance attributes.

        Attributes Modified:
        --------------------
        icov_Pk_kms : list of numpy.ndarray
            List of inverse covariance matrices for the power spectrum of the main dataset at different redshifts.

        full_icov_Pk_kms : numpy.ndarray or None
            Inverse covariance matrix for the full power spectrum of the main dataset.
            Set to `None` if the full power spectrum is not available.

        extra_icov_Pk_kms : list of numpy.ndarray
            List of inverse covariance matrices for the power spectrum of the additional dataset at different redshifts.
            Set to `None` if `extra_data` is not provided.

        extra_full_icov_Pk_kms : numpy.ndarray or None
            Inverse covariance matrix for the full power spectrum of the additional dataset.
            Set to `None` if the full power spectrum is not available or if `extra_data` is not provided.

        Notes:
        -----
        - The emulator error is added to the diagonal of the covariance matrix before inverting. The error is
          computed as `(data.Pk_kms * emu_cov_factor) ** 2`, where `emu_cov_factor` is an attribute of the object.
        - The method iterates over redshift bins (`data.z`) and processes the covariance matrices accordingly.
        - If the dataset (`data` or `extra_data`) is `None`, no processing occurs for that dataset.

        Raises:
        -------
        ValueError:
            If the covariance matrix inversion fails (e.g., due to singularity).
        """

        # get emulator error
        filename = "l1O_cov_" + self.theory.emulator.emulator_label + ".npy"
        full_path = os.path.join(get_path_repo("lace"), "data", "covariance", filename)
        emu_cov = np.load(full_path, allow_pickle=True).item()
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
        self.cov_Pk_kms = {}
        self.cov_emu_Pk_kms = {}
        # all redshifts together, for full Pk
        self.full_icov_Pk_kms = {}
        self.full_cov_Pk_kms = {}
        self.emu_full_cov_Pk_kms = {}

        # Iterate over both datasets: main dataset (idata = 0) and additional dataset (idata = 1)
        for key in self.data:
            data = self.data[key]
            # initialize, to store the results for different redshifts
            icov_Pk_kms = []
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
                dkms_dMpc = self.theory.fid_cosmo["cosmo"].get_dkms_dMpc(data.z[ii])
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

                # Compute and store the inverse covariance matrix
                icov_Pk_kms.append(np.linalg.inv(cov))
                cov_Pk_kms.append(cov)
                cov_emu_Pk_kms.append(add_emu_cov_kms)

            self.icov_Pk_kms[key] = icov_Pk_kms
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
                        dkms_dMpc = self.theory.fid_cosmo["cosmo"].get_dkms_dMpc(
                            data.full_zs[i0]
                        )
                        full_k_kms0 = data.full_k_kms[i0] * dkms_dMpc

                        # find closest z in cov
                        ind0 = np.argmin(np.abs(emu_cov["zz_zk"] - data.full_zs[i0]))
                        ind = np.argwhere(emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0])[
                            :, 0
                        ]
                        # find closest k for such z
                        ind1 = np.argmin(np.abs(emu_cov["k_Mpc_zk"] - full_k_kms0))
                        # closest index in z and k
                        j0 = ind[ind1]

                        # index to inflate
                        ind0_infl = np.argmin(
                            np.abs(self.cov_factor["z"] - data.full_zs[i0])
                        )

                        for i1 in range(cov.shape[0]):
                            dkms_dMpc = self.theory.fid_cosmo["cosmo"].get_dkms_dMpc(
                                data.full_zs[i1]
                            )
                            full_k_kms1 = data.full_k_kms[i1] * dkms_dMpc

                            # find closest z in cov
                            ind0 = np.argmin(
                                np.abs(emu_cov["zz_zk"] - data.full_zs[i1])
                            )
                            ind = np.argwhere(
                                emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0]
                            )[:, 0]
                            # find closest k for such z
                            ind1 = np.argmin(np.abs(emu_cov["k_Mpc_zk"] - full_k_kms1))
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

                # Compute and store the inverse covariance matrix
                self.full_icov_Pk_kms[key] = np.linalg.inv(cov)
                self.full_cov_Pk_kms[key] = cov
                self.emu_full_cov_Pk_kms[key] = full_emu_cov

    def set_free_parameters(self, free_param_names, free_param_limits):
        """Setup likelihood parameters that we want to vary"""

        if free_param_limits is not None:
            assert len(free_param_limits) == len(
                free_param_names
            ), "wrong number of parameter limits"

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        ## select free parameters, make sure ordering
        ## in self.free_params is same as in free_param_names
        # for par in params:
        #     if par.name not in free_param_names:
        #         print(par.name)

        # setup list of likelihood free parameters
        self.free_params = []
        # iterate over free parameters
        for par in free_param_names:
            found = False
            for p in params:
                if p.name == par:
                    if free_param_limits is not None:
                        ## Set min and max of each parameter if
                        ## a list is given. otherwise leave as default
                        ind = free_param_names.index(par.name)
                        par.min_value = free_param_limits[ind][0]
                        par.max_value = free_param_limits[ind][1]
                    self.free_params.append(p)
                    found = True
                    break
            if found == False:
                raise ValueError(
                    "Could not find free parameter {} in theory".format(par)
                )

        if self.verbose and (self.rank == 0):
            print("likelihood setup with {} free parameters".format(Nfree))

        return

    def sampling_point_from_parameters(self):
        """Translate likelihood parameters to array of values (in cube)"""

        values = np.zeros(len(self.free_params))
        for ii, par in enumerate(self.free_params):
            values[ii] = par.value_in_cube()

        return values

    def parameters_from_sampling_point(self, values):
        """Translate input array of values (in cube) to likelihood parameters"""

        if values is None:
            return []

        assert len(values) == len(self.free_params), "size mismatch"
        Npar = len(values)
        like_params = []
        for ip in range(Npar):
            par = self.free_params[ip].get_new_parameter(values[ip])
            like_params.append(par)

        return like_params

    def cosmology_params_from_sampling_point(self, values):
        """For a given point in sampling space, return a list of
        cosmology params"""

        like_params = self.parameters_from_sampling_point(values)

        ## Dictionary of cosmology parameters
        cosmo_dict = {}

        for like_param in like_params:
            if like_param.name == "ombh2":
                cosmo_dict["ombh2"] = like_param.value
            elif like_param.name == "omch2":
                cosmo_dict["omch2"] = like_param.value
            elif like_param.name == "cosmomc_theta":
                cosmo_dict["cosmomc_theta"] = like_param.value
            elif like_param.name == "As":
                cosmo_dict["As"] = like_param.value
            elif like_param.name == "ns":
                cosmo_dict["ns"] = like_param.value
            elif like_param.name == "mnu":
                cosmo_dict["mnu"] = like_param.value
            elif like_param.name == "nrun":
                cosmo_dict["nrun"] = like_param.value

        assert len(cosmo_dict) > 0, "No cosmology parameters found in sampling space"

        return cosmo_dict

    def set_truth(self):
        """Store true cosmology from the simulation used to make mock data"""

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
        for par in self.free_params:
            if (
                ("tau" in par.name)
                | ("sigT" in par.name)
                | ("gamma" in par.name)
                | ("kF" in par.name)
            ):
                if equal_IGM:
                    if "tau" in par.name:
                        self.truth["like_params"][par.name] = 1
                        self.truth["like_params_cube"][par.name] = (
                            par.get_value_in_cube(self.truth["like_params"][par.name])
                        )
                    else:
                        self.truth["like_params"][par.name] = 0
                        self.truth["like_params_cube"][par.name] = (
                            par.get_value_in_cube(self.truth["like_params"][par.name])
                        )
                else:
                    self.truth["like_params"][par.name] = np.infty
                    self.truth["like_params_cube"][par.name] = np.infty
            elif (par.name == "As") | (par.name == "ns") | (par.name == "nrun"):
                self.truth["like_params"][par.name] = self.truth["cosmo"][par.name]
                self.truth["like_params_cube"][par.name] = par.get_value_in_cube(
                    self.truth["like_params"][par.name]
                )
                self.truth["like_params"][pname2[par.name]] = self.truth["linP"][
                    pname2[par.name]
                ]
            # else:
            #     if par.name not in self.truth["cont"]:
            #         print("could not find {} in truth".format(par.name))
            #         continue
            #     self.truth["like_params"][par.name] = self.truth["cont"][
            #         par.name
            #     ]
            #     self.truth["like_params_cube"][
            #         par.name
            #     ] = par.get_value_in_cube(self.truth["cont"][par.name])

    def set_model(self):
        """Store fiducial cosmology assumed for the fit"""

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
        for par in self.free_params:
            self.fid["fit"][par.name] = par.value
            self.fid["fit_cube"][par.name] = par.get_value_in_cube(par.value)
            if (par.name == "As") | (par.name == "ns") | (par.name == "nrun"):
                self.fid["fit"][pname2[par.name]] = blob[pname2[par.name]]
                self.fid["linP"][pname2[par.name]] = blob[pname2[par.name]]

    def get_p1d_kms(
        self,
        values=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
        apply_hull=True,
        remove=None,
    ):
        """Compute theoretical prediction for P1D"""

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params = self.parameters_from_sampling_point(values)
        else:
            like_params = []

        all_p1ds = {}
        other_stuff = {}
        for key in self.Rebin_data.zs:
            _results = self.theory.get_p1d_kms(
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

        return all_p1ds, other_stuff

    def get_chi2(self, values=None, return_all=False, zmask=None):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""

        log_like, log_like_all = self.get_log_like(
            values, ignore_log_det_cov=True, zmask=zmask
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
        values=None,
        ignore_log_det_cov=True,
        return_blob=False,
        zmask=None,
    ):
        """Compute log(likelihood), including determinant of covariance
        unless you are setting ignore_log_det_cov=True."""

        # what to return if we are out of priors
        null_out = [-np.inf, -np.inf]
        if return_blob:
            blob = (0, 0, 0, 0, 0, 0)
            null_out.append(blob)

        # check that we are within unit cube
        if values is not None:
            if (values > 1.0).any() | (values < 0.0).any():
                return null_out

        # evaluate model
        _res = self.get_p1d_kms(values, return_blob=return_blob)
        if _res is None:
            return null_out
        else:
            if return_blob:
                emu_p1d, blob = _res
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
            full_icov_Pk_kms = self.full_icov_Pk_kms[key]

            # loop over redshift bins
            for iz in range(len(data.z)):
                if zmask is not None:
                    ind = np.argwhere(np.abs(zmask - data.z[iz]) < 1e-3)
                    if len(ind) == 0:
                        continue
                # compute chi2 for this redshift bin
                diff = data.Pk_kms[iz] - np.array(emu_p1d_use[iz]).reshape(-1)
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
                    log_det_cov = np.log(np.abs(1 / np.linalg.det(icov_Pk_kms[iz])))
                    log_like_all[key][iz] = -0.5 * (chi2_z + log_det_cov)

            if (full_icov_Pk_kms is None) | (zmask is not None):
                log_like += np.sum(log_like_all[key])
            else:
                # compute chi2 using full cov
                diff = data.full_Pk_kms - np.concatenate(emu_p1d_use)
                chi2_all = np.dot(np.dot(full_icov_Pk_kms, diff), diff)
                if ignore_log_det_cov:
                    log_like += -0.5 * chi2_all
                else:
                    log_det_cov = np.log(np.abs(1 / np.linalg.det(full_icov_Pk_kms)))
                    log_like += -0.5 * (chi2_all + log_det_cov)

        # something went wrong
        if np.isnan(log_like):
            return null_out

        out = [log_like, log_like_all]
        if return_blob:
            out.append(blob)
        return out

    def regulate_log_like(self, log_like):
        """Make sure that log_like is not NaN, nor tiny"""

        if (log_like is None) or math.isnan(log_like):
            return self.min_log_like

        return max(self.min_log_like, log_like)

    def compute_log_prob(
        self, values, return_blob=False, ignore_log_det_cov=True, zmask=None
    ):
        """Compute log likelihood plus log priors for input values
        - if return_blob==True, it will return also extra information"""

        # Always force parameter to be within range (for now)
        if (np.max(values) > 1.0) or (np.min(values) < 0.0):
            if return_blob:
                dummy_blob = self.theory.get_blob()
                return self.min_log_like, dummy_blob
            else:
                return self.min_log_like

        # compute log_prior
        if self.Gauss_priors is not None:
            log_prior = self.get_log_prior(values)
        else:
            log_prior = 0

        # compute log_like (option to ignore emulator covariance)
        if return_blob:
            log_like, chi2_all, blob = self.get_log_like(
                values,
                ignore_log_det_cov=ignore_log_det_cov,
                return_blob=True,
                zmask=zmask,
            )
        else:
            log_like, chi2_all = self.get_log_like(
                values,
                ignore_log_det_cov=ignore_log_det_cov,
                return_blob=False,
                zmask=zmask,
            )

        # regulate log-like (not NaN, not tiny)
        log_like = self.regulate_log_like(log_like)

        if return_blob:
            return log_like + log_prior, blob
        else:
            return log_like + log_prior

    def log_prob(self, values, ignore_log_det_cov=True, zmask=None):
        """Return log likelihood plus log priors"""

        return self.compute_log_prob(
            values,
            return_blob=False,
            ignore_log_det_cov=ignore_log_det_cov,
            zmask=zmask,
        )

    def log_prob_and_blobs(self, values, ignore_log_det_cov=True, zmask=None):
        """Function used by emcee to get both log_prob and extra information"""

        lnprob, blob = self.compute_log_prob(
            values,
            return_blob=True,
            ignore_log_det_cov=ignore_log_det_cov,
            zmask=zmask,
        )
        # unpack tuple
        out = lnprob, *blob
        return out

    def get_log_prior(self, values):
        """Compute logarithm of prior"""

        assert len(values) == len(self.free_params), "size mismatch"

        # Always force parameter to be within range (for now)
        if max(values) > 1:
            return self.min_log_like
        if min(values) < 0:
            return self.min_log_like

        fid_values = [p.value_in_cube() for p in self.free_params]
        log_prior = -np.sum(
            (np.array(fid_values) - values) ** 2 / (2 * self.Gauss_priors**2)
        )
        return log_prior

    def minus_log_prob(self, values, zmask=None, ind_fix=None, pfix=None):
        """Return minus log_prob (needed to maximise posterior)"""

        if ind_fix is not None:
            values[ind_fix] = pfix

        return -1.0 * self.log_prob(values, zmask=zmask)

    def maximise_posterior(self, initial_values=None, method="nelder-mead", tol=1e-4):
        """Run scipy minimizer to find maximum of posterior"""

        if not initial_values:
            initial_values = np.ones(len(self.free_params)) * 0.5

        return minimize(self.minus_log_prob, x0=initial_values, method=method, tol=tol)

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
        """Delegate to :func:`cup1d.postprocessing.likelihood.plot_p1d`."""
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
        """Delegate to :func:`cup1d.postprocessing.likelihood.plot_p1d_errors`."""
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
        """Delegate to :func:`cup1d.postprocessing.contaminants.plot_hcd_cont`."""
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
        """Delegate to :func:`cup1d.postprocessing.contaminants.plot_metal_cont_add`."""
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
        """Delegate to :func:`cup1d.postprocessing.contaminants.plot_metal_cont_mult`."""
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
        """Delegate to :func:`cup1d.postprocessing.igm.plot_likelihood_igm`."""
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
        """Delegate to :func:`cup1d.postprocessing.likelihood.plot_cov_terms`."""
        from cup1d.postprocessing.likelihood import plot_cov_terms as _plot

        return _plot(self, save_directory)

    def plot_cov_to_pk(
        self, use_pk_smooth=True, fname=None, ftsize=18, store_data=False
    ):
        """Delegate to :func:`cup1d.postprocessing.likelihood.plot_cov_to_pk`."""
        from cup1d.postprocessing.likelihood import plot_cov_to_pk as _plot

        return _plot(self, use_pk_smooth, fname, ftsize, store_data)

    def plot_correlation_matrix(self, save_directory=None):
        """Delegate to :func:`cup1d.postprocessing.likelihood.plot_correlation_matrix`."""
        from cup1d.postprocessing.likelihood import plot_correlation_matrix as _plot

        return _plot(self, save_directory)

    def plot_hull_fid(self, like_params=[]):
        """Delegate to :func:`cup1d.postprocessing.likelihood.plot_hull_fid`."""
        from cup1d.postprocessing.likelihood import plot_hull_fid as _plot

        return _plot(self, like_params)

    def set_ic_from_z_at_time(self, fname, verbose=True):
        """Set the initial conditions for the likelihood from a fit"""

        dir_out = np.load(fname, allow_pickle=True).item()

        # make a copy of free params, and set their values to the best-fit
        free_params = self.free_params.copy()
        for jj, p in enumerate(free_params):
            if p.name in ["As", "ns"]:
                continue
            pname, iistr = split_string(p.name)
            ii = int(iistr)

            if (pname + "_znodes") in self.args.fid_igm:
                znode = self.args.fid_igm[pname + "_znodes"][ii]
            elif (pname + "_znodes") in self.args.fid_cont:
                znode = self.args.fid_cont[pname + "_znodes"][ii]
            elif (pname + "_znodes") in self.args.fid_syst:
                znode = self.args.fid_syst[pname + "_znodes"][ii]
            else:
                raise ValueError("Could not find znode for " + p.name)

            iz = np.argmin(np.abs(dir_out["z"] - znode))
            # print(iz, znode, dir_out["z"][iz])
            # print(dir_out["pnames"][iz], pname + "_0")
            iname = np.argwhere(np.array(dir_out["pnames"][iz]) == (pname + "_0"))[0, 0]
            p.value = list(dir_out["mle"][iz].values())[iname]

            if verbose and (self.rank == 0):
                print(
                    p.name,
                    "\t",
                    np.round(p.value, 3),
                    "\t",
                    np.round(p.min_value, 3),
                    "\t",
                    np.round(p.max_value, 3),
                    "\t",
                    p.Gauss_priors_width,
                    p.fixed,
                )

        # reset the coefficients of the models
        self.theory.model_igm.models["F_model"].reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_igm.models["T_model"].reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_cont.hcd_model.reset_coeffs(free_params, rank=self.rank)
        self.theory.model_cont.metal_models["Si_mult"].reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_cont.metal_models["Si_add"].reset_coeffs(
            free_params, rank=self.rank
        )

    def set_ic_global(self, fname, verbose=True):
        """Set the initial conditions for the likelihood from a fit"""
        dir_out = np.load(fname, allow_pickle=True).item()

        # make a copy of free params, and set their values to the best-fit
        free_params = self.free_params.copy()
        for jj, p in enumerate(free_params):
            if p.name in ["As", "ns"]:
                continue
            pname, iistr = split_string(p.name)
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

            p.fixed = isfixed

            if (pname not in dir_out) and (pname == "HCD_const"):
                p.value = 0
            else:
                _z = dir_out[pname]["z"]
                _val = dir_out[pname]["val"]
                p.value = np.interp(znode, _z, _val)

            if verbose and (self.rank == 0):
                print(
                    p.name,
                    "\t",
                    np.round(p.value, 3),
                    "\t",
                    np.round(p.min_value, 3),
                    "\t",
                    np.round(p.max_value, 3),
                    "\t",
                    p.Gauss_priors_width,
                    p.fixed,
                )

        # reset the coefficients of the models
        # self.theory.model_igm.models["F_model"].reset_coeffs(
        #     free_params, rank=self.rank
        # )
        # self.theory.model_igm.models["T_model"].reset_coeffs(
        #     free_params, rank=self.rank
        # )
        for model in self.theory.model_igm.models:
            self.theory.model_igm.models[model].reset_coeffs(
                free_params, rank=self.rank
            )

        self.theory.model_cont.hcd_model.reset_coeffs(free_params, rank=self.rank)

        # self.theory.model_cont.metal_models["Si_mult"].reset_coeffs(
        #     free_params, rank=self.rank
        # )
        # self.theory.model_cont.metal_models["Si_add"].reset_coeffs(
        #     free_params, rank=self.rank
        # )

        for model in self.theory.model_cont.metal_models:
            self.theory.model_cont.metal_models[model].reset_coeffs(
                free_params, rank=self.rank
            )


def others_igm():
    # Galdwick 2021

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

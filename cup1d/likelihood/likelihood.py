import numpy as np
import os
import math
import copy
from mpi4py import MPI
from scipy.stats.distributions import chi2 as chi2_scipy
from scipy.optimize import minimize
from scipy.linalg import block_diag

from lace.cosmo import camb_cosmo
from cup1d.utils.utils import is_number_string
from cup1d.utils.compute_hessian import get_hessian

from cup1d.utils.utils import split_string
from cup1d.utils.utils import get_path_repo

from cup1d.utils.various_dicts import conv_strings


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


def get_bin_coverage(xmin_o, xmax_o, xmin_n, xmax_n):
    """Trick to accelerate rebinning"""
    # check out https://stcorp.github.io/harp/doc/html/algorithms/regridding.html
    cover = np.zeros((len(xmin_n), len(xmin_o)))
    for jj in range(len(xmin_n)):
        cover[jj] = np.fmax(
            (np.fmin(xmax_o, xmax_n[jj]) - np.fmax(xmin_o, xmin_n[jj]))
            / (xmax_o - xmin_o),
            0,
        )
    return cover


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
        extra_data=None,
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
        self.extra_data = extra_data
        # we only do this for latter save all relevant after fitting the model
        self.args = args

        if self.args.rebin_k != 1:
            self.rebin = {}
            self.rebin["k_kms"] = []  # new k_kms
            self.rebin["cover"] = []  # to accelerate rebinning
            self.rebin["sum_cover"] = []  # to accelerate rebinning
            for iz in range(len(self.data.z)):
                nelem = len(self.data.k_kms[iz]) * self.args.rebin_k
                _kms_reb = np.linspace(
                    self.data.k_kms_min[iz][0] * 0.95,
                    self.data.k_kms_max[iz][-1] * 1.05,
                    nelem,
                )
                self.rebin["k_kms"].append(_kms_reb)
                xmin_o = _kms_reb - 0.5 * (_kms_reb[1] - _kms_reb[0])
                xmax_o = _kms_reb + 0.5 * (_kms_reb[1] - _kms_reb[0])

                _cover = get_bin_coverage(
                    xmin_o,
                    xmax_o,
                    self.data.k_kms_min[iz],
                    self.data.k_kms_max[iz],
                )
                self.rebin["cover"].append(_cover)
                self.rebin["sum_cover"].append(np.sum(_cover, axis=1))

        self.theory = theory
        # Set inverse covariance. We do it here so we can account for emulator error
        self.set_icov()

        # setup parameters
        self.free_param_names = free_param_names
        self.set_free_parameters(free_param_names, free_param_limits)
        if verbose and (self.rank == 0):
            print(len(self.free_params), "free parameters")

        self.set_Gauss_priors()

        # sometimes we want to know the true theory (when working with mocks)
        self.set_truth()

        # store also fiducial model
        self.set_fid()

        # set blinding
        self.set_blinding()

        # set like to good starting point

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
                    print("No best fit found to set ICs:", args.file_ic)

    def rebinning(self, zs, Pk_kms_finek):
        """For rebinning Pk predictions"""
        Pk_kms_origk = []
        # _Pk_kms_finek = np.atleast_1d(Pk_kms_finek)
        for iz in range(len(zs)):
            indz = np.argmin(np.abs(self.data.z - zs[iz]))
            _Pk_kms = (
                np.sum(
                    self.rebin["cover"][indz] * Pk_kms_finek[iz][np.newaxis, :],
                    axis=1,
                )
                / self.rebin["sum_cover"][indz]
            )
            Pk_kms_origk.append(_Pk_kms)
        return Pk_kms_origk

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

    def set_blinding(self):
        """Set the blinding parameters"""
        blind_prior = {"Delta2_star": 0.05, "n_star": 0.01, "alpha_star": 0.005}
        if self.data.apply_blinding:
            seed = int.from_bytes(
                self.data.blinding.encode("utf-8"), byteorder="big"
            )
            rng = np.random.default_rng(seed)
        self.blind = {}
        for key in blind_prior:
            if self.data.apply_blinding:
                self.blind[key] = rng.normal(0, blind_prior[key])
            else:
                self.blind[key] = 0

    def apply_blinding(self, dict_cosmo, conv=False, sample=None):
        """Apply blinding to the dict_cosmo"""

        if self.data.apply_blinding:
            if sample is not None:
                if self.rank == 0:
                    print("Blinding " + sample)
            for key in self.blind:
                if conv:
                    key2 = conv_strings[key]
                else:
                    key2 = key

                try:
                    dict_cosmo[key2] += self.blind[key]
                except:
                    pass

        return dict_cosmo

    def apply_unblinding(self, dict_cosmo, conv=False):
        """Apply unblinding to the dict_cosmo"""
        out_dict = copy.deepcopy(dict_cosmo)
        for key in self.blind:
            if conv:
                key2 = conv_strings[key]
            else:
                key2 = key
            if key2 in dict_cosmo:
                out_dict[key2] = dict_cosmo[key2] - self.blind[key]
        return out_dict

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
        full_path = os.path.join(
            get_path_repo("lace"), "data", "covariance", filename
        )
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

        # Iterate over both datasets: main dataset (idata = 0) and additional dataset (idata = 1)
        for idata in range(2):
            if idata == 0:  # Main dataset
                data = self.data
                # Initialize list to store inverse covariance matrices for Pk_kms
                self.icov_Pk_kms = []
                self.cov_Pk_kms = []
                self.cov_emu_Pk_kms = []
                # Initialize the full inverse covariance matrix for Pk_kms
                self.full_icov_Pk_kms = None
                self.full_cov_Pk_kms = None
            else:  # Additional dataset
                data = self.extra_data
                # Initialize list for extra inverse covariance matrices
                self.extra_icov_Pk_kms = []
                self.extra_cov_Pk_kms = []
                self.extra_cov_emu_Pk_kms = []
                # Initialize the full inverse covariance matrix for extra data
                self.extra_full_icov_Pk_kms = None
                self.extra_full_cov_Pk_kms = None

            if data is None:  # Skip if no data is provided for this dataset
                continue

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
                dkms_dMpc = self.theory.fid_cosmo["cosmo"].dkms_dMpc(data.z[ii])
                k_Mpc = data.k_kms[ii] * dkms_dMpc

                # initialize emulator covariance
                add_emu_cov_kms = np.zeros((k_Mpc.shape[0], k_Mpc.shape[0]))

                # find closest z in cov
                ind0 = np.argmin(np.abs(emu_cov["zz_zk"] - data.z[ii]))
                # get cov from closest z
                ind = np.argwhere(emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0])[
                    :, 0
                ]
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
                if idata == 0:
                    self.icov_Pk_kms.append(np.linalg.inv(cov))
                    self.cov_Pk_kms.append(cov)
                    self.cov_emu_Pk_kms.append(add_emu_cov_kms)
                else:
                    self.extra_icov_Pk_kms.append(np.linalg.inv(cov))
                    self.extra_cov_Pk_kms.append(cov)
                    self.extra_cov_emu_Pk_kms.append(add_emu_cov_kms)

            # Process the full power spectrum data if available
            if data.full_Pk_kms is not None:
                # inflate errors
                cov_stat = data.full_cov_stat_Pk_kms.copy()
                cov_syst = data.full_cov_Pk_kms - cov_stat
                for i0 in range(cov_stat.shape[0]):
                    ind0 = np.argmin(
                        np.abs(self.cov_factor["z"] - data.full_zs[i0])
                    )
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
                        dkms_dMpc = self.theory.fid_cosmo["cosmo"].dkms_dMpc(
                            data.full_zs[i0]
                        )
                        full_k_kms0 = data.full_k_kms[i0] * dkms_dMpc

                        # find closest z in cov
                        ind0 = np.argmin(
                            np.abs(emu_cov["zz_zk"] - data.full_zs[i0])
                        )
                        ind = np.argwhere(
                            emu_cov["zz_zk"] == emu_cov["zz_zk"][ind0]
                        )[:, 0]
                        # find closest k for such z
                        ind1 = np.argmin(
                            np.abs(emu_cov["k_Mpc_zk"] - full_k_kms0)
                        )
                        # closest index in z and k
                        j0 = ind[ind1]

                        # index to inflate
                        ind0_infl = np.argmin(
                            np.abs(self.cov_factor["z"] - data.full_zs[i0])
                        )

                        for i1 in range(cov.shape[0]):
                            dkms_dMpc = self.theory.fid_cosmo[
                                "cosmo"
                            ].dkms_dMpc(data.full_zs[i1])
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
                                np.abs(emu_cov["k_Mpc_zk"] - full_k_kms1)
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
                    ind0 = np.argmin(
                        np.abs(self.cov_factor["z"] - data.full_zs[i0])
                    )
                    fact0 = self.cov_factor["val_full"][ind0]

                    for i1 in range(cov.shape[0]):
                        ind1 = np.argmin(
                            np.abs(self.cov_factor["z"] - data.full_zs[i1])
                        )
                        fact1 = self.cov_factor["val_full"][ind1]

                        cov[i0, i1] = cov[i0, i1] * fact0 * fact1

                # Compute and store the inverse covariance matrix
                if idata == 0:
                    self.full_icov_Pk_kms = np.linalg.inv(cov)
                    self.full_cov_Pk_kms = cov
                    self.emu_full_cov_Pk_kms = full_emu_cov
                else:
                    self.extra_full_icov_Pk_kms = np.linalg.inv(cov)
                    self.extra_full_cov_Pk_kms = cov
                    self.extra_emu_full_cov_Pk_kms = full_emu_cov

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

        assert (
            len(cosmo_dict) > 0
        ), "No cosmology parameters found in sampling space"

        return cosmo_dict

    def set_truth(self):
        """Store true cosmology from the simulation used to make mock data"""

        # access true cosmology used in mock data
        if hasattr(self.data, "truth") == False:
            if self.rank == 0:
                print("will not store truth, working with real data")
            self.truth = None
            return

        self.truth = {}
        for par in self.data.truth:
            self.truth[par] = self.data.truth[par]

        # make sure that we compare the correct zs
        ztruth = self.data.truth["igm"]["z"]
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
        for key in self.data.truth["igm"]:
            if key not in self.theory.model_igm.fid_igm:
                continue
            lenz = self.theory.model_igm.fid_igm[key].shape[0]
            if (
                np.allclose(
                    np.array(self.data.truth["igm"][key])[mask_z],
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
                        self.truth["like_params_cube"][
                            par.name
                        ] = par.get_value_in_cube(
                            self.truth["like_params"][par.name]
                        )
                    else:
                        self.truth["like_params"][par.name] = 0
                        self.truth["like_params_cube"][
                            par.name
                        ] = par.get_value_in_cube(
                            self.truth["like_params"][par.name]
                        )
                else:
                    self.truth["like_params"][par.name] = np.infty
                    self.truth["like_params_cube"][par.name] = np.infty
            elif (par.name == "As") | (par.name == "ns") | (par.name == "nrun"):
                self.truth["like_params"][par.name] = self.truth["cosmo"][
                    par.name
                ]
                self.truth["like_params_cube"][
                    par.name
                ] = par.get_value_in_cube(self.truth["like_params"][par.name])
                self.truth["like_params"][pname2[par.name]] = self.truth[
                    "linP"
                ][pname2[par.name]]
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

    def set_fid(self):
        """Store fiducial cosmology assumed for the fit"""

        self.fid = {}

        sim_cosmo = self.theory.fid_cosmo["cosmo"].cosmo

        self.fid["cosmo"] = {}
        self.fid["cosmo"]["ombh2"] = sim_cosmo.ombh2
        self.fid["cosmo"]["omch2"] = sim_cosmo.omch2
        self.fid["cosmo"]["As"] = sim_cosmo.InitPower.As
        self.fid["cosmo"]["ns"] = sim_cosmo.InitPower.ns
        self.fid["cosmo"]["nrun"] = sim_cosmo.InitPower.nrun
        self.fid["cosmo"]["H0"] = sim_cosmo.H0
        self.fid["cosmo"]["mnu"] = camb_cosmo.get_mnu(sim_cosmo)

        blob_params = ["Delta2_star", "n_star", "alpha_star"]
        blob = self.theory.fid_cosmo["cosmo"].get_linP_params()

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
        zs=None,
        _k_kms=None,
        values=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
        apply_hull=True,
        remove=None,
    ):
        """Compute theoretical prediction for 1D P(k)"""

        if _k_kms is None:
            k_kms = self.data.k_kms
        else:
            k_kms = _k_kms

        if zs is None:
            zs = self.data.z

        if self.args.rebin_k != 1:
            k_kms = []
            zs = np.atleast_1d(zs)
            for iz in range(len(zs)):
                ind = np.argmin(np.abs(zs[iz] - self.data.z))
                k_kms.append(self.rebin["k_kms"][ind])

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params = self.parameters_from_sampling_point(values)
        else:
            like_params = []

        results = self.theory.get_p1d_kms(
            zs,
            k_kms,
            like_params=like_params,
            return_covar=return_covar,
            return_blob=return_blob,
            return_emu_params=return_emu_params,
            apply_hull=apply_hull,
            remove=remove,
        )

        if results is None:
            return None

        out = []
        if return_blob | return_emu_params:
            p1ds = results[0]
        else:
            p1ds = results

        if self.args.rebin_k == 1:
            out.append(p1ds)
        else:
            out.append(self.rebinning(zs, p1ds))

        if return_blob | return_emu_params:
            for ii in range(1, len(results)):
                out.append(results[ii])

        return out

    def get_chi2(self, values=None, return_all=False, zmask=None):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""

        log_like, log_like_all = self.get_log_like(
            values, ignore_log_det_cov=True, zmask=zmask
        )

        if return_all:
            return -2.0 * log_like, -2.0 * log_like_all
        else:
            return -2.0 * log_like

    def get_error(self, p0):
        # get hessian to compute errors
        hess = get_hessian(self.minus_log_prob, p0)
        ihess = np.linalg.inv(hess)

        for par in self.free_params:
            if par.name == "As":
                scale_As = par.max_value - par.min_value
            elif par.name == "ns":
                scale_ns = par.max_value - par.min_value

        scaled_cov = np.zeros((2, 2))
        scaled_cov[0, 0] = ihess[0, 0] * scale_As**2
        scaled_cov[1, 1] = ihess[1, 1] * scale_ns**2
        scaled_cov[1, 0] = ihess[1, 0] * scale_As * scale_ns
        scaled_cov[0, 1] = ihess[0, 1] * scale_As * scale_ns

        like_params = self.parameters_from_sampling_point(p0)
        err = self.theory.err_star(scaled_cov, like_params)

        return {"err_Delta2star": err[0], "err_nstar": err[1]}

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

        # ask emulator prediction for P1D in each bin
        if zmask is not None:
            emu_p1d = []
            for iz in range(len(self.data.z)):
                ind = np.argwhere(np.abs(zmask - self.data.z[iz]) < 1e-3)
                if len(ind) == 0:
                    emu_p1d.append(0)
                else:
                    _res = self.get_p1d_kms(
                        np.atleast_1d(self.data.z[iz]),
                        np.atleast_2d(self.data.k_kms[iz]),
                        values,
                        return_blob=return_blob,
                    )
                    if _res is None:
                        return null_out

                    if return_blob:
                        blob = _res[1]

                    emu_p1d.append(_res[0])
        else:
            _res = self.get_p1d_kms(
                self.data.z, self.data.k_kms, values, return_blob=return_blob
            )
            if _res is None:
                return null_out

            if return_blob:
                emu_p1d, blob = _res
            else:
                emu_p1d = _res

        # out of priors
        if len(emu_p1d) == 1:
            if (len(emu_p1d[0]) == 1) | (len(emu_p1d[0]) == len(self.data.z)):
                emu_p1d = emu_p1d[0]

        # use high-res data
        if self.extra_data is not None:
            length = 2
            nz = np.max([len(self.data.z), len(self.extra_data.z)])

            if zmask is not None:
                _res = []
                for iz in range(len(self.extra_data.z)):
                    ind = np.argwhere(
                        np.abs(zmask - self.extra_data.z[iz]) < 1e-3
                    )
                    if len(ind) == 0:
                        _res.append(0)
                    else:
                        _res = self.get_p1d_kms(
                            np.atleast_1d(self.extra_data.z[iz]),
                            np.atleast_1d(self.extra_data.k_kms[iz]),
                            values,
                            return_blob=return_blob,
                        )
            else:
                _res = self.get_p1d_kms(
                    self.extra_data.z,
                    self.extra_data.k_kms,
                    values,
                    return_blob=return_blob,
                )

            emu_p1d_extra = _res
            # out of priors
            if emu_p1d_extra is None:
                return null_out

        else:
            length = 1
            nz = len(self.data.z)

        # compute log like contribution from each redshift bin
        log_like_all = np.zeros((length, nz))
        log_like = 0
        # loop over low and high res data
        for ii in range(length):
            if ii == 0:
                emu_p1d_use = emu_p1d
                data = self.data
                icov_Pk_kms = self.icov_Pk_kms
                full_icov_Pk_kms = self.full_icov_Pk_kms
            else:
                emu_p1d_use = emu_p1d_extra
                data = self.extra_data
                icov_Pk_kms = self.extra_icov_Pk_kms
                full_icov_Pk_kms = self.extra_full_icov_Pk_kms

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
                    log_like_all[ii, iz] = -0.5 * chi2_z
                else:
                    log_det_cov = np.log(
                        np.abs(1 / np.linalg.det(icov_Pk_kms[iz]))
                    )
                    log_like_all[ii, iz] = -0.5 * (chi2_z + log_det_cov)

            if (full_icov_Pk_kms is None) | (zmask is not None):
                log_like += np.sum(log_like_all[ii])
            else:
                # compute chi2 using full cov
                diff = data.full_Pk_kms - np.concatenate(emu_p1d_use)
                chi2_all = np.dot(np.dot(full_icov_Pk_kms, diff), diff)
                if ignore_log_det_cov:
                    log_like += -0.5 * chi2_all
                else:
                    log_det_cov = np.log(
                        np.abs(1 / np.linalg.det(full_icov_Pk_kms))
                    )
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

    def maximise_posterior(
        self, initial_values=None, method="nelder-mead", tol=1e-4
    ):
        """Run scipy minimizer to find maximum of posterior"""

        if not initial_values:
            initial_values = np.ones(len(self.free_params)) * 0.5

        return minimize(
            self.minus_log_prob, x0=initial_values, method=method, tol=tol
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
    ):
        """Plot P1D in theory vs data. If plot_every_iz >1,
        plot only few redshift bins"""

        if (zmask is not None) | (plot_realizations == False):
            n_perturb = 0

        if zmask is None:
            _data_z = self.data.z
            _data_k_kms = self.data.k_kms
        else:
            _data_z = []
            _data_k_kms = []
            for iz in range(len(self.data.z)):
                _ = np.argwhere(np.abs(zmask - self.data.z[iz]) < 1e-3)
                if len(_) != 0:
                    _data_z.append(self.data.z[iz])
                    _data_k_kms.append(self.data.k_kms[iz])
            _data_z = np.array(_data_z)

        # z at time fits or full fit
        if z_at_time is False:
            _res = self.get_p1d_kms(
                _data_z, _data_k_kms, values, return_covar=return_covar
            )
            if _res is None:
                return print("Prior out of range")
            if return_covar:
                emu_p1d, emu_cov = _res
            else:
                emu_p1d = _res

            # dict_save = {
            #     "z": _data_z,
            #     "k_kms": _data_k_kms,
            #     "p1d_model": emu_p1d,
            # }
            # np.save("test_model.npy", dict_save)

            if len(emu_p1d) == 1:
                emu_p1d = emu_p1d[0]

            # the sum of chi2_all may be different from chi2 due to covariance
            chi2, chi2_all = self.get_chi2(
                values=values, return_all=True, zmask=zmask
            )

            if chi2_nozcov:
                chi2 = np.sum(chi2_all)

        else:
            emu_p1d = []
            chi2_all = []
            ndeg_all = []
            for iz in range(len(_data_z)):
                _res = self.get_p1d_kms(
                    _data_z[iz],
                    _data_k_kms[iz],
                    values[iz],
                    return_covar=return_covar,
                )
                _ = np.argwhere(values[iz] != 0)[:, 0]
                # print(iz, len(_data_k_kms[iz]), len(_))
                ndeg_all.append(len(_data_k_kms[iz]) - len(_))
                if _res is None:
                    return print("Prior out of range for z = ", _data_z[iz])
                if return_covar:
                    emu_p1d.append(_res[0])
                else:
                    if len(_res) == 1:
                        emu_p1d.append(_res[0])
                    else:
                        emu_p1d.append(_res)

                _chi2, _ = self.get_chi2(
                    values=values[iz],
                    return_all=True,
                    zmask=np.array([_data_z[iz]]),
                )
                chi2_all.append(_chi2)
            chi2 = np.sum(chi2_all)
            # account for extra_data
            chi2_all = np.array([chi2_all])

        if self.extra_data is not None:
            _res = self.get_p1d_kms(
                self.extra_data.z,
                self.extra_data.k_kms,
                values,
                return_covar=return_covar,
            )
            if _res is None:
                return print("Prior out of range")
            if return_covar:
                emu_p1d_extra, emu_cov_extra = _res
            else:
                emu_p1d_extra = _res

        # if rand_posterior is not None:
        #     Nz = len(self.data.z)
        #     rand_emu = np.zeros((rand_posterior.shape[0], Nz, len(k_emu_kms)))
        #     for ii in range(rand_posterior.shape[0]):
        #         rand_emu[ii] = self.get_p1d_kms(
        #             self.data.z, k_emu_kms, rand_posterior[ii]
        #         )
        #     err_posterior = np.std(rand_emu, axis=0)

        #     if self.extra_data is not None:
        #         Nz = len(self.extra_data.z)
        #         rand_emu_extra = np.zeros(
        #             (rand_posterior.shape[0], Nz, len(k_emu_kms_extra))
        #         )
        #         for ii in range(rand_posterior.shape[0]):
        #             rand_emu_extra[ii] = self.get_p1d_kms(
        #                 self.extra_data.z, k_emu_kms_extra, rand_posterior[ii]
        #             )
        #         err_posterior_extra = np.std(rand_emu_extra, axis=0)

        if self.extra_data is None:
            if plot_panels:
                nrows = len(_data_z) // 3
                if len(_data_z) % 3 != 0:
                    nrows += 1
                if nrows == 0:
                    nrows = 1
                fig, ax = plt.subplots(
                    nrows, 3, figsize=(12, nrows * 2), sharex=True, sharey="row"
                )
                if len(_data_z) == 1:
                    ax = [ax]
                else:
                    ax = ax.reshape(-1)
                    if len(_data_z) % 2 != 0:
                        ax[-1].axis("off")

                length = 1
            else:
                fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                length = 1
                ax = [ax]
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            length = 2

        # figure out y range for plot
        ymin = 1e10
        ymax = -1e10

        # print chi2
        if z_at_time is False:
            n_free_p = len(self.free_params)
            ndeg = 0
            for iz in range(len(self.data.k_kms)):
                ndeg += np.sum(self.data.Pk_kms[iz] != 0)
            if self.extra_data is not None:
                for iz in range(len(self.extra_data.k_kms)):
                    ndeg += np.sum(self.extra_data.Pk_kms[iz] != 0)
            _ndeg = ndeg - n_free_p
            if fix_cosmo:
                _ndeg -= 2
        else:
            _ndeg = np.sum(ndeg_all)
        prob = chi2_scipy.sf(chi2, _ndeg)
        if self.rank == 0:
            print(prob * 100)

        if prob > 0.0001:
            str_chi2 = str(np.round(prob * 100, 2))
        else:
            str_chi2 = str(np.round(prob * 100, 4))
        label = (
            r"$\chi^2=$"
            + str(np.round(chi2, 2))
            + r", $n_\mathrm{deg}$="
            + str(_ndeg)
            + ", prob="
            + str_chi2
            + "%"
        )

        fig.suptitle(label, fontsize=fontsize)

        out = {}

        for ii in range(length):
            if ii == 0:
                data = self.data
                emu_p1d_use = emu_p1d
                if return_covar:
                    emu_cov_use = emu_cov
                if rand_posterior is not None:
                    err_posterior_use = err_posterior
                out["zs"] = []
                out["k_kms"] = []
                out["p1d_data"] = []
                out["p1d_model"] = []
                out["p1d_err"] = []
                out["chi2"] = []
                out["prob"] = []
            else:
                data = self.extra_data
                emu_p1d_use = emu_p1d_extra
                if return_covar:
                    emu_cov_use = emu_cov_extra
                if rand_posterior is not None:
                    err_posterior_use = err_posterior_extra
                out["extra_zs"] = []
                out["extra_k_kms"] = []
                out["extra_p1d_data"] = []
                out["extra_p1d_model"] = []
                out["extra_p1d_err"] = []
                out["extra_chi2"] = []
                out["extra_prob"] = []

            if n_perturb > 0:
                full_emu_p1d = np.concatenate(emu_p1d_use)
                perturb = np.random.multivariate_normal(
                    full_emu_p1d, self.full_cov_Pk_kms, n_perturb
                )

            zs = data.z
            Nz = len(zs)

            # plot only few redshifts for clarity
            for iz in range(0, Nz, plot_every_iz):
                if zmask is not None:
                    indemu = np.argwhere(np.abs(zmask - zs[iz]) < 1e-3)[:, 0]
                    if len(indemu) == 0:
                        continue
                    else:
                        indemu = indemu[0]
                else:
                    indemu = iz
                # access data for this redshift
                z = zs[iz]
                k_kms = data.k_kms[iz]
                p1d_data = data.Pk_kms[iz]
                p1d_cov = self.cov_Pk_kms[iz]
                p1d_err = np.sqrt(np.diag(p1d_cov))
                p1d_theory = emu_p1d_use[indemu]
                if len(p1d_theory) == 1:
                    p1d_theory = p1d_theory[0]

                if rand_posterior is None:
                    if return_covar:
                        cov_theory = emu_cov_use[iz]
                        err_theory = np.sqrt(np.diag(cov_theory))
                else:
                    err_theory = err_posterior_use[iz]

                # plot everything
                if Nz > 1:
                    col = plt.cm.jet(iz / (Nz - 1))
                    if collapse:
                        yshift = 0
                    else:
                        yshift = 4 * iz / (Nz - 1)
                else:
                    col = "C0"
                    yshift = 0

                if plot_panels:
                    col = "C0"
                    yshift = 0

                if residuals:
                    if plot_panels:
                        axs = ax[indemu]
                        yshift = 0
                    else:
                        axs = ax[ii]

                    try:
                        axs = axs[0]
                    except:
                        pass

                    axs.tick_params(
                        axis="both", which="major", labelsize=fontsize
                    )
                    # shift data in y axis for clarity
                    axs.errorbar(
                        k_kms,
                        p1d_data / p1d_theory + yshift,
                        color=col,
                        yerr=p1d_err / p1d_theory,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )

                    ind = self.data.full_zs == z
                    for kk in range(n_perturb):
                        axs.plot(
                            k_kms,
                            perturb[kk, ind] / p1d_theory + yshift,
                            color=col,
                            alpha=0.025,
                        )

                    # print chi2
                    xpos = k_kms[0]
                    ndeg = np.sum(p1d_data != 0)
                    # get degrees of freedom
                    if z_at_time:
                        _ndeg = ndeg_all[iz]
                    else:
                        _ndeg = ndeg - n_free_p
                    if glob_full:
                        _ndeg = ndeg - n_param_glob_full

                    prob = chi2_scipy.sf(chi2_all[ii, iz], _ndeg)

                    if print_chi2:
                        label = (
                            r"$\chi^2=$"
                            + str(np.round(chi2_all[ii, iz], 2))
                            + r", $n_\mathrm{deg}$="
                            + str(_ndeg)
                            + ", prob="
                            + str(np.round(prob * 100, 2))
                            + "%"
                        )
                    else:
                        label = (
                            r"$\chi^2=$"
                            + str(np.round(chi2_all[ii, iz], 2))
                            + r", $n_\mathrm{data}$="
                            + str(ndeg)
                        )

                    if print_chi2:
                        if plot_panels == False:
                            ypos = 0.75 + yshift
                            axs.text(xpos, ypos, label, fontsize=fontsize - 4)

                    if print_ratio:
                        if self.rank == 0:
                            print(p1d_data / p1d_theory)
                    ymin = min(ymin, min(p1d_data / p1d_theory + yshift))
                    ymax = max(ymax, max(p1d_data / p1d_theory + yshift))

                    axs.axhline(1, color="k", linestyle=":", alpha=0.5)

                    if return_covar | (rand_posterior is not None):
                        axs.fill_between(
                            k_kms,
                            (p1d_theory + err_theory) / p1d_theory + yshift,
                            (p1d_theory - err_theory) / p1d_theory + yshift,
                            alpha=0.35,
                            color=col,
                        )
                else:
                    ax[ii].errorbar(
                        k_kms,
                        p1d_data * k_kms / np.pi,
                        color=col,
                        yerr=p1d_err * k_kms / np.pi,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )

                    ind = self.data.full_zs == z
                    for kk in range(n_perturb):
                        ax[ii].plot(
                            k_kms,
                            perturb[kk, ind] * k_kms / np.pi,
                            color=col,
                            alpha=0.05,
                        )

                    # print chi2
                    xpos = k_kms[-1] + 0.001
                    ypos = (p1d_theory * k_kms / np.pi)[-1]
                    ndeg = np.sum(p1d_data != 0)
                    prob = chi2_scipy.sf(chi2_all[ii, iz], ndeg - n_free_p)

                    if print_chi2:
                        label = (
                            r"$\chi^2=$"
                            + str(np.round(chi2_all[ii, iz], 2))
                            + r", $n_\mathrm{deg}$="
                            + str(ndeg - n_free_p)
                            + ", prob="
                            + str(np.round(prob * 100, 2))
                            + "%"
                        )
                    else:
                        label = (
                            r"$\chi^2=$"
                            + str(np.round(chi2_all[ii, iz], 2))
                            + r", $n_\mathrm{data}$="
                            + str(ndeg)
                        )

                    ax[ii].text(xpos, ypos, label, fontsize=fontsize - 4)

                    ax[ii].plot(
                        k_kms,
                        (p1d_theory * k_kms) / np.pi,
                        color=col,
                        linestyle="dashed",
                    )
                    if return_covar | (rand_posterior is not None):
                        ax[ii].fill_between(
                            k_kms,
                            (p1d_theory + err_theory) * k_kms / np.pi,
                            (p1d_theory - err_theory) * k_kms / np.pi,
                            alpha=0.35,
                            color=col,
                        )
                    ymin = min(ymin, min(p1d_data * k_kms / np.pi))
                    ymax = max(ymax, max(p1d_data * k_kms / np.pi))

                if residuals & plot_panels:
                    axs.legend(loc="upper right", fontsize=fontsize - 4)
                    ymin = 1 - min((p1d_data - p1d_err) / p1d_theory + yshift)
                    ymax = 1 - max((p1d_data + p1d_err) / p1d_theory + yshift)
                    y2plot = 1.05 * np.max([np.abs(ymin), np.abs(ymax)])
                    if iz % 2 == 1:
                        axs.set_ylim(1 - y2plot, 1 + y2plot)
                    elif iz == len(zs) - 1:
                        axs.set_ylim(1 - y2plot, 1 + y2plot)

                    # if print_chi2:
                    axs.text(
                        0.05,
                        0.05,
                        label,
                        fontsize=fontsize - 4,
                        transform=axs.transAxes,
                    )

                if ii == 0:
                    out["zs"].append(z)
                    out["k_kms"].append(k_kms)
                    out["p1d_data"].append(p1d_data)
                    out["p1d_model"].append(p1d_theory)
                    out["p1d_err"].append(p1d_err)
                    out["chi2"].append(chi2_all[ii, iz])
                    out["prob"].append(prob)
                else:
                    out["extra_zs"].append(z)
                    out["extra_k_kms"].append(k_kms)
                    out["extra_p1d_data"].append(p1d_data)
                    out["extra_p1d_model"].append(p1d_theory)
                    out["extra_p1d_err"].append(p1d_err)
                    out["extra_chi2"].append(chi2_all[ii, iz])
                    out["extra_prob"].append(prob)

            # ax[ii].plot(k_kms[0], 1, linestyle="-", label="Data", color="k")
            # ax[ii].plot(k_kms[0], 1, linestyle="--", label="Fit", color="k")
            if residuals:
                if plot_panels == False:
                    axs.legend(fontsize=fontsize)
            else:
                ax[ii].legend(loc="lower right", ncol=4, fontsize=fontsize - 4)

            # ax[ii].set_xlim(min(k_kms[0]) - 0.001, max(k_kms[-1]) + 0.001)
            # if plot_panels == False:
            # ax[ii].set_xlabel(r"$k_\parallel$ [s/km]")
            # else:
            # ax[-1].set_xlabel(r"$k_\parallel$ [s/km]")

            if residuals:
                if plot_panels == False:
                    ax[ii].set_ylabel(
                        r"$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$",
                        fontsize=fontsize,
                    )
                    ax[ii].set_ylim(ymin - 0.3, ymax + 0.3)
            else:
                ax[ii].set_ylim(0.8 * ymin, 1.3 * ymax)
                ax[ii].set_yscale("log")
                ax[ii].set_ylabel(
                    r"$k_\parallel \, P_{\rm 1D}(z, k_\parallel) / \pi$",
                    fontsize=fontsize,
                )

        fig.supxlabel(
            r"$k_\parallel\,[\mathrm{km}^{-1}\mathrm{s}]$", fontsize=fontsize
        )
        fig.supylabel(
            r"$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$",
            fontsize=fontsize,
        )

        plt.tight_layout()

        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        if plot_fname is not None:
            plt.savefig(plot_fname + ".pdf")
            plt.savefig(plot_fname + ".png")
        else:
            if show:
                plt.show()

        if return_all:
            return out
        else:
            return

    def plot_p1d_errors(
        self,
        values=None,
        plot_fname=None,
        show=True,
        zmask=None,
        z_at_time=False,
        fontsize=16,
    ):
        """Plot P1D in theory vs data. If plot_every_iz >1,
        plot only few redshift bins"""

        import scipy.stats as stats

        if zmask is None:
            _data_z = self.data.z
            _data_k_kms = self.data.k_kms
        else:
            _data_z = []
            _data_k_kms = []
            for iz in range(len(self.data.z)):
                _ = np.argwhere(np.abs(zmask - self.data.z[iz]) < 1e-3)
                if len(_) != 0:
                    _data_z.append(self.data.z[iz])
                    _data_k_kms.append(self.data.k_kms[iz])
            _data_z = np.array(_data_z)

        # z at time fits or full fit
        if z_at_time is False:
            _res = self.get_p1d_kms(
                _data_z, _data_k_kms, values, return_covar=False
            )
            if _res is None:
                if self.rank == 0:
                    return print("Prior out of range")
            emu_p1d = _res

            # the sum of chi2_all may be different from chi2 due to covariance
            chi2, chi2_all = self.get_chi2(
                values=values, return_all=True, zmask=zmask
            )
        else:
            emu_p1d = []
            for iz in range(len(_data_z)):
                _res = self.get_p1d_kms(
                    _data_z[iz],
                    _data_k_kms[iz],
                    values[iz],
                    return_covar=False,
                )
                # print(iz, _data_z[iz], _res)
                if _res is None:
                    if self.rank == 0:
                        return print("Prior out of range for z = ", _data_z[iz])
                if len(_res) == 1:
                    emu_p1d.append(_res[0])
                else:
                    emu_p1d.append(_res)

        if self.extra_data is not None:
            _res = self.get_p1d_kms(
                self.extra_data.z,
                self.extra_data.k_kms,
                values,
                return_covar=return_covar,
            )
            if _res is None:
                if self.rank == 0:
                    return print("Prior out of range")
            if return_covar:
                emu_p1d_extra, emu_cov_extra = _res
            else:
                emu_p1d_extra = _res

        fig, ax = plt.subplots(
            len(_data_z) // 2 + len(_data_z) % 2,
            2,
            figsize=(12, len(_data_z)),
            sharex=True,
            sharey=True,
        )
        if len(_data_z) == 1:
            ax = [ax]
        else:
            ax = ax.reshape(-1)
        length = 1
        # if (len(_data_z) % 2 + 1) != 0:
        #     ax[-1].axis("off")

        out = {}
        bins = np.linspace(-5, 5, 50)
        out["bins"] = bins

        data = self.data
        emu_p1d_use = emu_p1d
        out["zs"] = []
        out["(d-m)/err"] = []

        mu = 0
        sigma = 1
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)

        zs = data.z
        Nz = len(zs)

        for iz in range(0, Nz):
            if zmask is not None:
                indemu = np.argwhere(np.abs(zmask - zs[iz]) < 1e-3)[:, 0]
                if len(indemu) == 0:
                    continue
                else:
                    indemu = indemu[0]
            else:
                indemu = iz

            # access data for this redshift
            z = zs[iz]
            k_kms = data.k_kms[iz]
            p1d_data = data.Pk_kms[iz]
            p1d_cov = self.cov_Pk_kms[iz]
            p1d_err = np.sqrt(np.diag(p1d_cov))

            if len(emu_p1d_use) == 1:
                if len(emu_p1d_use[0]) == 1:
                    p1d_theory = emu_p1d_use[0][0]
                else:
                    p1d_theory = emu_p1d_use[0][indemu]
            else:
                p1d_theory = emu_p1d_use[indemu]

            dme = (p1d_data - p1d_theory) / p1d_err

            ax[iz].tick_params(
                axis="both", which="major", labelsize=fontsize - 4
            )
            ax[iz].hist(
                dme,
                label="z=" + str(np.round(z, 2)),
                bins=bins,
                color="C0",
                density=True,
            )

            ax[iz].plot(x, stats.norm.pdf(x, mu, sigma), color="C1")

            out["zs"].append(z)
            out["(d-m)/err"].append(dme)

            ax[iz].legend()

        dme = np.concatenate(out["(d-m)/err"])
        if self.rank == 0:
            print("dme, mean", np.mean(dme))
            print("dme, std", np.std(dme))
        ax[iz + 1].hist(dme, label="All", bins=bins, density=True)
        ax[iz + 1].plot(x, stats.norm.pdf(x, mu, sigma), color="C1")
        ax[iz + 1].legend()

        fig.supxlabel(r"(d-m)/err", fontsize=fontsize)
        fig.supylabel(r"$PDF$", fontsize=fontsize)

        plt.tight_layout()
        # plt.savefig("test.pdf")
        if plot_fname is not None:
            plt.savefig(plot_fname + ".pdf")
            plt.savefig(plot_fname + ".png")
        else:
            if show:
                plt.show()

        return out

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
    ):
        if chain is not None:
            if len(chain.shape) == 3:
                chain_use = chain.reshape(-1, chain.shape[-1])
            else:
                chain_use = chain.copy()
            ind = np.random.permutation(np.arange(0, chain_use.shape[0]))[
                :nelem
            ]
            chain_use = chain_use[ind]

        ind = np.argwhere(self.data.z == zstar)[0][0]
        k_kms_inter = np.linspace(
            self.data.k_kms[ind].min(), self.data.k_kms[ind].max(), 500
        )

        labels = ["LLS", "sub-DLA", "small DLA", "large DLA", "All"]
        fig, ax = plt.subplots(figsize=(8, 6))
        ls = ["--", "-.", (0, (2, 2, 2, 2)), ":", "-"]

        for ii in range(5):
            par_plot = "HCD_damp" + str(ii + 1)
            # print(par_plot)

            if chain is None:
                free_params = self.parameters_from_sampling_point(p0)
                for par in free_params:
                    if ii + 1 <= 4:
                        if "HCD_damp" in par.name:
                            # print(par.name, par.value)
                            if par.name.startswith(par_plot):
                                pass
                            else:
                                par.value = -20

                hcd_cont = self.theory.model_cont.hcd_model.get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    like_params=free_params,
                )
                ax.plot(
                    k_kms_inter,
                    hcd_cont,
                    label=labels[ii],
                    alpha=0.75,
                    ls=ls[ii],
                    lw=3,
                    color="C" + str(ii),
                )
            else:
                all_hcd_cont = np.zeros((nelem, len(k_kms_inter)))

                for jj in range(nelem):
                    free_params = self.parameters_from_sampling_point(
                        chain_use[jj]
                    )
                    for par in free_params:
                        if ii + 1 <= 4:
                            if "HCD_damp" in par.name:
                                if par.name.startswith(par_plot):
                                    pass
                                else:
                                    par.value = -20
                    all_hcd_cont[
                        jj, :
                    ] = self.theory.model_cont.hcd_model.get_contamination(
                        z=np.array([zstar]),
                        k_kms=[k_kms_inter],
                        like_params=free_params,
                    )
                hcd_cont = np.percentile(all_hcd_cont, [16, 50, 84], axis=0)

                ax.plot(
                    k_kms_inter,
                    hcd_cont[1],
                    label=labels[ii],
                    alpha=0.75,
                    ls=ls[ii],
                    lw=3,
                    color="C" + str(ii),
                )
                ax.fill_between(
                    k_kms_inter,
                    hcd_cont[0],
                    hcd_cont[2],
                    alpha=0.3,
                    color="C" + str(ii),
                )
        ax.axhline(1, color="k", ls=":", lw=2)
        ax.set_ylabel(r"$C_\mathrm{HCD}$", fontsize=ftsize)
        ax.set_xlabel(
            r"$k_\parallel\, [\mathrm{km}^{-1}\mathrm{s}]$", fontsize=ftsize
        )
        ax.tick_params(axis="both", which="major", labelsize=ftsize)
        ax.legend(fontsize=ftsize - 2)

        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cont_hcd")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_metal_cont_add(
        self,
        free_params=None,
        chain=None,
        save_directory=None,
        ftsize=24,
        nelem=5000,
    ):
        if chain is not None:
            if len(chain.shape) == 3:
                chain_use = chain.reshape(-1, chain.shape[-1])
            else:
                chain_use = chain.copy()
            ind = np.random.permutation(np.arange(0, chain_use.shape[0]))[
                :nelem
            ]
            chain_use = chain_use[ind]

        fig, ax = plt.subplots(figsize=(8, 6))
        ls = ["-", "--", "-.", ":"]
        for ii, zstar in enumerate([2.2, 2.8, 3.4, 4.0]):
            ind = np.argwhere(self.data.z == zstar)[0][0]
            k_kms_inter = np.linspace(
                self.data.k_kms[ind].min(), self.data.k_kms[ind].max(), 500
            )
            k_kms = self.data.k_kms[ind].copy()
            mF = self.theory.model_igm.models["F_model"].get_mean_flux(
                zstar, like_params=free_params
            )

            if chain is None:
                si_add_cont_all = self.theory.model_cont.metal_models[
                    "Si_add"
                ].get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    mF=np.array([mF]),
                    like_params=free_params,
                )
                ax.plot(
                    k_kms_inter,
                    si_add_cont_all[0],
                    label=r"$z=$" + str(zstar),
                    alpha=0.75,
                    ls=ls[ii],
                    lw=4,
                    color="C" + str(ii),
                )
            else:
                all_si_add_cont = np.zeros((nelem, len(k_kms_inter)))
                for jj in range(nelem):
                    free_params = self.parameters_from_sampling_point(
                        chain_use[jj]
                    )
                    all_si_add_cont[
                        jj, :
                    ] = self.theory.model_cont.metal_models[
                        "Si_add"
                    ].get_contamination(
                        z=np.array([zstar]),
                        k_kms=[k_kms_inter],
                        mF=np.array([mF]),
                        like_params=free_params,
                    )[
                        0
                    ]
                si_add_cont = np.percentile(
                    all_si_add_cont, [16, 50, 84], axis=0
                )

                ax.plot(
                    k_kms_inter,
                    si_add_cont[1],
                    label=r"$z=$" + str(zstar),
                    alpha=0.75,
                    ls=ls[ii],
                    lw=2,
                    color="C" + str(ii),
                )
                ax.fill_between(
                    k_kms_inter,
                    si_add_cont[0],
                    si_add_cont[2],
                    alpha=0.3,
                    color="C" + str(ii),
                )

        ax.axhline(0, color="k", ls=":", lw=2)
        ax.legend(fontsize=ftsize - 4, loc="upper right")
        ax.tick_params(axis="both", which="major", labelsize=ftsize)
        ax.set_ylabel(
            r"$C_\mathrm{SiII-SiII}\,[\mathrm{km}\,\mathrm{s}^{-1}]$",
            fontsize=ftsize,
        )
        ax.set_xlabel(
            r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize
        )
        ax.set_ylim(-0.2, 3)

        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cont_metal_add")
            plt.savefig(name + ".pdf", bbox_inches="tight")
            plt.savefig(name + ".png", bbox_inches="tight")
        else:
            plt.show()

    def plot_metal_cont_mult(
        self,
        free_params=None,
        chain=None,
        zstar=3,
        save_directory=None,
        ftsize=24,
        nelem=5000,
    ):
        """Plot metallicity contours"""

        if chain is not None:
            if len(chain.shape) == 3:
                chain_use = chain.reshape(-1, chain.shape[-1])
            else:
                chain_use = chain.copy()
            ind = np.random.permutation(np.arange(0, chain_use.shape[0]))[
                :nelem
            ]
            chain_use = chain_use[ind]

        ind = np.argwhere(self.data.z == zstar)[0][0]
        k_kms_inter = np.linspace(
            self.data.k_kms[ind].min(), self.data.k_kms[ind].max(), 500
        )
        # k_kms = self.data.k_kms[ind].copy()

        # dat_si_mult_cont_all = self.theory.model_cont.metal_models[
        #     "Si_mult"
        # ].get_contamination(
        #     z=np.array([zstar]),
        #     k_kms=[k_kms],
        #     mF=np.array([mF]),
        #     like_params=free_params,
        # )

        if chain is None:
            mF = self.theory.model_igm.models["F_model"].get_mean_flux(
                zstar, like_params=free_params
            )

            si_mult_cont_all = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
            )

            remove = {
                "SiIII_Lya": 1,
                "SiIIa_Lya": 0,
                "SiIIb_Lya": 0,
                "SiIIc_Lya": 0,
                "SiIII_SiIIa": 0,
                "SiIII_SiIIb": 0,
                "SiIII_SiIIc": 0,
                "SiIIc_SiIIb": 0,
                "SiIIc_SiIIa": 0,
                "SiIIb_SiIIa": 0,
            }

            si_mult_cont_SiIII = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
                remove=remove,
            )

            remove = {
                "SiIII_Lya": 0,
                "SiIIa_Lya": 1,
                "SiIIb_Lya": 1,
                "SiIIc_Lya": 0,
                "SiIII_SiIIa": 0,
                "SiIII_SiIIb": 0,
                "SiIII_SiIIc": 0,
                "SiIIc_SiIIb": 0,
                "SiIIc_SiIIa": 0,
                "SiIIb_SiIIa": 0,
            }

            si_mult_cont_SiII = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
                remove=remove,
            )

            remove = {
                "SiIII_Lya": 0,
                "SiIIa_Lya": 0,
                "SiIIb_Lya": 0,
                "SiIIc_Lya": 0,
                "SiIII_SiIIa": 1,
                "SiIII_SiIIb": 1,
                "SiIII_SiIIc": 0,
                "SiIIc_SiIIb": 0,
                "SiIIc_SiIIa": 0,
                "SiIIb_SiIIa": 0,
            }

            si_mult_cont_Si23 = self.theory.model_cont.metal_models[
                "Si_mult"
            ].get_contamination(
                z=np.array([zstar]),
                k_kms=[k_kms_inter],
                mF=np.array([mF]),
                like_params=free_params,
                remove=remove,
            )

            fig, ax = plt.subplots(4, figsize=(8, 6), sharey=True, sharex=True)
            ax[0].plot(
                k_kms_inter,
                si_mult_cont_SiIII[0],
                label=r"Ly$\alpha$-SiIII",
                alpha=0.75,
                ls="-",
                lw=3,
                color="C0",
            )
            ax[1].plot(
                k_kms_inter,
                si_mult_cont_SiII[0],
                label=r"Ly$\alpha$-SiII",
                alpha=0.75,
                ls="-",
                lw=3,
                color="C1",
            )
            ax[2].plot(
                k_kms_inter,
                si_mult_cont_Si23[0],
                label=r"SiII-SiIII",
                alpha=0.75,
                ls="-",
                lw=3,
                color="C2",
            )
            ax[3].plot(
                k_kms_inter,
                si_mult_cont_all[0],
                label=r"All",
                alpha=0.75,
                ls="-",
                lw=3,
                color="C3",
            )
        else:
            si_mult_cont_all = np.zeros((nelem, len(k_kms_inter)))
            si_mult_cont_SiIII = np.zeros((nelem, len(k_kms_inter)))
            si_mult_cont_SiII = np.zeros((nelem, len(k_kms_inter)))
            si_mult_cont_Si23 = np.zeros((nelem, len(k_kms_inter)))

            for jj in range(nelem):
                free_params = self.parameters_from_sampling_point(chain_use[jj])

                mF = self.theory.model_igm.models["F_model"].get_mean_flux(
                    zstar, like_params=free_params
                )

                si_mult_cont_all[jj] = self.theory.model_cont.metal_models[
                    "Si_mult"
                ].get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    mF=np.array([mF]),
                    like_params=free_params,
                )[
                    0
                ]

                remove = {
                    "SiIII_Lya": 1,
                    "SiIIa_Lya": 0,
                    "SiIIb_Lya": 0,
                    "SiIIc_Lya": 0,
                    "SiIII_SiIIa": 0,
                    "SiIII_SiIIb": 0,
                    "SiIII_SiIIc": 0,
                    "SiIIc_SiIIb": 0,
                    "SiIIc_SiIIa": 0,
                    "SiIIb_SiIIa": 0,
                }

                si_mult_cont_SiIII[jj] = self.theory.model_cont.metal_models[
                    "Si_mult"
                ].get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    mF=np.array([mF]),
                    like_params=free_params,
                    remove=remove,
                )[
                    0
                ]

                remove = {
                    "SiIII_Lya": 0,
                    "SiIIa_Lya": 1,
                    "SiIIb_Lya": 1,
                    "SiIIc_Lya": 0,
                    "SiIII_SiIIa": 0,
                    "SiIII_SiIIb": 0,
                    "SiIII_SiIIc": 0,
                    "SiIIc_SiIIb": 0,
                    "SiIIc_SiIIa": 0,
                    "SiIIb_SiIIa": 0,
                }

                si_mult_cont_SiII[jj] = self.theory.model_cont.metal_models[
                    "Si_mult"
                ].get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    mF=np.array([mF]),
                    like_params=free_params,
                    remove=remove,
                )[
                    0
                ]

                remove = {
                    "SiIII_Lya": 0,
                    "SiIIa_Lya": 0,
                    "SiIIb_Lya": 0,
                    "SiIIc_Lya": 0,
                    "SiIII_SiIIa": 1,
                    "SiIII_SiIIb": 1,
                    "SiIII_SiIIc": 0,
                    "SiIIc_SiIIb": 0,
                    "SiIIc_SiIIa": 0,
                    "SiIIb_SiIIa": 0,
                }

                si_mult_cont_Si23[jj] = self.theory.model_cont.metal_models[
                    "Si_mult"
                ].get_contamination(
                    z=np.array([zstar]),
                    k_kms=[k_kms_inter],
                    mF=np.array([mF]),
                    like_params=free_params,
                    remove=remove,
                )[
                    0
                ]

            per_siIII = np.percentile(si_mult_cont_SiIII, [16, 50, 84], axis=0)
            per_siII = np.percentile(si_mult_cont_SiII, [16, 50, 84], axis=0)
            per_si23 = np.percentile(si_mult_cont_Si23, [16, 50, 84], axis=0)
            per_siall = np.percentile(si_mult_cont_all, [16, 50, 84], axis=0)

            fig, ax = plt.subplots(4, figsize=(8, 6), sharey=True, sharex=True)
            ax[0].plot(
                k_kms_inter,
                per_siIII[1],
                label=r"Ly$\alpha$-SiIII",
                alpha=0.75,
                ls="-",
                lw=2,
                color="C0",
            )
            ax[0].fill_between(
                k_kms_inter,
                per_siIII[0],
                per_siIII[2],
                alpha=0.3,
                color="C0",
            )

            ax[1].plot(
                k_kms_inter,
                per_siII[1],
                label=r"Ly$\alpha$-SiII",
                alpha=0.75,
                ls="-",
                lw=2,
                color="C1",
            )
            ax[1].fill_between(
                k_kms_inter,
                per_siII[0],
                per_siII[2],
                alpha=0.3,
                color="C1",
            )

            ax[2].plot(
                k_kms_inter,
                per_si23[1],
                label=r"SiII-SiIII",
                alpha=0.75,
                ls="-",
                lw=2,
                color="C2",
            )
            ax[2].fill_between(
                k_kms_inter,
                per_si23[0],
                per_si23[2],
                alpha=0.3,
                color="C2",
            )

            ax[3].plot(
                k_kms_inter,
                per_siall[1],
                label=r"All",
                alpha=0.75,
                ls="-",
                lw=2,
                color="C3",
            )
            ax[3].fill_between(
                k_kms_inter,
                per_siall[0],
                per_siall[2],
                alpha=0.3,
                color="C3",
            )

        # ax[3].scatter(k_kms, dat_si_mult_cont_all[0], s=30, color="C3")
        for ii in range(4):
            ax[ii].axhline(1, color="k", ls=":", lw=2)
            ax[ii].legend(fontsize=ftsize - 4, loc="lower right")
            ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        fig.supylabel(r"$C_\mathrm{metal}$", fontsize=ftsize)
        ax[-1].set_xlabel(
            r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize
        )

        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cont_metal_mult")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

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
    ):
        """Plot IGM histories"""

        # true IGM parameters
        # if self.truth is not None:
        #     pars_true = {}
        #     pars_true["z"] = self.truth["igm"]["z"]
        #     pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
        #     pars_true["gamma"] = self.truth["igm"]["gamma"]
        #     pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
        #     pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

        zs = np.linspace(self.data.z.min(), self.data.z.max(), 100)
        p0 = self.sampling_point_from_parameters()

        for ii in range(3):
            if ii == 0:
                p0[:] = 0.5
            elif ii == 1:
                p0[:] = 0
            elif ii == 2:
                p0[:] = 1
            fid_params = self.parameters_from_sampling_point(p0)
            pars = {}
            pars["z"] = zs
            pars["tau_eff"] = self.theory.model_igm.models[
                "F_model"
            ].get_tau_eff(zs, like_params=fid_params)
            pars["mF"] = self.theory.model_igm.models["F_model"].get_mean_flux(
                zs, like_params=fid_params
            )
            pars["gamma"] = self.theory.model_igm.models["T_model"].get_gamma(
                zs, like_params=fid_params
            )
            pars["sigT_kms"] = self.theory.model_igm.models[
                "T_model"
            ].get_sigT_kms(zs, like_params=fid_params)
            pars["T0"] = (
                self.theory.model_igm.models["T_model"].get_T0(
                    zs, like_params=fid_params
                )
                / 1e4
            )
            pars["kF_kms"] = self.theory.model_igm.models["P_model"].get_kF_kms(
                zs, like_params=fid_params
            )
            if ii == 0:
                pars_fid = pars.copy()
            elif ii == 1:
                pars_min = pars.copy()
            elif ii == 2:
                pars_max = pars.copy()

        chain = None
        if chain_uformat is not None:
            if len(chain_uformat.shape) == 3:
                chain = chain_uformat.reshape(-1, chain_uformat.shape[-1])
            else:
                chain = chain_uformat.copy()
            ind = np.random.permutation(np.arange(0, chain.shape[0]))[:nelem]
            chain = chain[ind]

            if zmask is not None:
                zs2 = zmask
            else:
                zs2 = self.data.z

            pars_chain = {}
            pars_chain["z"] = zs
            pars_chain["tau_eff"] = np.zeros((chain.shape[0], zs.shape[0]))
            pars_chain["mF"] = np.zeros((chain.shape[0], zs.shape[0]))
            pars_chain["gamma"] = np.zeros((chain.shape[0], zs.shape[0]))
            pars_chain["sigT_kms"] = np.zeros((chain.shape[0], zs.shape[0]))
            pars_chain["T0"] = np.zeros((chain.shape[0], zs.shape[0]))

            pars_chain2 = {}
            pars_chain2["z"] = zs2
            # pars_chain2["tau_eff"] = np.zeros((chain.shape[0], zs2.shape[0]))
            pars_chain2["mF"] = np.zeros((chain.shape[0], zs2.shape[0]))
            pars_chain2["gamma"] = np.zeros((chain.shape[0], zs2.shape[0]))
            # pars_chain2["sigT_kms"] = np.zeros((chain.shape[0], zs2.shape[0]))
            pars_chain2["T0"] = np.zeros((chain.shape[0], zs2.shape[0]))

            for ii in range(chain.shape[0]):
                chain_params = self.parameters_from_sampling_point(chain[ii, :])
                # pars_chain["tau_eff"][ii] = self.theory.model_igm.models[
                #     "F_model"
                # ].get_tau_eff(zs, like_params=chain_params)
                pars_chain["mF"][ii] = self.theory.model_igm.models[
                    "F_model"
                ].get_mean_flux(zs, like_params=chain_params)
                pars_chain["gamma"][ii] = self.theory.model_igm.models[
                    "T_model"
                ].get_gamma(zs, like_params=chain_params)
                # pars_chain["sigT_kms"][ii] = self.theory.model_igm.models[
                #     "T_model"
                # ].get_sigT_kms(zs, like_params=chain_params)
                pars_chain["T0"][ii] = (
                    self.theory.model_igm.models["T_model"].get_T0(
                        zs, like_params=chain_params
                    )
                    / 1e4
                )

                pars_chain2["mF"][ii] = self.theory.model_igm.models[
                    "F_model"
                ].get_mean_flux(zs2, like_params=chain_params)
                pars_chain2["gamma"][ii] = self.theory.model_igm.models[
                    "T_model"
                ].get_gamma(zs2, like_params=chain_params)
                pars_chain2["T0"][ii] = (
                    self.theory.model_igm.models["T_model"].get_T0(
                        zs2, like_params=chain_params
                    )
                    / 1e4
                )

            tab_out = []
            tab_out.append(zs2)
            tab_out.append(
                np.percentile(pars_chain2["mF"], [16, 50, 84], axis=0)
            )
            tab_out.append(
                np.percentile(pars_chain2["T0"], [16, 50, 84], axis=0)
            )
            tab_out.append(
                np.percentile(pars_chain2["gamma"], [16, 50, 84], axis=0)
            )
            # print(np.percentile(pars_chain2["mF"], [16, 50, 84], axis=0))
            # print(np.percentile(pars_chain["mF"], [16, 50, 84], axis=0))
            pars_chain2 = 0

        if free_params is not None:
            if zmask is not None:
                zs = zmask
            else:
                zs = self.data.z
            pars_test = {}
            pars_test["z"] = zs
            pars_test["tau_eff"] = self.theory.model_igm.models[
                "F_model"
            ].get_tau_eff(zs, like_params=free_params)
            pars_test["mF"] = self.theory.model_igm.models[
                "F_model"
            ].get_mean_flux(zs, like_params=free_params)
            pars_test["gamma"] = self.theory.model_igm.models[
                "T_model"
            ].get_gamma(zs, like_params=free_params)
            pars_test["sigT_kms"] = self.theory.model_igm.models[
                "T_model"
            ].get_sigT_kms(zs, like_params=free_params)
            pars_test["T0"] = (
                self.theory.model_igm.models["T_model"].get_T0(
                    zs, like_params=free_params
                )
                / 1e4
            )
            pars_test["kF_kms"] = self.theory.model_igm.models[
                "P_model"
            ].get_kF_kms(zs, like_params=free_params)

        if plot_type == "all":
            fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
            arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
            latex_labs = [
                r"$\tau_\mathrm{eff}$",
                r"$\gamma$",
                r"$\sigma_\mathrm{T} [\mathrm{km\,s^{-1}}]$",
                r"$k_F$ [km/s]",
            ]
        elif plot_type == "tau_sigT":
            fig, ax = plt.subplots(
                3,
                1,
                figsize=(8, 10),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1]},
            )
            # arr_labs = ["tau_eff", "sigT_kms", "gamma"]
            # latex_labs = [
            #     r"$\tau_\mathrm{eff}$",
            #     r"$\sigma_\mathrm{T}\,\left[\mathrm{km\,s^{-1}}\right]$",
            #     r"$\gamma$",
            # ]
            arr_labs = ["mF", "T0", "gamma"]
            nexp_mF = 1
            latex_labs = [
                # r"$(1+z)\bar{F}$",
                r"$\bar{F}$",
                r"$T_0[K]/10^4$",
                r"$\gamma$",
            ]
            gal21, tu24 = others_igm()

        ax = ax.reshape(-1)

        for ii in range(len(arr_labs)):
            # if self.truth is not None:
            #     _ = pars_true[arr_labs[ii]] != 0
            #     ax[ii].plot(
            #         pars_true["z"][_],
            #         pars_true[arr_labs[ii]][_],
            #         "C0:o",
            #         alpha=0.75,
            #         label="true",
            #     )

            if cloud:
                for jj, sim_label in enumerate(self.theory.emu_igm_all):
                    if is_number_string(sim_label[-1]) == False:
                        continue
                    if jj == 0:
                        lab = "Training data"
                        alpha = 0.75
                    else:
                        lab = None
                        alpha = 0.1

                    _ = np.argwhere(
                        self.theory.emu_igm_all[sim_label][arr_labs[ii]] != 0
                    )[:, 0]
                    if len(_) > 0:
                        ax[ii].scatter(
                            self.theory.emu_igm_all[sim_label]["z"][_],
                            self.theory.emu_igm_all[sim_label][arr_labs[ii]][_],
                            marker=".",
                            color="C1",
                            alpha=alpha,
                            label=lab,
                            s=10,
                        )

            if chain is not None:
                _ = pars_fid[arr_labs[ii]] != 0
                norm = 1
                ax[ii].fill_between(
                    pars_fid["z"][_],
                    norm
                    * np.percentile(pars_chain[arr_labs[ii]][:, _], 5, axis=0),
                    norm
                    * np.percentile(pars_chain[arr_labs[ii]][:, _], 95, axis=0),
                    color="lightblue",
                    alpha=0.5,
                )
                ax[ii].fill_between(
                    pars_fid["z"][_],
                    norm
                    * np.percentile(pars_chain[arr_labs[ii]][:, _], 16, axis=0),
                    norm
                    * np.percentile(pars_chain[arr_labs[ii]][:, _], 84, axis=0),
                    color="C0",
                    alpha=0.5,
                )

            if free_params is not None:
                _ = pars_test[arr_labs[ii]] != 0
                if arr_labs[ii] == "mF":
                    # norm = (1 + pars_test["z"][_]) ** nexp_mF
                    norm = 1
                else:
                    norm = 1

                ax[ii].plot(
                    pars_test["z"][_],
                    norm * pars_test[arr_labs[ii]][_],
                    "C0:",
                    label=variation_label,
                    alpha=1,
                    lw=3,
                )

                if arr_labs[ii] == "mF":
                    lab = "tau_eff_znodes"
                elif arr_labs[ii] == "T0":
                    lab = "sigT_kms_znodes"
                else:
                    lab = arr_labs[ii] + "_znodes"
                if lab in self.args.fid_igm:
                    yy = np.interp(
                        self.args.fid_igm[lab],
                        pars_test["z"][_],
                        pars_test[arr_labs[ii]][_],
                    )
                    if arr_labs[ii] == "mF":
                        # norm = (1 + self.args.fid_igm[lab]) ** nexp_mF
                        norm = 1
                    else:
                        norm = 1
                    ax[ii].scatter(
                        self.args.fid_igm[lab],
                        norm * yy,
                        marker="o",
                        color="C0",
                    )

            if arr_labs[ii] == "mF":
                # norm = (1 + gal21["z"]) ** nexp_mF
                norm = 1
                ax[ii].errorbar(
                    gal21["z"],
                    norm * gal21["mF"],
                    yerr=norm * gal21["mF_err"],
                    fmt="--",
                    color="C1",
                    label="Gaikwad+2021",
                    alpha=0.75,
                    lw=2,
                )
                # norm = (1 + tu24["z"]) ** nexp_mF
                norm = 1
                ax[ii].errorbar(
                    tu24["z"],
                    norm * tu24["mF"],
                    yerr=norm * tu24["mF_err"],
                    fmt="-.",
                    color="C2",
                    label="Turner+2024",
                    alpha=0.75,
                    lw=2,
                )
            elif arr_labs[ii] == "T0":
                ax[ii].errorbar(
                    gal21["z"],
                    gal21["T0"],
                    yerr=gal21["T0_err"],
                    fmt="--",
                    color="C1",
                    label="Galdwick+2021",
                    alpha=0.75,
                    lw=2,
                )
            elif arr_labs[ii] == "gamma":
                ax[ii].errorbar(
                    gal21["z"],
                    gal21["gamma"],
                    yerr=gal21["gamma_err"],
                    fmt="--",
                    color="C1",
                    label="Galdwick+2021",
                    alpha=0.75,
                    lw=2,
                )

        if plot_more_igm:
            more_igm = np.load("more_igm_data.npy", allow_pickle=True).item()
            ax[0].plot(
                more_igm["z"],
                more_igm["mF"][1],
                "C5--",
                alpha=0.5,
                lw=3,
                label=r"IGM $n_z=8$",
            )
            ax[1].plot(
                more_igm["z"], more_igm["T0"][1], "C5--", alpha=0.5, lw=3
            )
            ax[2].plot(
                more_igm["z"], more_igm["gamma"][1], "C5--", alpha=0.5, lw=3
            )

        if plot_fid:
            for ii in range(len(arr_labs)):
                for kk in range(3):
                    if kk == 0:
                        pars = pars_fid.copy()
                        label = lab_fid
                        lsk = "-"
                        alpha = 0.5
                        lw = 1.5
                    elif kk == 1:
                        pars = pars_min.copy()
                        label = None
                        lsk = "-"
                        alpha = 0.3
                        lw = 1
                    elif kk == 2:
                        pars = pars_max.copy()
                        label = None
                        lsk = "-"
                        alpha = 0.3
                        lw = 1

                    _ = pars[arr_labs[ii]] != 0
                    if arr_labs[ii] == "mF":
                        # norm = (1 + pars["z"][_]) ** nexp_mF
                        norm = 1
                    else:
                        norm = 1

                    ax[ii].plot(
                        pars["z"][_],
                        norm * pars[arr_labs[ii]][_],
                        "C3" + lsk,
                        label=label,
                        alpha=alpha,
                        lw=lw,
                    )

        for ii in range(len(arr_labs)):
            ax[ii].set_ylabel(latex_labs[ii], fontsize=ftsize)
            if ii == 0:
                if arr_labs[ii] != "mF":
                    ax[ii].set_yscale("log")
                ax[ii].legend(fontsize=ftsize, loc="lower left", ncol=1)

            if (ii == 2) | (ii == len(arr_labs) - 1):
                ax[ii].set_xlabel(r"$z$", fontsize=ftsize)

            ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
            ax[ii].tick_params(axis="both", which="minor", labelsize=ftsize - 2)
            ax[ii].yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))

        if pre_xylims:
            ax[0].set_ylim(0.35, 0.9)
            ax[1].set_ylim(0.0, 3.2)
            ax[2].set_ylim(0.8, 2.2)
        fig.suptitle(title, fontsize=ftsize + 2)
        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "IGM_histories")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

        return tab_out

    def plot_cov_terms(self, save_directory=None):
        npanels = int(np.round(np.sqrt(len(self.cov_Pk_kms))))
        fig, ax = plt.subplots(
            npanels + 1, npanels, sharex=True, sharey=True, figsize=(10, 8)
        )
        ax = ax.reshape(-1)
        for ii in range(len(self.cov_Pk_kms)):
            cov_stat = np.diag(self.data.covstat_Pk_kms[ii])
            cov_syst = np.diag(self.data.cov_Pk_kms[ii]) - cov_stat
            cov_emu = np.diag(self.cov_emu_Pk_kms[ii])
            cov_tot = np.diag(self.cov_Pk_kms[ii])
            ax[ii].plot(
                self.data.k_kms[ii], cov_stat / cov_tot, label=r"$x$ = Stat"
            )
            ax[ii].plot(
                self.data.k_kms[ii], cov_syst / cov_tot, label=r"$x$ = Syst"
            )
            ax[ii].plot(
                self.data.k_kms[ii], cov_emu / cov_tot, label=r"$x$ = Emu"
            )
            ax[ii].text(0.0, 0.1, "z=" + str(self.data.z[ii]))
        if len(ax) > len(self.cov_Pk_kms):
            for ii in range(len(self.cov_Pk_kms), len(ax)):
                ax[ii].axis("off")
        ax[0].legend()
        fig.supxlabel(r"$k\,[\mathrm{km}^{-1}\mathrm{s}]$")
        fig.supylabel(r"$\sigma^2_x/\sigma^2_\mathrm{total}$")
        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(save_directory, "cov_terms")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_cov_to_pk(self, use_pk_smooth=True, fname=None, ftsize=18):
        npanels = int(np.round(np.sqrt(len(self.cov_Pk_kms))))

        fig, ax = plt.subplots(
            npanels + 1, npanels, sharex=True, sharey="row", figsize=(10, 8)
        )
        ax = ax.reshape(-1)
        for ii in range(len(self.cov_Pk_kms)):
            cov_stat = np.diag(self.data.covstat_Pk_kms[ii])
            cov_syst = np.diag(self.data.cov_Pk_kms[ii]) - cov_stat

            ind = np.argmin(np.abs(self.cov_factor["z"] - self.data.z[ii]))
            # inflate errors stat
            cov_stat = cov_stat * self.cov_factor["val_stat"][ind] ** 2
            # inflate errors syst
            cov_syst = cov_syst * self.cov_factor["val_syst"][ind] ** 2

            cov_emu = np.diag(self.cov_emu_Pk_kms[ii])
            cov_tot = np.diag(self.cov_Pk_kms[ii])
            if use_pk_smooth:
                pk = self.data.Pksmooth_kms[ii].copy()
            else:
                pk = self.data.Pk_kms[ii].copy()

            if ii == 0:
                lab = r"$x$ = Stat"
            else:
                lab = None
            ax[ii].plot(self.data.k_kms[ii], np.sqrt(cov_stat) / pk, label=lab)
            if ii == 1:
                lab = r"$x$ = Syst"
            else:
                lab = None
            ax[ii].plot(self.data.k_kms[ii], np.sqrt(cov_syst) / pk, label=lab)
            if ii == 2:
                lab = r"$x$ = Emu"
            else:
                lab = None
            ax[ii].plot(self.data.k_kms[ii], np.sqrt(cov_emu) / pk, label=lab)
            if ii == 3:
                lab = r"$x$ = Total"
            else:
                lab = None
            ax[ii].plot(self.data.k_kms[ii], np.sqrt(cov_tot) / pk, label=lab)
            ax[ii].text(
                0.95,
                0.2,
                "z=" + str(self.data.z[ii]),
                ha="right",
                va="top",
                transform=ax[ii].transAxes,
                fontsize=ftsize,
            )
            ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        if len(ax) > len(self.cov_Pk_kms):
            for ii in range(len(self.cov_Pk_kms), len(ax)):
                ax[ii].axis("off")

        for ii in range(4):
            ax[ii].legend(fontsize=ftsize - 2, loc="upper right")
        fig.supxlabel(
            r"$k_\parallel\,[\mathrm{km}^{-1}\mathrm{s}]$", fontsize=ftsize + 2
        )
        fig.supylabel(r"$\sigma_x/P_\mathrm{1D}$", fontsize=ftsize + 2)
        ax[0].set_ylim(0.0, 0.06)
        ax[3].set_ylim(0.0, 0.06)
        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname + ".pdf")
            plt.savefig(fname + ".png")
        else:
            plt.show()

    def plot_correlation_matrix(self, save_directory=None):
        def correlation_from_covariance(covariance):
            v = np.sqrt(np.diag(covariance))
            outer_v = np.outer(v, v)
            correlation = covariance / outer_v
            correlation[covariance == 0] = 0
            return correlation

        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        plt.imshow(correlation_from_covariance(self.full_cov_Pk_kms))
        plt.colorbar()

        if save_directory is not None:
            name = os.path.join(save_directory, "correlation")
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

    def plot_hull_fid(self, like_params=[]):
        emu_call, M_of_z = self.theory.get_emulator_calls(
            self.data.z, like_params=like_params
        )
        p1 = np.zeros(
            (
                self.theory.hull.nz,
                len(self.theory.hull.params),
            )
        )
        for jj, key in enumerate(self.theory.hull.params):
            p1[:, jj] = emu_call[key]

        self.theory.hull.plot_hulls(p1)

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
            iname = np.argwhere(
                np.array(dir_out["pnames"][iz]) == (pname + "_0")
            )[0, 0]
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
        self.theory.model_cont.hcd_model.reset_coeffs(
            free_params, rank=self.rank
        )
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
                    pname
                    + "_znodes not found in either fid_igm, fid_cont, or fid_syst"
                )

            p.fixed = isfixed

            # if (pname not in dir_out) and (pname == "kF_kms"):
            #     p.value = 1
            # elif (pname not in dir_out) and (pname == "HCD_const"):
            #     p.value = 0
            # elif (pname not in dir_out) and (pname == "HCD_damp2"):
            #     p.value = -4.5
            # elif (pname not in dir_out) and (pname == "HCD_damp3"):
            #     p.value = -5.3
            # elif (pname not in dir_out) and (pname == "R_coeff"):
            #     p.value = 0
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
        self.theory.model_igm.models["F_model"].reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_igm.models["T_model"].reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_cont.hcd_model.reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_cont.metal_models["Si_mult"].reset_coeffs(
            free_params, rank=self.rank
        )
        self.theory.model_cont.metal_models["Si_add"].reset_coeffs(
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
        np.array(
            [9500, 11000, 12750, 13500, 14750, 14750, 12750, 11250, 10250, 9250]
        )
        / 1e4
    )
    dT0 = (
        np.array([1393, 1028, 1132, 1390, 1341, 1322, 1493, 1125, 1070, 876])
        / 1e4
    )

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

    return gal21, tu24

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats.distributions import chi2 as chi2_scipy
from scipy.optimize import minimize

from lace.cosmo import camb_cosmo
from cup1d.utils.utils import is_number_string


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(
        self,
        data,
        theory,
        free_param_names=None,
        free_param_limits=None,
        verbose=False,
        prior_Gauss_rms=0.2,
        emu_cov_factor=None,
        extra_data=None,
        min_log_like=-1e100,
        args=None,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - free_param_names is a list of param names, in any order
        - free_param_limits list of tuples, same order than free_param_names
        - if prior_Gauss_rms is None it will use uniform priors
        - ignore k-bins with k > kmin_kms
        - emu_cov_factor adjusts the contribution from emulator covariance
        set between 0 and 1.
        - extra_p1d_data: extra P1D data, e.g., from HIRES
        - min_log_like: use this instead of - infinity"""

        self.verbose = verbose
        self.prior_Gauss_rms = prior_Gauss_rms
        self.emu_cov_factor = emu_cov_factor
        self.min_log_like = min_log_like
        self.data = data
        self.extra_data = extra_data
        # we only do this for latter save all relevant after fitting the model
        self.args = {}
        for attr, value in args.__dict__.items():
            if attr not in ["archive", "emulator"]:
                self.args[attr] = value

        self.theory = theory
        # Set inverse covariance. We do it here so we can account for emulator error
        self.set_icov()

        # setup parameters
        self.free_param_names = free_param_names
        self.set_free_parameters(free_param_names, free_param_limits)
        if verbose:
            print(len(self.free_params), "free parameters")

        # sometimes we want to know the true theory (when working with mocks)
        self.set_truth()

        # store also fiducial model
        self.set_fid()

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

        # Iterate over both datasets: main dataset (idata = 0) and additional dataset (idata = 1)
        for idata in range(2):
            if idata == 0:  # Main dataset
                data = self.data
                # Initialize list to store inverse covariance matrices for Pk_kms
                self.icov_Pk_kms = []
                self.cov_Pk_kms = []
                # Initialize the full inverse covariance matrix for Pk_kms
                self.full_icov_Pk_kms = None
                self.full_cov_Pk_kms = None
            else:  # Additional dataset
                data = self.extra_data
                # Initialize list for extra inverse covariance matrices
                self.extra_icov_Pk_kms = []
                self.extra_cov_Pk_kms = []
                # Initialize the full inverse covariance matrix for extra data
                self.extra_full_icov_Pk_kms = None
                self.extra_full_cov_Pk_kms = None

            if data is None:  # Skip if no data is provided for this dataset
                continue

            # Total number of k values across all redshifts
            nks = 0
            for ii in range(len(data.z)):
                nks += len(data.Pk_kms[ii])

            # Process each redshift bin
            for ii in range(len(data.z)):
                # Copy the covariance matrix for the current redshift bin
                cov = data.cov_Pk_kms[ii].copy()
                # Indices of the diagonal elements
                ind = np.arange(len(data.Pk_kms[ii]))
                # Add emulator error to the diagonal
                if self.emu_cov_factor is not None:
                    dkms_dMpc = self.theory.fid_cosmo["cosmo"].dkms_dMpc(
                        data.z[ii]
                    )
                    logk_Mpc = np.log(data.k_kms[ii] * dkms_dMpc)
                    rat = np.exp(np.poly1d(self.emu_cov_factor)(logk_Mpc))
                    # print(data.z[ii])
                    # print(data.k_kms[ii] * dkms_dMpc)
                    # print(rat)
                    cov[ind, ind] += (data.Pk_kms[ii] * rat) ** 2
                # Compute and store the inverse covariance matrix
                if idata == 0:
                    self.icov_Pk_kms.append(np.linalg.inv(cov))
                    self.cov_Pk_kms.append(cov)
                else:
                    self.extra_icov_Pk_kms.append(np.linalg.inv(cov))
                    self.extra_cov_Pk_kms.append(cov)

            # Process the full power spectrum data if available
            if data.full_Pk_kms is not None:
                # Copy the full covariance matrix
                cov = data.full_cov_Pk_kms.copy()
                # Indices of the diagonal elements
                ind = np.arange(len(data.full_Pk_kms))
                # Add emulator error to the diagonal
                if self.emu_cov_factor is not None:
                    dkms_dMpc = self.theory.fid_cosmo["cosmo"].dkms_dMpc(
                        data.z[ii]
                    )
                    logk_Mpc = np.log(data.full_k_kms * dkms_dMpc)
                    rat = np.exp(np.poly1d(self.emu_cov_factor)(logk_Mpc))
                    cov[ind, ind] += (data.full_Pk_kms * rat) ** 2

                # Compute and store the inverse covariance matrix
                if idata == 0:
                    self.full_icov_Pk_kms = np.linalg.inv(cov)
                    self.full_cov_Pk_kms = cov
                else:
                    self.extra_full_icov_Pk_kms = np.linalg.inv(cov)
                    self.extra_full_cov_Pk_kms = cov

    def set_free_parameters(self, free_param_names, free_param_limits):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params = []

        if free_param_limits is not None:
            assert len(free_param_limits) == len(
                free_param_names
            ), "wrong number of parameter limits"

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        ## select free parameters, make sure ordering
        ## in self.free_params is same as in free_param_names
        for par_name in free_param_names:
            for par in params:
                if par.name == par_name:
                    if free_param_limits is not None:
                        ## Set min and max of each parameter if
                        ## a list is given. otherwise leave as default
                        par.min_value = free_param_limits[
                            free_param_names.index(par.name)
                        ][0]
                        par.max_value = free_param_limits[
                            free_param_names.index(par.name)
                        ][1]
                    self.free_params.append(par)

        Nfree = len(self.free_params)
        Nin = len(free_param_names)

        assert Nfree == Nin, "could not setup free parameters"

        if self.verbose:
            print("likelihood setup with {} free parameters".format(Nfree))

        return

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
                    if ("tau" in par.name) and (
                        self.args["mF_model_type"] == "chunks"
                    ):
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
            else:
                if par.name not in self.truth["cont"]:
                    print("could not find {} in truth".format(par.name))
                    continue
                self.truth["like_params"][par.name] = self.truth["cont"][
                    par.name
                ]
                self.truth["like_params_cube"][
                    par.name
                ] = par.get_value_in_cube(self.truth["cont"][par.name])

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
        k_kms=None,
        values=None,
        return_covar=False,
        return_blob=False,
        return_emu_params=False,
    ):
        """Compute theoretical prediction for 1D P(k)"""

        if k_kms is None:
            k_kms = self.data.k_kms

        if zs is None:
            zs = self.data.z

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params = self.parameters_from_sampling_point(values)
        else:
            like_params = []

        return self.theory.get_p1d_kms(
            zs,
            k_kms,
            like_params=like_params,
            return_covar=return_covar,
            return_blob=return_blob,
            return_emu_params=return_emu_params,
        )

    def get_chi2(self, values=None, return_all=False):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""

        log_like, log_like_all = self.get_log_like(
            values, ignore_log_det_cov=True
        )

        if return_all:
            return -2.0 * log_like, -2.0 * log_like_all
        else:
            return -2.0 * log_like

    def get_log_like(
        self, values=None, ignore_log_det_cov=True, return_blob=False
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
        _res = self.get_p1d_kms(
            self.data.z,
            self.data.k_kms,
            values,
            return_blob=return_blob,
        )
        # out of priors
        if _res is None:
            return null_out

        if return_blob:
            emu_p1d, blob = _res
        else:
            emu_p1d = _res

        # use high-res data
        if self.extra_data is not None:
            length = 2
            nz = np.max([len(self.data.z), len(self.extra_data.z)])
            emu_p1d_extra = self.get_p1d_kms(
                self.extra_data.z,
                self.extra_data.k_kms,
                values,
                return_blob=False,
            )
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
                # compute chi2 for this redshift bin
                diff = data.Pk_kms[iz] - emu_p1d_use[iz]
                chi2_z = np.dot(np.dot(icov_Pk_kms[iz], diff), diff)
                # check whether to add determinant of covariance as well
                if ignore_log_det_cov:
                    log_like_all[ii, iz] = -0.5 * chi2_z
                else:
                    log_det_cov = np.log(
                        np.abs(1 / np.linalg.det(icov_Pk_kms[iz]))
                    )
                    log_like_all[ii, iz] = -0.5 * (chi2_z + log_det_cov)

            if full_icov_Pk_kms is None:
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
        self, values, return_blob=False, ignore_log_det_cov=False
    ):
        """Compute log likelihood plus log priors for input values
        - if return_blob==True, it will return also extra information"""

        # Always force parameter to be within range (for now)
        if (max(values) > 1.0) or (min(values) < 0.0):
            if return_blob:
                dummy_blob = self.theory.get_blob()
                return self.min_log_like, dummy_blob
            else:
                return self.min_log_like

        # compute log_prior
        log_prior = self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        if return_blob:
            log_like, chi2_all, blob = self.get_log_like(
                values, ignore_log_det_cov=ignore_log_det_cov, return_blob=True
            )
        else:
            log_like, chi2_all = self.get_log_like(
                values, ignore_log_det_cov=ignore_log_det_cov, return_blob=False
            )

        # regulate log-like (not NaN, not tiny)
        log_like = self.regulate_log_like(log_like)

        if return_blob:
            return log_like + log_prior, blob
        else:
            return log_like + log_prior

    def log_prob(self, values, ignore_log_det_cov=False):
        """Return log likelihood plus log priors"""

        return self.compute_log_prob(
            values, return_blob=False, ignore_log_det_cov=ignore_log_det_cov
        )

    def log_prob_and_blobs(self, values, ignore_log_det_cov=False):
        """Function used by emcee to get both log_prob and extra information"""

        lnprob, blob = self.compute_log_prob(
            values, return_blob=True, ignore_log_det_cov=ignore_log_det_cov
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

        if self.prior_Gauss_rms is None:
            return 0
        else:
            rms = self.prior_Gauss_rms
            fid_values = [p.value_in_cube() for p in self.free_params]
            log_prior = -np.sum(
                (np.array(fid_values) - values) ** 2 / (2 * rms**2)
            )
            return log_prior

    def minus_log_prob(self, values):
        """Return minus log_prob (needed to maximise posterior)"""

        return -1.0 * self.log_prob(values)

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
        sampling_p1d=100,
        return_covar=False,
        print_ratio=False,
        print_chi2=True,
        return_all=False,
        collapse=False,
    ):
        """Plot P1D in theory vs data. If plot_every_iz >1,
        plot only few redshift bins"""

        _res = self.get_p1d_kms(
            self.data.z, self.data.k_kms, values, return_covar=return_covar
        )
        if _res is None:
            return print("Prior out of range")
        if return_covar:
            emu_p1d, emu_cov = _res
        else:
            emu_p1d = _res

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

        # the sum of chi2_all may be different from chi2 due to covariance
        chi2, chi2_all = self.get_chi2(values=values, return_all=True)

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
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            length = 1
            ax = [ax]
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            length = 2

        # figure out y range for plot
        ymin = 1e10
        ymax = -1e10

        # print chi2
        n_free_p = len(self.free_params)
        ndeg = 0
        for iz in range(len(self.data.k_kms)):
            ndeg += np.sum(self.data.Pk_kms[iz] != 0)
        if self.extra_data is not None:
            for iz in range(len(self.extra_data.k_kms)):
                ndeg += np.sum(self.extra_data.Pk_kms[iz] != 0)
        prob = chi2_scipy.sf(chi2, ndeg - n_free_p)
        label = (
            r"$\chi^2=$"
            + str(np.round(chi2, 6))
            + " (ndeg="
            + str(ndeg - n_free_p)
            + ", prob="
            + str(np.round(prob * 100, 6))
            + "%)"
        )
        if print_chi2:
            ax[0].set_title(label, fontsize=14)

        out = {}

        for ii in range(length):
            if ii == 0:
                data = self.data
                emu_p1d_use = emu_p1d
                if return_covar:
                    emu_cov_use = emu_cov
                if rand_posterior is not None:
                    err_posterior_use = err_posterior
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
                out["extra_k_kms"] = []
                out["extra_p1d_data"] = []
                out["extra_p1d_model"] = []
                out["extra_p1d_err"] = []
                out["extra_chi2"] = []
                out["extra_prob"] = []

            zs = data.z
            Nz = len(zs)

            # plot only few redshifts for clarity
            for iz in range(0, Nz, plot_every_iz):
                # access data for this redshift
                z = zs[iz]
                k_kms = data.k_kms[iz]
                p1d_data = data.Pk_kms[iz]
                # p1d_cov = data.cov_Pk_kms[iz]
                p1d_cov = self.cov_Pk_kms[iz]
                p1d_err = np.sqrt(np.diag(p1d_cov))
                p1d_theory = emu_p1d_use[iz]
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
                    col = "blue"
                    yshift = 0

                if residuals:
                    # shift data in y axis for clarity
                    ax[ii].errorbar(
                        k_kms,
                        p1d_data / p1d_theory + yshift,
                        color=col,
                        yerr=p1d_err / p1d_theory,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )

                    # print chi2
                    xpos = k_kms[0]
                    ypos = 0.75 + yshift
                    ndeg = np.sum(p1d_data != 0)
                    prob = chi2_scipy.sf(chi2_all[ii, iz], ndeg - n_free_p)
                    label = (
                        r"$\chi^2=$"
                        + str(np.round(chi2_all[ii, iz], 2))
                        + " (ndeg="
                        + str(ndeg - n_free_p)
                        + ", prob="
                        + str(np.round(prob * 100, 2))
                        + "%)"
                    )
                    if print_chi2:
                        ax[ii].text(xpos, ypos, label, fontsize=10)

                    if print_ratio:
                        print(p1d_data / p1d_theory)
                    ymin = min(ymin, min(p1d_data / p1d_theory + yshift))
                    ymax = max(ymax, max(p1d_data / p1d_theory + yshift))
                    ax[ii].plot(
                        k_kms,
                        p1d_theory / p1d_theory + yshift,
                        color=col,
                        linestyle="dashed",
                    )
                    if return_covar | (rand_posterior is not None):
                        ax[ii].fill_between(
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

                    # print chi2
                    xpos = k_kms[-1] + 0.001
                    ypos = (p1d_theory * k_kms / np.pi)[-1]
                    ndeg = np.sum(p1d_data != 0)
                    prob = chi2_scipy.sf(chi2_all[ii, iz], ndeg - n_free_p)
                    label = (
                        r"$\chi^2=$"
                        + str(np.round(chi2_all[ii, iz], 2))
                        + " ("
                        + str(ndeg - n_free_p)
                        + ", "
                        + str(np.round(prob * 100, 2))
                        + "%)"
                    )
                    if print_chi2:
                        ax[ii].text(xpos, ypos, label, fontsize=8)

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

                if ii == 0:
                    out["k_kms"].append(k_kms)
                    out["p1d_data"].append(p1d_data)
                    out["p1d_model"].append(p1d_theory)
                    out["p1d_err"].append(p1d_err)
                    out["chi2"].append(chi2_all[ii, iz])
                    out["prob"].append(prob)
                else:
                    out["extra_k_kms"].append(k_kms)
                    out["extra_p1d_data"].append(p1d_data)
                    out["extra_p1d_model"].append(p1d_theory)
                    out["extra_p1d_err"].append(p1d_err)
                    out["extra_chi2"].append(chi2_all[ii, iz])
                    out["extra_prob"].append(prob)

            # ax[ii].plot(k_kms[0], 1, linestyle="-", label="Data", color="k")
            ax[ii].plot(k_kms[0], 1, linestyle="--", label="Fit", color="k")
            if residuals:
                ax[ii].legend()
            else:
                ax[ii].legend(loc="lower right", ncol=4)

            # ax[ii].set_xlim(min(k_kms[0]) - 0.001, max(k_kms[-1]) + 0.001)
            ax[ii].set_xlabel(r"$k_\parallel$ [s/km]")

            if residuals:
                ax[ii].set_ylabel(
                    r"$P_{\rm 1D}^{\rm data}/P_{\rm 1D}^{\rm fit}$"
                )
                ax[ii].set_ylim(ymin - 0.3, ymax + 0.3)
            else:
                ax[ii].set_ylim(0.8 * ymin, 1.3 * ymax)
                ax[ii].set_yscale("log")
                ax[ii].set_ylabel(
                    r"$k_\parallel \, P_{\rm 1D}(z, k_\parallel) / \pi$"
                )

        plt.tight_layout()
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

    def plot_igm(
        self,
        cloud=False,
        free_params=None,
        save_directory=None,
        stat_best_fit="mle",
    ):
        """Plot IGM histories"""

        # true IGM parameters
        if self.truth is not None:
            pars_true = {}
            pars_true["z"] = self.truth["igm"]["z"]
            pars_true["tau_eff"] = self.truth["igm"]["tau_eff"]
            pars_true["gamma"] = self.truth["igm"]["gamma"]
            pars_true["sigT_kms"] = self.truth["igm"]["sigT_kms"]
            pars_true["kF_kms"] = self.truth["igm"]["kF_kms"]

        pars_fid = {}
        pars_fid["z"] = self.fid["igm"]["z"]
        pars_fid["tau_eff"] = self.fid["igm"]["tau_eff"]
        pars_fid["gamma"] = self.fid["igm"]["gamma"]
        pars_fid["sigT_kms"] = self.fid["igm"]["sigT_kms"]
        pars_fid["kF_kms"] = self.fid["igm"]["kF_kms"]

        if free_params is not None:
            zs = self.fid["igm"]["z"]
            pars_test = {}
            pars_test["z"] = zs
            pars_test["tau_eff"] = self.theory.model_igm.F_model.get_tau_eff(
                zs, like_params=free_params
            )
            pars_test["gamma"] = self.theory.model_igm.T_model.get_gamma(
                zs, like_params=free_params
            )
            pars_test["sigT_kms"] = self.theory.model_igm.T_model.get_sigT_kms(
                zs, like_params=free_params
            )
            pars_test["kF_kms"] = self.theory.model_igm.P_model.get_kF_kms(
                zs, like_params=free_params
            )

        fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
        ax = ax.reshape(-1)

        arr_labs = ["tau_eff", "gamma", "sigT_kms", "kF_kms"]
        latex_labs = [
            r"$\tau_\mathrm{eff}$",
            r"$\gamma$",
            r"$\sigma_T$",
            r"$k_F$",
        ]

        for ii in range(len(arr_labs)):
            if self.truth is not None:
                _ = pars_true[arr_labs[ii]] != 0
                ax[ii].plot(
                    pars_true["z"][_],
                    pars_true[arr_labs[ii]][_],
                    "C0:o",
                    alpha=0.75,
                    label="true",
                )

            _ = pars_fid[arr_labs[ii]] != 0
            ax[ii].plot(
                pars_fid["z"][_],
                pars_fid[arr_labs[ii]][_],
                "C1s--",
                label="fiducial",
                alpha=0.75,
                lw=3,
            )

            if free_params is not None:
                _ = pars_test[arr_labs[ii]] != 0
                ax[ii].plot(
                    pars_test["z"][_],
                    pars_test[arr_labs[ii]][_],
                    "C2.-.",
                    label="fit",
                    alpha=0.75,
                    lw=3,
                )

            if cloud:
                for sim_label in self.theory.emu_igm_all:
                    if is_number_string(sim_label[-1]) == False:
                        continue

                    _ = np.argwhere(
                        self.theory.emu_igm_all[sim_label][arr_labs[ii]] != 0
                    )[:, 0]
                    if len(_) > 0:
                        ax[ii].scatter(
                            self.theory.emu_igm_all[sim_label]["z"][_],
                            self.theory.emu_igm_all[sim_label][arr_labs[ii]][_],
                            marker=".",
                            color="black",
                            alpha=0.05,
                        )

            ax[ii].set_ylabel(latex_labs[ii])
            if ii == 0:
                ax[ii].set_yscale("log")
                ax[ii].legend()

            if (ii == 2) | (ii == 3):
                ax[ii].set_xlabel(r"$z$")

        plt.tight_layout()

        if save_directory is not None:
            name = os.path.join(
                save_directory, "IGM_histories_" + stat_best_fit
            )
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".png")
        else:
            plt.show()

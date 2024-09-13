import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import minimize

from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP

from cup1d.likelihood import cosmologies
from cup1d.likelihood import lya_theory
from cup1d.nuisance import mean_flux_model
from cup1d.nuisance import thermal_model
from cup1d.nuisance import pressure_model


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(
        self,
        data,
        theory,
        emulator=None,
        free_param_names=None,
        free_param_limits=None,
        verbose=False,
        prior_Gauss_rms=0.2,
        kmin_kms=None,
        emu_cov_factor=1,
        extra_data=None,
        min_log_like=-1e100,
    ):
        """Setup likelihood from theory and data. Options:
        - data (required) is the data to model
        - theory (required) instance of lya_theory
        - emulator (optional) only needed if theory not provided
        - cosmo_fid_label (optional) to specify fiducial cosmology
                    default: use default Planck-like cosmology
                    truth: read true cosmology used in simulation
                    look at cosmologies.py for more options
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
        # (optionally) get rid of low-k data points
        if kmin_kms is not None:
            self.data.cull_data(kmin_kms)

        self.theory = theory

        # setup parameters
        self.set_free_parameters(free_param_names, free_param_limits)
        if verbose:
            print(len(self.free_params), "free parameters")

        # sometimes we want to know the true theory (when working with mocks)
        self.set_truth()

        return

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

    def set_truth(self, z_star=3.0, kp_kms=0.009):
        """Store true cosmology from the simulation used to make mock data"""

        # access true cosmology used in mock data
        if hasattr(self.data, "truth") == False:
            print("will not store truth, working with real data")
            self.truth = None
            return

        self.truth = {}
        for par in self.data.truth:
            self.truth[par] = self.data.truth[par]

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

    def get_chi2(self, values=None):
        """Compute chi2 using data and theory, without adding
        emulator covariance"""

        log_like = self.get_log_like(values, ignore_log_det_cov=True)
        if log_like is None:
            return None
        else:
            return -2.0 * log_like

    def get_log_like(
        self, values=None, ignore_log_det_cov=True, return_blob=False
    ):
        """Compute log(likelihood), including determinant of covariance
        unless you are setting ignore_log_det_cov=True."""

        # ask emulator prediction for P1D in each bin
        if self.emu_cov_factor == 0:
            return_covar = False
        else:
            return_covar = True

        _res = self.get_p1d_kms(
            self.data.z,
            self.data.k_kms,
            values,
            return_covar=return_covar,
            return_blob=return_blob,
        )
        if return_covar:
            if return_blob:
                emu_p1d, emu_covar, blob = _res
            else:
                emu_p1d, emu_covar = _res
        else:
            if return_blob:
                emu_p1d, blob = _res
            else:
                emu_p1d = _res

        if self.extra_data is not None:
            length = 2
            _res_hi = self.get_p1d_kms(
                self.extra_data.z,
                self.extra_data.k_kms,
                values,
                return_covar=return_covar,
                return_blob=False,
            )
            if return_covar:
                emu_p1d_extra, emu_covar_extra = _res_hi
            else:
                emu_p1d_extra = _res_hi
        else:
            length = 1

        if self.verbose:
            print("got P1D from emulator")

        # compute log like contribution from each redshift bin
        log_like = 0
        # loop over low and high res data
        for ii in range(length):
            if ii == 0:
                emu_p1d_use = emu_p1d
                data = self.data
                if return_covar:
                    emu_covar_use = emu_covar
            else:
                emu_p1d_use = emu_p1d_extra
                data = self.extra_data
                if return_covar:
                    emu_covar_use = emu_covar_extra

            # loop over redshift bins
            for iz in range(len(data.z)):
                # acess data for this redshift
                z = data.z[iz]
                # make sure that theory is valid
                if emu_p1d_use[iz] is None:
                    if self.verbose:
                        print(z, "theory did not emulate p1d")
                    return None
                if self.verbose:
                    print("compute chi2 for z={}".format(z))
                # get data
                p1d = data.get_Pk_iz(iz)
                # add covariance from emulator
                if return_covar:
                    icov = np.linalg.inv(
                        data.get_cov_iz(iz) + emu_cov_factor * emu_covar_use[iz]
                    )
                else:
                    icov = data.get_icov_iz(iz)

                # compute chi2 for this redshift bin
                diff = p1d - emu_p1d_use[iz]
                chi2_z = np.dot(np.dot(icov, diff), diff)
                # check whether to add determinant of covariance as well
                if ignore_log_det_cov:
                    log_like_z = -0.5 * chi2_z
                else:
                    log_det_cov = np.log(np.abs(1 / np.linalg.det(icov)))
                    log_like_z = -0.5 * (chi2_z + log_det_cov)
                log_like += log_like_z
                if self.verbose:
                    print("added {} to log_like".format(log_like_z))

        if return_blob:
            return log_like, blob
        else:
            return log_like

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
            log_like, blob = self.get_log_like(
                values, ignore_log_det_cov=ignore_log_det_cov, return_blob=True
            )
        else:
            log_like = self.get_log_like(
                values, ignore_log_det_cov=ignore_log_det_cov, return_blob=False
            )

        # # if required, add extra P1D likelihood from, e.g., HIRES
        # if self.extra_p1d_like:
        #     extra_log_like = self.extra_p1d_like.get_log_like(
        #         values, ignore_log_det_cov=False, return_blob=False
        #     )
        #     log_like += extra_log_like

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
        show=False,
        sampling_p1d=100,
        return_covar=False,
    ):
        """Plot P1D in theory vs data. If plot_every_iz >1,
        plot only few redshift bins"""

        # get measured bins from data
        Nz = len(self.data.z)
        k_emu_kms = np.zeros((Nz, sampling_p1d))
        for iz in range(Nz):
            k_emu_kms[iz] = np.logspace(
                np.log10(min(self.data.k_kms[iz])),
                np.log10(max(self.data.k_kms[iz])),
                sampling_p1d,
            )
        if self.extra_data is not None:
            Nz = len(self.extra_data.z)
            k_emu_kms_extra = np.zeros((Nz, sampling_p1d))
            for iz in range(Nz):
                k_emu_kms_extra[iz] = np.logspace(
                    np.log10(min(self.extra_data.k_kms[iz])),
                    np.log10(max(self.extra_data.k_kms[iz])),
                    sampling_p1d,
                )

        _res = self.get_p1d_kms(
            self.data.z, k_emu_kms, values, return_covar=return_covar
        )
        if return_covar:
            emu_p1d, emu_cov = _res
        else:
            emu_p1d = _res
        if self.extra_data is not None:
            _res = self.get_p1d_kms(
                self.extra_data.z,
                k_emu_kms_extra,
                values,
                return_covar=return_covar,
            )
            if return_covar:
                emu_p1d_extra, emu_cov_extra = _res
            else:
                emu_p1d_extra = _res

        # figure out y range for plot
        ymin = 1e10
        ymax = -1e10

        if rand_posterior is not None:
            Nz = len(self.data.z)
            rand_emu = np.zeros((rand_posterior.shape[0], Nz, len(k_emu_kms)))
            for ii in range(rand_posterior.shape[0]):
                rand_emu[ii] = self.get_p1d_kms(
                    self.data.z, k_emu_kms, rand_posterior[ii]
                )
            err_posterior = np.std(rand_emu, axis=0)

            if self.extra_data is not None:
                Nz = len(self.extra_data.z)
                rand_emu_extra = np.zeros(
                    (rand_posterior.shape[0], Nz, len(k_emu_kms_extra))
                )
                for ii in range(rand_posterior.shape[0]):
                    rand_emu_extra[ii] = self.get_p1d_kms(
                        self.extra_data.z, k_emu_kms_extra, rand_posterior[ii]
                    )
                err_posterior_extra = np.std(rand_emu_extra, axis=0)

        if self.extra_data is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            length = 1
            ax = [ax]
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            length = 2

        for ii in range(length):
            if ii == 0:
                data = self.data
                emu_p1d_use = emu_p1d
                k_emu_kms_use = k_emu_kms
                if return_covar:
                    emu_cov_use = emu_cov
                if rand_posterior is not None:
                    err_posterior_use = err_posterior
            else:
                data = self.extra_data
                emu_p1d_use = emu_p1d_extra
                k_emu_kms_use = k_emu_kms_extra
                if return_covar:
                    emu_cov_use = emu_cov_extra
                if rand_posterior is not None:
                    err_posterior_use = err_posterior_extra
            zs = data.z
            k_kms = data.k_kms
            Nz = len(zs)

            # plot only few redshifts for clarity
            for iz in range(0, Nz, plot_every_iz):
                # access data for this redshift
                z = zs[iz]
                p1d_data = data.get_Pk_iz(iz)
                p1d_cov = data.get_cov_iz(iz)
                p1d_err = np.sqrt(np.diag(p1d_cov))
                p1d_theory = emu_p1d_use[iz]
                if rand_posterior is None:
                    if return_covar:
                        cov_theory = emu_cov_use[iz]
                        err_theory = np.sqrt(np.diag(cov_theory))
                else:
                    err_theory = err_posterior_use[iz]

                if p1d_theory is None:
                    if self.verbose:
                        print(z, "emulator did not provide P1D")
                    continue
                # plot everything
                if Nz > 1:
                    col = plt.cm.jet(iz / (Nz - 1))
                    yshift = iz / (Nz - 1)
                else:
                    col = "blue"
                    yshift = 0

                if residuals:
                    # interpolate theory to data kp values
                    model = np.interp(k_kms[iz], k_emu_kms_use[iz], p1d_theory)
                    # shift data in y axis for clarity
                    ax[ii].errorbar(
                        k_kms[iz],
                        p1d_data / model + yshift,
                        color=col,
                        yerr=p1d_err / model,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )
                    ymin = min(ymin, min(p1d_data / model + yshift))
                    ymax = max(ymax, max(p1d_data / model + yshift))
                    ax[ii].plot(
                        k_kms[iz],
                        model / model + yshift,
                        color=col,
                        linestyle="dashed",
                    )
                    if return_covar | (rand_posterior is not None):
                        err_model = np.interp(
                            k_kms[iz], k_emu_kms_use[iz], err_theory
                        )
                        ax[ii].fill_between(
                            k_kms[iz],
                            (model + err_model) / model + yshift,
                            (model - err_model) / model + yshift,
                            alpha=0.35,
                            color=col,
                        )
                else:
                    _ = (k_emu_kms_use[iz] >= np.min(k_kms[iz]) * 0.9) & (
                        k_emu_kms_use[iz] <= np.max(k_kms[iz]) * 1.1
                    )
                    ax[ii].errorbar(
                        k_kms[iz],
                        p1d_data * k_kms[iz] / np.pi,
                        color=col,
                        yerr=p1d_err * k_kms[iz] / np.pi,
                        fmt="o",
                        ms="4",
                        label="z=" + str(np.round(z, 2)),
                    )
                    ax[ii].plot(
                        k_emu_kms_use[iz][_],
                        (p1d_theory[_] * k_emu_kms_use[iz][_]) / np.pi,
                        color=col,
                        linestyle="dashed",
                    )
                    if return_covar | (rand_posterior is not None):
                        ax[ii].fill_between(
                            k_emu_kms_use[iz][_],
                            (p1d_theory[_] + err_theory[_])
                            * k_emu_kms_use[iz][_]
                            / np.pi,
                            (p1d_theory[_] - err_theory[_])
                            * k_emu_kms_use[iz][_]
                            / np.pi,
                            alpha=0.35,
                            color=col,
                        )
                    ymin = min(ymin, min(p1d_data * k_kms[iz] / np.pi))
                    ymax = max(ymax, max(p1d_data * k_kms[iz] / np.pi))

            ax[ii].plot(
                k_emu_kms_use[iz][0], 1, linestyle="-", label="Data", color="k"
            )
            ax[ii].plot(
                k_emu_kms_use[iz][0], 1, linestyle=":", label="Fit", color="k"
            )
            ax[ii].legend()

            # ax[ii].set_xlim(min(k_kms[iz]) - 0.001, max(k_kms[iz]) + 0.001)
            ax[ii].set_xlabel(r"$k_\parallel$ [s/km]")

            if residuals:
                ax[ii].set_ylabel(r"$P_{\rm 1D}(z,k_\parallel)$ residuals")
                # ax[ii].set_ylim(ymin - 0.1, ymax + 0.1)
            else:
                ax[ii].set_ylim(0.8 * ymin, 1.2 * ymax)
                ax[ii].set_yscale("log")
                ax[ii].set_ylabel(
                    r"$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$"
                )

        plt.tight_layout()
        if plot_fname:
            plt.savefig(plot_fname)

        if show:
            plt.show()

        return

    def overplot_emulator_calls(
        self,
        param_1,
        param_2,
        values=None,
        tau_scalings=True,
        temp_scalings=True,
    ):
        """For parameter pair (param1,param2), overplot emulator calls
        with values stored in archive, color coded by redshift"""

        # mask post-process scalings (optional)
        emu_data = self.theory.emulator.archive.data
        Nemu = len(emu_data)
        if not tau_scalings:
            mask_tau = [x["scale_tau"] == 1.0 for x in emu_data]
        else:
            mask_tau = [True] * Nemu
        if not temp_scalings:
            mask_temp = [
                (x["scale_T0"] == 1.0) & (x["scale_gamma"] == 1.0)
                for x in emu_data
            ]
        else:
            mask_temp = [True] * Nemu

        # figure out values of param_1,param_2 in archive
        emu_1 = np.array(
            [
                emu_data[i][param_1]
                for i in range(Nemu)
                if (mask_tau[i] & mask_temp[i])
            ]
        )
        emu_2 = np.array(
            [
                emu_data[i][param_2]
                for i in range(Nemu)
                if (mask_tau[i] & mask_temp[i])
            ]
        )

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params = self.parameters_from_sampling_point(values)
        else:
            like_params = []
        emu_calls = self.theory.get_emulator_calls(like_params=like_params)
        # figure out values of param_1,param_2 called
        call_1 = [emu_call[param_1] for emu_call in emu_calls]
        call_2 = [emu_call[param_2] for emu_call in emu_calls]

        # overplot
        zs = self.data.z
        emu_z = np.array(
            [
                emu_data[i]["z"]
                for i in range(Nemu)
                if (mask_tau[i] & mask_temp[i])
            ]
        )
        zmin = min(min(emu_z), min(zs))
        zmax = max(max(emu_z), max(zs))
        plt.scatter(emu_1, emu_2, c=emu_z, s=1, vmin=zmin, vmax=zmax)
        plt.scatter(call_1, call_2, c=zs, s=50, vmin=zmin, vmax=zmax)
        cbar = plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return

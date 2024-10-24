import os, time, emcee, json

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chainconsumer import ChainConsumer, Chain, Truth
from warnings import warn

import numdifftools as nd
from pyDOE2 import lhs

# import multiprocessing as mp
# from schwimmbad import MPIPool
from mpi4py import MPI

# our own modules
import cup1d, lace
from lace.cosmo import fit_linP, camb_cosmo
from lace.archive import gadget_archive
from lace.emulator import gp_emulator, nn_emulator
from cup1d.p1ds import (
    data_nyx,
    data_gadget,
    mock_data,
    data_Chabanier2019,
    data_Karacayli2022,
)
from cup1d.likelihood import likelihood
from cup1d.utils.utils import create_print_function


def purge_chains(ln_prop_chains, nsplit=7, abs_diff=5):
    """Purge emcee chains that have not converged"""
    minval = np.median(ln_prop_chains) - 10
    # split each walker in nsplit chunks
    split_arr = np.array_split(ln_prop_chains, nsplit, axis=0)
    # compute median of each chunck
    split_med = []
    for ii in range(nsplit):
        split_med.append(split_arr[ii].mean(axis=0))
    # (nwalkers, nchucks)
    split_res = np.array(split_med).T
    # compute median of chunks for each walker ()
    split_res_med = split_res.mean(axis=1)

    # step-dependence convergence
    # check that average logprob does not vary much with step
    # compute difference between chunks and median of each chain
    keep1 = (np.abs(split_res - split_res_med[:, np.newaxis]) < abs_diff).all(
        axis=1
    )
    # total-dependence convergence
    # check that average logprob is close to minimum logprob of all chains
    # check that all chunks are above a target minimum value
    keep2 = (split_res > minval).all(axis=1)

    # combine both criteria
    both = keep1 & keep2
    keep = np.argwhere(both)[:, 0]
    keep_not = np.argwhere(both == False)[:, 0]

    return keep, keep_not


class Fitter(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(
        self,
        like=None,
        nwalkers=None,
        nsteps=None,
        nburnin=None,
        read_chain_file=None,
        verbose=False,
        subfolder=None,
        rootdir=None,
        save_chain=True,
        progress=False,
        get_autocorr=False,
        parallel=False,
        explore=False,
        fix_cosmology=False,
    ):
        """Setup sampler from likelihood, or use default.
        If read_chain_file is provided, read pre-computed chain.
        rootdir allows user to search for saved chains in a different
        location to the code itself."""

        self.parallel = parallel
        self.explore = explore
        self.verbose = verbose
        self.progress = progress
        self.get_autocorr = get_autocorr
        self.burnin_nsteps = nburnin

        self.param_dict = param_dict

        if self.parallel:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        self.fix_cosmology = fix_cosmology

        self.print = create_print_function(self.verbose)

        if read_chain_file:
            self.print("will read chain from file", read_chain_file)
            assert not like, "likelihood specified but reading chain from file"
            self.read_chain_from_file(read_chain_file, rootdir, subfolder)
            self.burnin_pos = None
        else:
            self.like = like
            # number of free parameters to sample
            self.ndim = len(self.like.free_params)

            if self.rank == 0:
                self._setup_chain_folder(rootdir, subfolder)

            if save_chain:
                backend_string = self.save_directory + "/backend.h5"
                self.backend = emcee.backends.HDFBackend(backend_string)
            else:
                self.backend = None

            # number of walkers
            if nwalkers:
                if nwalkers > 2 * self.ndim:
                    self.nwalkers = nwalkers
                else:
                    self.print(
                        "nwalkers={} ; ndim={}".format(nwalkers, self.ndim)
                    )
                    raise ValueError("specified number of walkers too small")
                self.nsteps = nsteps
            else:
                max_walkers = 40 * self.ndim
                min_walkers = 2 * self.ndim
                nwalkers = max_walkers // self.size + 1
                combined_steps = max_walkers * (nsteps + self.burnin_nsteps)

                if nwalkers < min_walkers:
                    nwalkers = min_walkers
                    nsteps = (
                        combined_steps // (nwalkers * self.size)
                        - self.burnin_nsteps
                    )

                self.nwalkers = nwalkers
                self.nsteps = nsteps

            self.print(
                "setup with ",
                self.size,
                " ranks, ",
                self.nwalkers,
                " walkers, and ",
                self.nsteps,
                " steps",
            )
            self.print(
                "combined steps ",
                self.nwalkers * self.size * (self.nsteps + self.burnin_nsteps),
                "(should be close to ",
                combined_steps,
                ")",
            )

        ## Set up list of parameter names in tex format for plotting
        self.paramstrings = []
        for param in self.like.free_params:
            self.paramstrings.append(param_dict[param.name])

        # when running on simulated data, we can store true cosmo values
        self.set_truth()

        # Figure out what extra information will be provided as blobs
        self.blobs_dtype = self.like.theory.get_blobs_dtype()
        self.mle = None

    def set_truth(self):
        """Set up dictionary with true values of cosmological
        likelihood parameters for plotting purposes"""

        # likelihood contains true parameters, but not in latex names
        like_truth = self.like.truth

        # when running on data, we do not know the truth
        if like_truth is None:
            self.truth = None
            return

        # store truth for all parameters, with LaTeX keywords
        self.truth = {}
        for param in like_truth["like_params"]:
            if param in param_dict:
                self.truth[param_dict[param]] = like_truth["like_params"][param]
            else:
                self.truth[param] = like_truth["like_params"][param]

    def run_sampler(
        self,
        pini=None,
        log_func=None,
        timeout=None,
        force_timeout=False,
    ):
        """Set up sampler, run burn in, run chains,
        return chains
            - timeout is the time in hours to run the
              sampler for
            - force_timeout will continue to run the chains
              until timeout, regardless of convergence"""
        if self.get_autocorr:
            # We'll track how the average autocorrelation time estimate changes
            self.autocorr = np.array([])
            # This will be useful to testing convergence
            old_tau = np.inf

        if self.parallel == False:
            ## Get initial walkers
            p0 = self.get_initial_walkers(pini=pini)
            if log_func is None:
                log_func = self.like.log_prob_and_blobs
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                log_func,
                backend=self.backend,
                blobs_dtype=self.blobs_dtype,
            )
            for sample in sampler.sample(
                p0,
                iterations=self.burnin_nsteps + self.nsteps,
                progress=self.progress,
            ):
                if sampler.iteration % 100 == 0:
                    self.print(
                        "Step %d out of %d "
                        % (sampler.iteration, self.burnin_nsteps + self.nsteps)
                    )

                # if self.get_autocorr:
                #     # Compute the autocorrelation time so far
                #     # Using tol=0 means that we'll always get an estimate even
                #     # if it isn't trustworthy
                #     tau = sampler.get_autocorr_time(
                #         tol=0, discard=self.burnin_nsteps
                #     )
                #     self.autocorr = np.append(self.autocorr, np.mean(tau))

                #     # Check convergence
                #     converged = np.all(tau * 100 < sampler.iteration)
                #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                #     if converged:
                #         break
                #     old_tau = tau

            ## Get samples, flat=False to be able to mask not converged chains latter
            self.lnprob = sampler.get_log_prob(
                flat=False, discard=self.burnin_nsteps
            )
            self.chain = sampler.get_chain(
                flat=False, discard=self.burnin_nsteps
            )
            self.blobs = sampler.get_blobs(
                flat=False, discard=self.burnin_nsteps
            )
        else:
            # MPIPool does not work in nersc for whatever reason
            # I need to get creative

            np.random.seed(self.rank)
            p0 = self.get_initial_walkers(pini=pini)
            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, log_func, blobs_dtype=self.blobs_dtype
            )

            for sample in sampler.sample(
                p0, iterations=self.burnin_nsteps + self.nsteps
            ):
                if sampler.iteration % 100 == 0:
                    self.print(
                        "Step %d out of %d "
                        % (sampler.iteration, self.nsteps + self.burnin_nsteps)
                    )

            print(f"Rank {self.rank} done", flush=True)
            _lnprob = sampler.get_log_prob(
                flat=False, discard=self.burnin_nsteps
            )
            _chain = sampler.get_chain(flat=False, discard=self.burnin_nsteps)
            _blobs = sampler.get_blobs(flat=False, discard=self.burnin_nsteps)

            if self.rank != 0:
                self.comm.send(_lnprob, dest=0, tag=1000 + self.rank)
                self.comm.send(_chain, dest=0, tag=2000 + self.rank)
                self.comm.send(_blobs, dest=0, tag=3000 + self.rank)

            if self.rank == 0:
                chain = []
                lnprob = []
                blobs = []

                lnprob.append(_lnprob)
                chain.append(_chain)
                blobs.append(_blobs)

                for irank in range(1, self.size):
                    self.print("Receiving from rank %d" % irank)
                    lnprob.append(
                        self.comm.recv(source=irank, tag=1000 + irank)
                    )
                    chain.append(self.comm.recv(source=irank, tag=2000 + irank))
                    blobs.append(self.comm.recv(source=irank, tag=3000 + irank))

                self.lnprob = np.concatenate(lnprob, axis=1)
                self.chain = np.concatenate(chain, axis=1)
                self.blobs = np.concatenate(blobs, axis=1)

        # if not MPIPool.enabled():
        #     raise SystemError(
        #         "Tried to run with MPI but MPIPool not enabled!"
        #     )

        # with MPIPool() as pool:
        #     if not pool.is_master():
        #         pool.wait()
        #         sys.exit(0)

        #     print("\n Running with MPI on {0} cores \n".format(pool.size))

        #     sampler.run_mcmc(p0, burn_in + max_steps)
        # sampler.pool = pool
        # print(f"Number of threads being used: {pool._processes}")
        # sampler.run_mcmc(p0, burn_in + max_steps)
        # p0 = self.get_initial_walkers()
        # mp.set_start_method("spawn")
        # with mp.Pool() as pool:
        #     sampler = emcee.EnsembleSampler(
        #         self.nwalkers,
        #         self.ndim,
        #         log_func,
        #         backend=self.backend,
        #         blobs_dtype=self.blobs_dtype,
        #         pool=pool,
        #     )
        #     sampler.pool = pool
        #     print(f"Number of threads being used: {pool._processes}")
        #     sampler.run_mcmc(p0, burn_in + max_steps)
        # if timeout:
        #     time_end = time.time() + 3600 * timeout

        # for sample in sampler.sample(
        #     p0, iterations=burn_in + max_steps, progress=self.progress
        # ):
        #     print(sampler.iteration, flush=True)
        #     # Only check convergence every 100 steps
        #     if (
        #         sampler.iteration % 100
        #         or sampler.iteration < burn_in + 1
        #     ):
        #         continue

        #     if self.progress == False:
        #         print(
        #             "Step %d out of %d "
        #             % (sampler.iteration, burn_in + max_steps)
        #         )

        #     if self.get_autocorr:
        #         # Compute the autocorrelation time so far
        #         # Using tol=0 means that we'll always get an estimate even
        #         # if it isn't trustworthy
        #         tau = sampler.get_autocorr_time(tol=0, discard=burn_in)
        #         self.autocorr = np.append(self.autocorr, np.mean(tau))

        #         # Check convergence
        #         converged = np.all(tau * 100 < sampler.iteration)
        #         converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

        #         ## Check if we are over time limit
        #         if timeout:
        #             if time.time() > time_end:
        #                 print("Timed out")
        #                 break
        #         ## If not, only halt on convergence criterion if
        #         ## force_timeout is false
        #         if (force_timeout == False) and (converged == True):
        #             print("Chains have converged")
        #             break
        #         old_tau = tau

        return sampler

    def run_minimizer(self, log_func_minimize=None, p0=None, nsamples=8):
        """Minimizer"""
        npars = len(self.like.free_params)

        if p0 is None:
            arr_p0 = lhs(npars, samples=nsamples)

            mle = None
            chi2 = 1e10

            for ii in range(nsamples):
                res = scipy.optimize.minimize(
                    log_func_minimize,
                    arr_p0[ii],
                    method="Nelder-Mead",
                    bounds=((0.0, 1.0),) * npars,
                )
                if res.fun < chi2:
                    chi2 = res.fun
                    mle = res.x

            # start at the minimum
            res = scipy.optimize.minimize(
                log_func_minimize,
                mle,
                method="Nelder-Mead",
                bounds=((0.0, 1.0),) * npars,
            )
            if res.fun < chi2:
                chi2 = res.fun
                mle = res.x

            print("Minimization:", chi2, flush=True)
        else:
            chi2_ini = log_func_minimize(p0)

            for ii in range(2):
                if ii == 0:
                    ini = p0.copy()
                else:
                    ini = mle.copy()

                _ = ini <= 0
                ini[_] = 0.01
                _ = ini >= 1
                ini[_] = 0.99

                res = scipy.optimize.minimize(
                    log_func_minimize,
                    ini,
                    method="Nelder-Mead",
                    bounds=((0.0, 1.0),) * npars,
                )
                mle = res.x
                chi2 = res.fun
            print("Minimization improved:", chi2_ini, chi2, flush=True)

        self.mle_cube = mle
        mle_no_cube = mle.copy()
        for ii, par_i in enumerate(self.like.free_params):
            scale_i = par_i.max_value - par_i.min_value
            mle_no_cube[ii] = par_i.value_from_cube(mle[ii])

        print("Fit params cube:", self.mle_cube, flush=True)
        print("Fit params no cube:", mle_no_cube, flush=True)

        self.mle_cosmo = self.get_cosmo_err(log_func_minimize)

        self.lnprop_mle, *blobs = self.like.log_prob_and_blobs(self.mle_cube)

        # Array for all parameters
        all_params = np.hstack([mle_no_cube, np.array(blobs)])
        # Ordered strings for all parameters
        all_strings = self.paramstrings + blob_strings

        self.mle = {}
        for ii, par in enumerate(all_strings):
            self.mle[par] = all_params[ii]

        if "A_s" not in self.paramstrings[0]:
            return

        for par in self.mle_cosmo:
            if par == "Delta2_star":
                print("MLE, err")
                if self.like.truth is not None:
                    print("Truth, MLE - Truth")

            if "err" in par:
                continue
            print(par)
            print(
                np.round(self.mle_cosmo[par], 5),
                np.round(self.mle_cosmo["err_" + par], 5),
            )
            if self.like.truth is not None:
                if par in self.like.truth["like_params"]:
                    print(
                        np.round(self.like.truth["like_params"][par], 5),
                        np.round(
                            np.abs(
                                self.like.truth["like_params"][par]
                                - self.mle_cosmo[par]
                            ),
                            5,
                        ),
                    )

    def get_cosmo_err(self, fun_minimize):
        hess = nd.Hessian(fun_minimize)
        ii = 0
        for par_i in self.like.free_params:
            if par_i.name == "As":
                ii += 1
            elif par_i.name == "ns":
                ii += 1
            elif par_i.name == "nrun":
                ii += 1

        if ii == 0:
            return

        cov = hess(self.mle_cube)[:ii, :ii]
        mle_cov_cube = np.linalg.inv(cov)

        mle_cov = np.zeros((3, 3))
        for par_i in self.like.free_params:
            if par_i.name == "As":
                ii = 0
            elif par_i.name == "ns":
                ii = 1
            elif par_i.name == "nrun":
                ii = 2
            else:
                continue
            scale_i = par_i.max_value - par_i.min_value
            for par_j in self.like.free_params:
                if par_j.name == "As":
                    jj = 0
                elif par_j.name == "ns":
                    jj = 1
                elif par_j.name == "nrun":
                    jj = 2
                else:
                    continue
                scale_j = par_j.max_value - par_j.min_value
                mle_cov[ii, jj] = mle_cov_cube[ii, jj] * scale_i * scale_j

        like_pars = self.like.parameters_from_sampling_point(self.mle_cube)

        res = self.like.theory.get_err_linP_Mpc_params(like_pars, mle_cov)

        return res

    def resume_sampler(
        self, max_steps, log_func=None, timeout=None, force_timeout=False
    ):
        """Use the emcee backend to restart a chain from the last iteration
        - max_steps is the maximum number of steps for this run
        - log_func should be sampler.like.log_prob_and_blobs
          with pool apparently
        - timeout is the amount of time to run in hours before wrapping
          the job up. This is used to make sure timeouts on compute nodes
          don't corrupt the backend
        - force_timeout will force chains to run for the time duration set
          instead of cutting at autocorrelation time convergence"""

        ## Make sure we have a backend
        assert self.backend is not None, "No backend found, cannot run sampler"
        old_tau = np.inf
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                self.backend.shape[0],
                self.backend.shape[1],
                log_func,
                pool=pool,
                backend=self.backend,
                blobs_dtype=self.blobs_dtype,
            )
            if timeout:
                time_end = time.time() + 3600 * timeout
            start_step = self.backend.iteration
            for sample in sampler.sample(
                self.backend.get_last_sample(),
                iterations=max_steps,
                progress=self.progress,
            ):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                if self.progress == False:
                    self.print(
                        "Step %d out of %d "
                        % (self.backend.iteration, start_step + max_steps)
                    )

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                self.autocorr = np.append(self.autocorr, np.mean(tau))

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if force_timeout == False:
                    if converged:
                        self.print("Chains have converged")
                        break
                if timeout:
                    if time.time() > time_end:
                        self.print("Timed out")
                        break
                old_tau = tau

        self.chain = sampler.get_chain(flat=False, discard=self.burnin_nsteps)
        self.lnprob = sampler.get_log_prob(
            flat=False, discard=self.burnin_nsteps
        )
        self.blobs = sampler.get_blobs(flat=False, discard=self.burnin_nsteps)

        return

    def get_initial_walkers(self, pini=None, rms=0.05):
        """Setup initial states of walkers in sensible points
        -- initial will set a range within unit volume around the
           fiducial values to initialise walkers (if no prior is used)"""

        ndim = self.ndim
        nwalkers = self.nwalkers

        self.print("set %d walkers with %d dimensions" % (nwalkers, ndim))

        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        for ii in range(ndim):
            if pini is None:
                p0[:, ii] = 0.5 + p0[:, ii] * rms
            else:
                p0[:, ii] = pini[ii] + p0[:, ii] * rms
        _ = p0 >= 1.0
        p0[_] = 0.95
        _ = p0 <= 0.0
        p0[_] = 0.05

        return p0

    def get_trunc_norm(self, mean, n_samples):
        """Wrapper for scipys truncated normal distribution
        Runs in the range [0,1] with a rms specified on initialisation"""

        rms = self.like.prior_Gauss_rms
        values = scipy.stats.truncnorm.rvs(
            (0.0 - mean) / rms,
            (1.0 - mean) / rms,
            scale=rms,
            loc=mean,
            size=n_samples,
        )

        return values

    def get_chain(self, cube=True, extra_nburn=0, delta_lnprob_cut=None):
        """Figure out whether chain has been read from file, or computed.
        - if cube=True, return values in range [0,1]
        - if delta_lnprob_cut is set, use it to remove low-prob islands"""

        # mask walkers not converged
        if self.explore == False:
            mask, _ = purge_chains(self.lnprob[extra_nburn:, :])
        else:
            mask = np.ones(self.lnprob.shape[1], dtype=bool)
        lnprob = self.lnprob[extra_nburn:, mask].reshape(-1)
        chain = self.chain[extra_nburn:, mask, :].reshape(
            -1, self.chain.shape[-1]
        )
        blobs = self.blobs[extra_nburn:, mask].reshape(-1)

        if delta_lnprob_cut:
            max_lnprob = np.max(lnprob)
            cut_lnprob = max_lnprob - delta_lnprob_cut
            mask = lnprob > cut_lnprob
            # total number and masked points in chain
            nt = len(lnprob)
            nm = sum(mask)
            self.print("will keep {} \ {} points from chain".format(nm, nt))
            chain = chain[mask]
            lnprob = lnprob[mask]
            blobs = blobs[mask]

        if cube == False:
            cube_values = chain
            list_values = [
                self.like.free_params[ip].value_from_cube(cube_values[:, ip])
                for ip in range(self.ndim)
            ]
            chain = np.array(list_values).transpose()

        return chain, lnprob, blobs

    def plot_autocorrelation_time(self):
        """Plot autocorrelation time as a function of sample number"""

        plt.figure()

        n = 100 * np.arange(1, len(self.autocorr) + 1)
        plt.plot(n, n / 100.0, "--k")
        plt.plot(n, self.autocorr)
        plt.xlim(0, n.max())
        plt.ylim(
            0,
            self.autocorr.max()
            + 0.1 * (self.autocorr.max() - self.autocorr.min()),
        )
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")
        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/autocorr_time.pdf")
            plt.close()
        else:
            plt.show()

        return

    def get_all_params(self, delta_lnprob_cut=None, extra_nburn=0):
        """Get a merged array of both sampled and derived parameters
        returns a 2D array of all parameters, and an ordered list of
        the LaTeX strings for each.
            - if delta_lnprob_cut is set, keep only high-prob points"""

        chain, lnprob, blobs = self.get_chain(
            cube=False,
            delta_lnprob_cut=delta_lnprob_cut,
            extra_nburn=extra_nburn,
        )

        if blobs is None:
            ## Old chains will have no blobs
            all_params = chain
            all_strings = self.paramstrings
        elif len(blobs[0]) == 6:
            # Build an array of chain + blobs, as chainconsumer doesn't know
            # about the difference between sampled and derived parameters
            all_params = np.zeros((chain.shape[0], chain.shape[1] + 6))
            all_params[:, : chain.shape[1]] = chain
            for ii in range(6):
                all_params[:, chain.shape[1] + ii] = blobs[
                    blob_strings_orig[ii]
                ]

            # Ordered strings for all parameters
            all_strings = self.paramstrings + blob_strings
        else:
            self.print(
                "Unknown blob configuration, just returning sampled params"
            )
            all_params = chain
            all_strings = self.paramstrings

        return all_params, all_strings, lnprob

    # def read_chain_from_file(self, chain_number, rootdir, subfolder):
    #     """Read chain from file, check parameters and setup likelihood"""

    #     if rootdir:
    #         chain_location = rootdir
    #     else:
    #         assert "CUP1D_PATH" in os.environ, "export CUP1D_PATH"
    #         chain_location = os.environ["CUP1D_PATH"] + "/chains/"
    #     if subfolder:
    #         self.save_directory = (
    #             chain_location + "/" + subfolder + "/chain_" + str(chain_number)
    #         )
    #     else:
    #         self.save_directory = chain_location + "/chain_" + str(chain_number)

    #     with open(self.save_directory + "/config.json") as json_file:
    #         config = json.load(json_file)

    #     self.print("Setup emulator")

    #     # new runs specify emulator_label, old ones use Pedersen23
    #     if "emulator_label" in config:
    #         emulator_label = config["emulator_label"]
    #     else:
    #         emulator_label = "Pedersen23"

    #     # setup emulator based on emulator_label
    #     if emulator_label == "Pedersen23":
    #         # check consistency in old book-keepings
    #         if "emu_type" in config:
    #             if config["emu_type"] != "polyfit":
    #                 raise ValueError("emu_type not polyfit", config["emu_type"])
    #         # emulator_label='Pedersen23' would ignore kmax_Mpc
    #         self.print("setup GP emulator used in Pedersen et al. (2023)")
    #         emulator = gp_emulator.GPEmulator(
    #             training_set="Pedersen21", kmax_Mpc=config["kmax_Mpc"]
    #         )
    #     elif emulator_label == "Cabayol23":
    #         self.print(
    #             "setup NN emulator used in Cabayol-Garcia et al. (2023)"
    #         )
    #         emulator = nn_emulator.NNEmulator(
    #             training_set="Cabayol23", emulator_label="Cabayol23"
    #         )
    #     elif emulator_label == "Nyx":
    #         self.print("setup NN emulator using Nyx simulations")
    #         emulator = nn_emulator.NNEmulator(
    #             training_set="Nyx23", emulator_label="Cabayol23_Nyx"
    #         )
    #     else:
    #         raise ValueError("wrong emulator_label", emulator_label)

    #     # Figure out redshift range in data
    #     if "z_list" in config:
    #         z_list = config["z_list"]
    #         zmin = min(z_list)
    #         zmax = max(z_list)
    #     else:
    #         zmin = config["data_zmin"]
    #         zmax = config["data_zmax"]

    #     # Setup mock data
    #     if "data_type" in config:
    #         data_type = config["data_type"]
    #     else:
    #         data_type = "gadget"
    #     self.print("Setup data of type =", data_type)
    #     if data_type == "mock":
    #         # using a mock_data P1D (computed from theory)
    #         data = mock_data.Mock_P1D(
    #             emulator=emulator,
    #             data_label=config["data_mock_label"],
    #             zmin=zmin,
    #             zmax=zmax,
    #         )
    #         # (optionally) setup extra P1D from high-resolution
    #         if "extra_p1d_label" in config:
    #             extra_data = mock_data.Mock_P1D(
    #                 emulator=emulator,
    #                 data_label=config["extra_p1d_label"],
    #                 zmin=config["extra_p1d_zmin"],
    #                 zmax=config["extra_p1d_zmax"],
    #             )
    #         else:
    #             extra_data = None
    #     elif data_type == "gadget":
    #         # using a data_gadget P1D (from Gadget sim)
    #         if "data_sim_number" in config:
    #             sim_label = config["data_sim_number"]
    #         else:
    #             sim_label = config["data_sim_label"]
    #         if not sim_label[:3] == "mpg":
    #             sim_label = "mpg_" + sim_label
    #         # check that sim is not from emulator suite
    #         assert sim_label not in range(30)
    #         # figure out p1d covariance used
    #         if "data_year" in config:
    #             data_cov_label = config["data_year"]
    #         else:
    #             data_cov_label = config["data_cov_label"]
    #         # we can get the archive from the emulator (should be consistent)
    #         data = data_gadget.Gadget_P1D(
    #             archive=emulator.archive,
    #             input_sim=sim_label,
    #             z_max=zmax,
    #             data_cov_factor=config["data_cov_factor"],
    #             data_cov_label=data_cov_label,
    #             polyfit_kmax_Mpc=emulator.kmax_Mpc,
    #             polyfit_ndeg=emulator.ndeg,
    #         )
    #         # (optionally) setup extra P1D from high-resolution
    #         if "extra_p1d_label" in config:
    #             extra_data = data_gadget.Gadget_P1D(
    #                 archive=emulator.archive,
    #                 input_sim=sim_label,
    #                 z_max=config["extra_p1d_zmax"],
    #                 data_cov_label=config["extra_p1d_label"],
    #                 polyfit_kmax_Mpc=emulator.kmax_Mpc,
    #                 polyfit_ndeg=emulator.ndeg,
    #             )
    #         else:
    #             extra_data = None
    #     elif data_type == "nyx":
    #         # using a data_nyx P1D (from Nyx sim)
    #         sim_label = config["data_sim_label"]
    #         # check that sim is not from emulator suite
    #         assert sim_label not in range(15)
    #         # figure out p1d covariance used
    #         if "data_year" in config:
    #             data_cov_label = config["data_year"]
    #         else:
    #             data_cov_label = config["data_cov_label"]
    #         data = data_nyx.Nyx_P1D(
    #             archive=emulator.archive,
    #             input_sim=sim_label,
    #             z_max=zmax,
    #             data_cov_factor=config["data_cov_factor"],
    #             data_cov_label=data_cov_label,
    #             polyfit_kmax_Mpc=emulator.kmax_Mpc,
    #             polyfit_ndeg=emulator.ndeg,
    #         )
    #         # (optionally) setup extra P1D from high-resolution
    #         if "extra_p1d_label" in config:
    #             extra_data = data_nyx.Nyx_P1D(
    #                 archive=emulator.archive,
    #                 input_sim=sim_label,
    #                 z_max=config["extra_p1d_zmax"],
    #                 data_cov_label=config["extra_p1d_label"],
    #                 polyfit_kmax_Mpc=emulator.kmax_Mpc,
    #                 polyfit_ndeg=emulator.ndeg,
    #             )
    #         else:
    #             extra_data = None
    #     elif data_type == "Chabanier2019":
    #         data = data_Chabanier2019.P1D_Chabanier2019(zmin=zmin, zmax=zmax)
    #         # (optionally) setup extra P1D from high-resolution
    #         if "extra_p1d_label" in config:
    #             if config["extra_p1d_label"] == "Karacayli2022":
    #                 extra_data = data_Karacayli2022.P1D_Karacayli2022(
    #                     diag_cov=True, kmax_kms=0.09, zmin=zmin, zmax=zmax
    #                 )
    #             else:
    #                 raise ValueError("unknown extra_p1d_label", extra_p1d_label)
    #         else:
    #             extra_data = None
    #     else:
    #         raise ValueError("unknown data type")

    #     # Setup free parameters
    #     self.print("Setting up likelihood")
    #     free_param_names = []
    #     for item in config["free_params"]:
    #         free_param_names.append(item[0])
    #     free_param_limits = config["free_param_limits"]

    #     # Setup fiducial cosmo and likelihood
    #     cosmo_fid_label = config["cosmo_fid_label"]
    #     self.like = likelihood.Likelihood(
    #         data=data,
    #         emulator=emulator,
    #         free_param_names=free_param_names,
    #         free_param_limits=free_param_limits,
    #         prior_Gauss_rms=config["prior_Gauss_rms"],
    #         emu_cov_factor=config["emu_cov_factor"],
    #         cosmo_fid_label=cosmo_fid_label,
    #         extra_p1d_data=extra_data,
    #     )

    #     # Verify we have a backend, and load it
    #     assert os.path.isfile(
    #         self.save_directory + "/backend.h5"
    #     ), "Backend not found, can't load chains"
    #     self.backend = emcee.backends.HDFBackend(
    #         self.save_directory + "/backend.h5"
    #     )

    #     ## Load chains - build a sampler object to access the backend
    #     sampler = emcee.EnsembleSampler(
    #         self.backend.shape[0],
    #         self.backend.shape[1],
    #         self.like.log_prob_and_blobs,
    #         backend=self.backend,
    #     )

    #     self.burnin_nsteps = config["burn_in"]
    #     self.chain = sampler.get_chain(flat=False, discard=self.burnin_nsteps)
    #     self.lnprob = sampler.get_log_prob(
    #         flat=False, discard=self.burnin_nsteps
    #     )
    #     self.blobs = sampler.get_blobs(flat=False, discard=self.burnin_nsteps)

    #     self.ndim = len(self.like.free_params)
    #     self.nwalkers = config["nwalkers"]
    #     self.autocorr = np.asarray(config["autocorr"])

    #     return

    def _setup_chain_folder(self, rootdir=None, subfolder=None):
        """Set up a directory to save files for this sampler run"""

        if rootdir:
            chain_location = rootdir
        else:
            repo = os.path.dirname(cup1d.__path__[0]) + "/"
            chain_location = repo + "data/chains/"
        if subfolder:
            # If there is one, check if it exists, if not make it
            if not os.path.isdir(chain_location + "/" + subfolder):
                os.mkdir(chain_location + "/" + subfolder)
            base_string = chain_location + "/" + subfolder + "/chain_"
        else:
            base_string = chain_location + "/chain_"

        # Create a new folder for this chain
        chain_count = 1
        while True:
            sampler_directory = base_string + str(chain_count)
            if os.path.isdir(sampler_directory):
                chain_count += 1
                continue
            else:
                try:
                    os.mkdir(sampler_directory)
                    self.print("Created directory:", sampler_directory)
                    break
                except FileExistsError:
                    self.print("Race condition for:", sampler_directory)
                    # try again after one mili-second
                    time.sleep(0.001)
                    chain_count += 1
                    continue
        self.save_directory = sampler_directory

        return

    def _write_dict_to_text(self, saveDict):
        """Write the settings for this chain
        to a more easily readable .txt file"""

        ## What keys don't we want to include in the info file
        dontPrint = ["lnprob", "flatchain", "blobs", "autocorr"]

        with open(self.save_directory + "/info.txt", "w") as f:
            for item in saveDict.keys():
                if item not in dontPrint:
                    f.write("%s: %s\n" % (item, str(saveDict[item])))

        return

    def get_best_fit(self, delta_lnprob_cut=None, stat_best_fit="mean"):
        """Return an array of best fit values (mean) from the MCMC chain,
        in unit likelihood space.
            - if delta_lnprob_cut is set, use only high-prob points"""

        if stat_best_fit == "mean":
            chain, lnprob, blobs = self.get_chain(
                delta_lnprob_cut=delta_lnprob_cut
            )
            best_values = np.mean(chain, axis=0)
        elif stat_best_fit == "median":
            chain, lnprob, blobs = self.get_chain(
                delta_lnprob_cut=delta_lnprob_cut
            )
            best_values = np.median(chain, axis=0)
        elif stat_best_fit == "mle":
            best_values = self.mle_cube
        else:
            raise ValueError(stat_best_fit + " not implemented")

        return best_values

    def write_chain_to_file(self, residuals=True, extra_nburn=0):
        """Write flat chain to file"""

        # TO BE UPDATED

        saveDict = {}

        # # Emulator settings
        # emulator = self.like.theory.emulator
        # saveDict["kmax_Mpc"] = emulator.kmax_Mpc
        # # if isinstance(emulator, gp_emulator.GPEmulator):
        # #     saveDict["emu_type"] = emulator.emu_type
        # # else:
        # #     # this is dangerous, there might be different settings
        # #     if isinstance(emulator.archive, gadget_archive.GadgetArchive):
        # #         saveDict["emulator_label"] = "Cabayol23"
        # #     else:
        # #         saveDict["emulator_label"] = "Nyx"

        # # Data settings
        # if isinstance(self.like.data, data_gadget.Gadget_P1D):
        #     # using a data_gadget P1D (from Gadget sim)
        #     saveDict["data_type"] = "gadget"
        #     saveDict["data_sim_label"] = self.like.data.input_sim
        #     saveDict["data_cov_label"] = self.like.data.data_cov_label
        #     saveDict["data_cov_factor"] = self.like.data.data_cov_factor
        # elif isinstance(self.like.data, data_nyx.Nyx_P1D):
        #     # using a data_nyx P1D (from Nyx sim)
        #     saveDict["data_type"] = "nyx"
        #     saveDict["data_sim_label"] = self.like.data.input_sim
        #     saveDict["data_cov_label"] = self.like.data.data_cov_label
        #     saveDict["data_cov_factor"] = self.like.data.data_cov_factor
        # elif hasattr(self.like.data, "theory"):
        #     # using a mock_data P1D (computed from theory)
        #     saveDict["data_type"] = "mock"
        #     saveDict["data_mock_label"] = self.like.data.data_label
        # elif isinstance(self.like.data, data_Chabanier2019.P1D_Chabanier2019):
        #     saveDict["data_type"] = "Chabanier2019"
        # else:
        #     saveDict["data_type"] = "other"
        # saveDict["data_zmin"] = min(self.like.theory.zs)
        # saveDict["data_zmax"] = max(self.like.theory.zs)

        # # Other likelihood settings
        # saveDict["prior_Gauss_rms"] = self.like.prior_Gauss_rms
        # saveDict["cosmo_fid_label"] = self.like.cosmo_fid_label
        # saveDict["emu_cov_factor"] = self.like.emu_cov_factor
        # free_params_save = []
        # free_param_limits = []
        # for par in self.like.free_params:
        #     free_params_save.append([par.name, par.min_value, par.max_value])
        #     free_param_limits.append([par.min_value, par.max_value])
        # saveDict["free_params"] = free_params_save
        # saveDict["free_param_limits"] = free_param_limits

        # # Sampler stuff
        # saveDict["burn_in"] = self.burnin_nsteps
        # saveDict["nwalkers"] = self.nwalkers
        # if self.get_autocorr:
        #     saveDict["autocorr"] = self.autocorr.tolist()

        # # Save dictionary to json file in the appropriate directory
        # if self.save_directory is None:
        #     self._setup_chain_folder()
        # with open(self.save_directory + "/config.json", "w") as json_file:
        #     json.dump(saveDict, json_file)

        # # save config info in plain text as well
        # self._write_dict_to_text(saveDict)

        tries = True

        dict_out = {}
        if tries:
            # plots
            mask_use = self.plot_lnprob(extra_nburn=extra_nburn)
            plt.close()
            self.plot_p1d(residuals=residuals)
            plt.close()
            self.plot_igm()
            plt.close()
            if self.fix_cosmology == False:
                self.plot_corner(only_cosmo=True, extra_nburn=extra_nburn)
                plt.close()
            dict_out["summary"] = self.plot_corner(extra_nburn=extra_nburn)
            plt.close()

            # for stat_best_fit in ["mle"]:
            #     self.plot_p1d(
            #         residuals=residuals,
            #         rand_posterior=rand_posterior,
            #         stat_best_fit=stat_best_fit,
            #     )

            # try:
            #     self.plot_prediction(residuals=residuals)
            # except:
            #     self.print("Can't plot prediction")

            # if self.get_autocorr:
            #     try:
            #         self.plot_autocorrelation_time()
            #     except:
            #         self.print("Can't plot autocorrelation time")

        else:
            mask_use = self.plot_lnprob()
            self.plot_p1d(residuals=residuals, stat_best_fit="mean")
            for stat_best_fit in ["mean"]:
                rand_posterior = self.plot_igm(stat_best_fit=stat_best_fit)
                self.plot_p1d(
                    residuals=residuals,
                    rand_posterior=rand_posterior,
                    stat_best_fit=stat_best_fit,
                )
                self.plot_prediction(residuals=residuals)
            if self.fix_cosmology == False:
                _ = self.plot_corner(only_cosmo=True)
            dict_out["summary"] = self.plot_corner()

        dict_out["walkers_survive"] = mask_use
        dict_out["truth"] = self.truth

        all_param, all_names, lnprob = self.get_all_params(
            extra_nburn=extra_nburn
        )
        dict_out["param_names"] = all_names
        dict_out["param_percen"] = np.percentile(
            all_param, [16, 50, 84], axis=0
        ).T

        dict_out["param_mle"] = self.mle
        dict_out["lnprob_mle"] = self.lnprop_mle

        np.save(self.save_directory + "/results.npy", dict_out)
        np.save(self.save_directory + "/chain.npy", all_param)
        np.save(self.save_directory + "/lnprob.npy", lnprob)

    def plot_histograms(self, cube=False, delta_lnprob_cut=None):
        """Make histograms for all dimensions, using re-normalized values if
        cube=True
        - if delta_lnprob_cut is set, use only high-prob points"""

        # get chain (from sampler or from file)
        chain, lnprob, blobs = self.get_chain(delta_lnprob_cut=delta_lnprob_cut)
        plt.figure()

        for ip in range(self.ndim):
            param = self.like.free_params[ip]
            if cube:
                values = chain[:, ip]
                title = param.name + " in cube"
            else:
                cube_values = chain[:, ip]
                values = param.value_from_cube(cube_values)
                title = param.name

            plt.hist(values, 100, color="k", histtype="step")
            plt.title(title)
            plt.show()

        return

    def plot_mle_cosmo(self, nyx_version="Jul2024"):
        """Plot MLE cosmology"""

        suite_emu = self.like.theory.emulator.list_sim_cube[0][:3]
        if suite_emu == "mpg":
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
        elif suite_emu == "nyx":
            fname = (
                os.environ["NYX_PATH"] + "nyx_emu_cosmo_" + nyx_version + ".npy"
            )
        else:
            ValueError("cosmo_label should be 'mpg' or 'nyx'")

        try:
            data_cosmo = np.load(fname, allow_pickle=True)
        except:
            ValueError(f"{fname} not found")

        labs = []
        delta2_star = np.zeros(len(data_cosmo))
        n_star = np.zeros(len(data_cosmo))
        alpha_star = np.zeros(len(data_cosmo))

        for ii in range(len(data_cosmo)):
            labs.append(data_cosmo[ii]["sim_label"])
            delta2_star[ii] = data_cosmo[ii]["star_params"]["Delta2_star"]
            n_star[ii] = data_cosmo[ii]["star_params"]["n_star"]
            alpha_star[ii] = data_cosmo[ii]["star_params"]["alpha_star"]

        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
        ax[0].scatter(delta2_star, n_star)
        ax[1].scatter(n_star, alpha_star)

        dif0 = delta2_star.max() - delta2_star.min()
        dif1 = n_star.max() - n_star.min()
        dif2 = alpha_star.max() - alpha_star.min()
        sep_x = 0.05
        for ii, lab in enumerate(labs):
            if data_cosmo[ii]["sim_label"][-1].isdigit():
                sep_y = 0
            else:
                sep_y = 0.05
            ax[0].annotate(
                lab[4:],
                (delta2_star[ii] + sep_x * dif0, n_star[ii] + sep_y * dif1),
                fontsize=8,
            )
            ax[1].annotate(
                lab[4:],
                (n_star[ii] + sep_x * dif1, alpha_star[ii] + sep_y * dif2),
                fontsize=8,
            )

        ax[0].errorbar(
            self.mle_cosmo["Delta2_star"],
            self.mle_cosmo["n_star"],
            xerr=self.mle_cosmo["err_Delta2_star"],
            yerr=self.mle_cosmo["err_n_star"],
            marker="X",
            color="C1",
        )
        ax[1].errorbar(
            self.mle_cosmo["n_star"],
            self.mle_cosmo["alpha_star"],
            xerr=self.mle_cosmo["err_n_star"],
            yerr=self.mle_cosmo["err_alpha_star"],
            marker="X",
            color="C1",
        )

        fontsize = 16
        ax[0].set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
        ax[0].set_ylabel(r"$n_\star$", fontsize=fontsize)
        ax[1].set_xlabel(r"$n_\star$", fontsize=fontsize)
        ax[1].set_ylabel(r"$\alpha_\star$", fontsize=fontsize)
        plt.tight_layout()

        return

    def plot_corner(
        self,
        plot_params=None,
        delta_lnprob_cut=None,
        usetex=True,
        serif=True,
        only_cosmo=False,
        save_directory=None,
        extra_nburn=0,
    ):
        """Make corner plot in ChainConsumer
        - plot_params: Pass a list of parameters to plot (in LaTeX form),
                    or leave as None to
                    plot all (including derived)
        - if delta_lnprob_cut is set, keep only high-prob points"""

        params_plot, strings_plot, _ = self.get_all_params(
            delta_lnprob_cut=delta_lnprob_cut, extra_nburn=extra_nburn
        )
        if only_cosmo:
            yesplot = ["$\\Delta^2_\\star$", "$n_\\star$"]
            for param in self.like.free_params:
                if param.name == "nrun":
                    yesplot.append("$\\alpha_\\star$")
        else:
            diff = np.max(params_plot, axis=0) - np.min(params_plot, axis=0)
            yesplot = np.array(strings_plot)[diff != 0]

        dict_pd = {}
        for ii, par in enumerate(strings_plot):
            if par in yesplot:
                dict_pd[par] = params_plot[:, ii]
        pd_data = pd.DataFrame(data=dict_pd)

        c = ChainConsumer()
        chain = Chain(
            samples=pd_data,
            name="a",
        )
        c.add_chain(chain)
        summary = c.analysis.get_summary()["a"]
        if self.truth is not None:
            c.add_truth(
                Truth(
                    location=self.truth,
                    line_style="--",
                    color="C1",
                )
            )
        c.add_truth(
            Truth(
                location=self.mle,
                line_style=":",
                color="C2",
            )
        )

        fig = c.plotter.plot(figsize=(12, 12))

        if save_directory is not None:
            save_directory = save_directory
        elif self.save_directory is not None:
            save_directory = self.save_directory

        if save_directory is not None:
            if only_cosmo:
                plt.savefig(save_directory + "/corner_cosmo.pdf")
            else:
                plt.savefig(save_directory + "/corner.pdf")

        return summary

    def plot_lnprob(self, extra_nburn=0, save_directory=None):
        """Plot lnprob"""

        mask, _ = purge_chains(self.lnprob[extra_nburn:, :])
        mask_use = (
            "Using "
            + str(mask.shape[0])
            + " chains out of "
            + str(self.lnprob.shape[1])
        )

        for ii in range(self.lnprob.shape[1]):
            if ii in mask:
                plt.plot(self.lnprob[extra_nburn:, ii], alpha=0.5)
            # else:
            #     plt.plot(self.lnprob[extra_nburn:, ii], "--", alpha=0.5)

        # save to file
        if save_directory is not None:
            save_directory = save_directory
        elif self.save_directory is not None:
            save_directory = self.save_directory

        if save_directory is not None:
            plt.savefig(self.save_directory + "/lnprob.pdf")
        # plt.close()

        return mask_use

    def plot_p1d(
        self,
        values=None,
        plot_every_iz=1,
        residuals=False,
        rand_posterior=None,
        stat_best_fit="mle",
        save_directory=None,
    ):
        """Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        if values is None:
            values = self.get_best_fit(stat_best_fit=stat_best_fit)

        # save to file
        if save_directory is not None:
            save_directory = save_directory
        elif self.save_directory is not None:
            save_directory = self.save_directory

        if save_directory is not None:
            if rand_posterior is None:
                fname = "best_fit_" + stat_best_fit + "_err_emu"
            else:
                fname = "best_fit_" + stat_best_fit + "_err_posterior"
            if residuals:
                plot_fname = save_directory + "/" + fname + "_residuals.pdf"
            else:
                plot_fname = save_directory + "/" + fname + ".pdf"
        else:
            plot_fname = None

        self.like.plot_p1d(
            values=values,
            plot_every_iz=plot_every_iz,
            residuals=residuals,
            rand_posterior=rand_posterior,
            plot_fname=plot_fname,
        )

    def plot_prediction(
        self, figsize=(8, 6), values=None, plot_every_iz=1, residuals=False
    ):
        """Plot the P1D of the data and the emulator prediction
        for the fiducial model"""

        plt.figure(figsize=figsize)
        if values == None:
            plt.title("Fiducial model")
        else:
            plt.title("P1D at %s" % values)
        self.like.plot_p1d(
            values=values, plot_every_iz=plot_every_iz, residuals=residuals
        )

        if self.save_directory is not None:
            plt.savefig(self.save_directory + "/fiducial.pdf")
            plt.close()
        else:
            plt.show()

        return

    def plot_igm(
        self,
        value=None,
        rand_sample=None,
        stat_best_fit="mle",
        cloud=False,
        save_directory=None,
    ):
        """Plot IGM histories"""

        # true IGM parameters
        if self.like.truth is not None:
            pars_true = {}
            pars_true["z"] = self.like.truth["igm"]["z"]
            pars_true["tau_eff"] = self.like.truth["igm"]["tau_eff"]
            pars_true["gamma"] = self.like.truth["igm"]["gamma"]
            pars_true["sigT_kms"] = self.like.truth["igm"]["sigT_kms"]
            pars_true["kF_kms"] = self.like.truth["igm"]["kF_kms"]

        # fiducial IGM parameters
        pars_fid = {}
        pars_fid["z"] = self.like.fid["igm"]["z"]
        pars_fid["tau_eff"] = self.like.fid["igm"]["tau_eff"]
        pars_fid["gamma"] = self.like.fid["igm"]["gamma"]
        pars_fid["sigT_kms"] = self.like.fid["igm"]["sigT_kms"]
        pars_fid["kF_kms"] = self.like.fid["igm"]["kF_kms"]

        # all IGM histories in the training sample
        if cloud:
            emu_label_igm = self.like.theory.emulator.training_data[0][
                "sim_label"
            ]
            all_emu_igm = self.like.theory.model_igm.get_igm(
                emu_label_igm, return_all=True
            )

        # best-fitting IGM parameters
        if value is None:
            value = self.get_best_fit(stat_best_fit=stat_best_fit)
        like_params = self.like.parameters_from_sampling_point(value)

        pars_best = {}
        pars_best["z"] = np.array(self.like.data.z)
        pars_best["tau_eff"] = self.like.theory.model_igm.F_model.get_tau_eff(
            pars_best["z"], like_params=like_params
        )
        pars_best["gamma"] = self.like.theory.model_igm.T_model.get_gamma(
            pars_best["z"], like_params=like_params
        )
        pars_best["sigT_kms"] = self.like.theory.model_igm.T_model.get_sigT_kms(
            pars_best["z"], like_params=like_params
        )
        pars_best["kF_kms"] = self.like.theory.model_igm.P_model.get_kF_kms(
            pars_best["z"], like_params=like_params
        )

        if rand_sample is not None:
            # chain, lnprob, blobs = self.get_chain()
            # nn = min(chain.shape[0], nn)
            # mask = np.random.permutation(chain.shape[0])[:nn]
            # rand_sample = chain[mask]

            # sample the chain to get errors on IGM parameters
            pars_samp = {}
            pars_samp["tau_eff"] = np.zeros((nn, len(z)))
            pars_samp["gamma"] = np.zeros((nn, len(z)))
            pars_samp["sigT_kms"] = np.zeros((nn, len(z)))
            pars_samp["kF_kms"] = np.zeros((nn, len(z)))
            for ii in range(nn):
                like_params = self.like.parameters_from_sampling_point(
                    rand_sample[ii]
                )
                models = self.like.theory.update_igm_models(like_params)
                pars_samp["tau_eff"][ii] = models["F_model"].get_tau_eff(z)
                pars_samp["gamma"][ii] = models["T_model"].get_gamma(z)
                pars_samp["sigT_kms"][ii] = models["T_model"].get_sigT_kms(z)
                pars_samp["kF_kms"][ii] = models["P_model"].get_kF_kms(z)

        # plot the IGM histories
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
            if self.like.truth is not None:
                _ = pars_true[arr_labs[ii]] != 0
                ax[ii].plot(
                    pars_true["z"][_],
                    pars_true[arr_labs[ii]][_],
                    ":o",
                    label="true",
                    alpha=0.75,
                )
            _ = pars_fid[arr_labs[ii]] != 0
            ax[ii].plot(
                pars_fid["z"][_],
                pars_fid[arr_labs[ii]][_],
                "--",
                label="fiducial",
                lw=3,
                alpha=0.75,
            )

            _ = pars_best[arr_labs[ii]] != 0
            if rand_sample is not None:
                err = np.abs(
                    np.percentile(pars_samp[arr_labs[ii]], [16, 84], axis=0)
                    - pars_best[arr_labs[ii]]
                )
                ax[ii].errorbar(
                    pars_best["z"],
                    pars_best[arr_labs[ii]],
                    err,
                    label="best-fitting",
                    alpha=0.75,
                )
            else:
                ax[ii].plot(
                    pars_best["z"][_],
                    pars_best[arr_labs[ii]][_],
                    label="fit",
                    lw=3,
                    alpha=0.75,
                )

            if cloud:
                for sim_label in all_emu_igm:
                    _ = all_emu_igm[sim_label][arr_labs[ii]] != 0
                    ax[ii].scatter(
                        all_emu_igm[sim_label]["z"][_],
                        all_emu_igm[sim_label][arr_labs[ii]][_],
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
            save_directory = save_directory
        elif self.save_directory is not None:
            save_directory = self.save_directory

        if save_directory is not None:
            plt.savefig(
                save_directory + "/IGM_histories_" + stat_best_fit + ".pdf"
            )
            # plt.close()
        else:
            plt.show()


## Dictionary to convert likelihood parameters into latex strings
param_dict = {
    "Delta2_p": "$\Delta^2_p$",
    "mF": "$F$",
    "gamma": "$\gamma$",
    "sigT_Mpc": "$\sigma_T$",
    "kF_Mpc": "$k_F$",
    "n_p": "$n_p$",
    "Delta2_star": "$\Delta^2_\star$",
    "n_star": "$n_\star$",
    "alpha_star": "$\\alpha_\star$",
    "g_star": "$g_\star$",
    "f_star": "$f_\star$",
    "ln_tau_0": "$\mathrm{ln}\,\\tau_0$",
    "ln_tau_1": "$\mathrm{ln}\,\\tau_1$",
    "ln_tau_2": "$\mathrm{ln}\,\\tau_2$",
    "ln_tau_3": "$\mathrm{ln}\,\\tau_3$",
    "ln_tau_4": "$\mathrm{ln}\,\\tau_4$",
    "ln_sigT_kms_0": "$\mathrm{ln}\,\sigma^T_0$",
    "ln_sigT_kms_1": "$\mathrm{ln}\,\sigma^T_1$",
    "ln_sigT_kms_2": "$\mathrm{ln}\,\sigma^T_2$",
    "ln_sigT_kms_3": "$\mathrm{ln}\,\sigma^T_3$",
    "ln_sigT_kms_4": "$\mathrm{ln}\,\sigma^T_4$",
    "ln_gamma_0": "$\mathrm{ln}\,\gamma_0$",
    "ln_gamma_1": "$\mathrm{ln}\,\gamma_1$",
    "ln_gamma_2": "$\mathrm{ln}\,\gamma_2$",
    "ln_gamma_3": "$\mathrm{ln}\,\gamma_3$",
    "ln_gamma_4": "$\mathrm{ln}\,\gamma_4$",
    # it might be better to specify kF_kms here as well
    "ln_kF_0": "$\mathrm{ln}\,k^F_0$",
    "ln_kF_1": "$\mathrm{ln}\,k^F_1$",
    "ln_kF_2": "$\mathrm{ln}\,k^F_2$",
    "ln_kF_3": "$\mathrm{ln}\,k^F_3$",
    "ln_kF_4": "$\mathrm{ln}\,k^F_4$",
    # each metal contamination should have its own parameters here
    "ln_SiIII_0": "$\mathrm{ln}\,f^{SiIII}_0$",
    "ln_SiIII_1": "$\mathrm{ln}\,f^{SiIII}_1$",
    "ln_SiII_0": "$\mathrm{ln}\,f^{SiII}_0$",
    "ln_SiII_1": "$\mathrm{ln}\,f^{SiII}_1$",
    # each HCD contamination should have its own parameters here
    "ln_A_damp_0": "$\mathrm{ln}\,\mathrm{HCD}_0$",
    "ln_A_damp_1": "$\mathrm{ln}\,\mathrm{HCD}_1$",
    "ln_SN_0": "$\mathrm{ln}\,\mathrm{SN}_0$",
    "ln_SN_1": "$\mathrm{ln}\,\mathrm{SN}_1$",
    "H0": "$H_0$",
    "mnu": "$\Sigma m_\\nu$",
    "As": "$A_s$",
    "ns": "$n_s$",
    "nrun": "$n_\mathrm{run}$",
    "ombh2": "$\omega_b$",
    "omch2": "$\omega_c$",
    "cosmomc_theta": "$\\theta_{MC}$",
}


## List of all possibly free cosmology params for the truth array
## for chainconsumer plots
cosmo_params = [
    "Delta2_star",
    "n_star",
    "alpha_star",
    "f_star",
    "g_star",
    "cosmomc_theta",
    "H0",
    "mnu",
    "As",
    "ns",
    "nrun",
    "ombh2",
    "omch2",
]

## list of strings for blobs
blob_strings = [
    "$\Delta^2_\star$",
    "$n_\star$",
    "$\\alpha_\star$",
    "$f_\star$",
    "$g_\star$",
    "$H_0$",
]
blob_strings_orig = [
    "Delta2_star",
    "n_star",
    "alpha_star",
    "f_star",
    "g_star",
    "H0",
]


def compare_corners(
    chain_files,
    labels,
    plot_params=None,
    save_string=None,
    rootdir=None,
    subfolder=None,
    delta_lnprob_cut=None,
    usetex=True,
    serif=True,
):
    """Function to take a list of chain files and overplot the chains
    Pass a list of chain files (ints) and a list of labels (strings)
     - plot_params: list of parameters (in code variables, not latex form)
                    to plot if only a subset is desired
     - save_string: to save the plot. Must include
                    file extension (i.e. .pdf, .png etc)
     - if delta_lnprob_cut is set, keep only high-prob points"""

    assert len(chain_files) == len(labels)

    truth_dict = {}
    c = ChainConsumer()

    ## Add each chain we want to plot
    for aa, chain_file in enumerate(chain_files):
        sampler = EmceeSampler(
            read_chain_file=chain_file, subfolder=subfolder, rootdir=rootdir
        )
        params, strings, _ = sampler.get_all_params(
            delta_lnprob_cut=delta_lnprob_cut
        )
        c.add_chain(params, parameters=strings, name=labels[aa])

        ## Do not check whether truth results are the same for now
        ## Take the longest truth dictionary for disjoint chains
        if len(sampler.truth) > len(truth_dict):
            truth_dict = sampler.truth

    c.configure(
        diagonal_tick_labels=False,
        tick_font_size=15,
        label_font_size=25,
        max_ticks=4,
        usetex=usetex,
        serif=serif,
    )

    if plot_params == None:
        fig = c.plotter.plot(figsize=(15, 15), truth=truth_dict)
    else:
        ## From plot_param list, build list of parameter
        ## strings to plot
        plot_param_strings = []
        for par in plot_params:
            plot_param_strings.append(param_dict[par])
        fig = c.plotter.plot(
            figsize=(10, 10), parameters=plot_param_strings, truth=truth_dict
        )
    if save_string:
        fig.savefig("%s" % save_string)
    fig.show()

    return

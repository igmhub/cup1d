import os, time, emcee, json

import scipy.stats
import numpy as np
from warnings import warn

from pyDOE2 import lhs
import copy
from mpi4py import MPI

# our own modules
import cup1d, lace
from lace.cosmo import camb_cosmo
from cup1d.utils.utils import create_print_function, purge_chains


class Fitter(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(
        self,
        like=None,
        nwalkers=None,
        nsteps=None,
        nburnin=None,
        verbose=False,
        subfolder=None,
        rootdir=None,
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
        self.burnin_nsteps = nburnin

        self.param_dict = param_dict
        self.param_dict_rev = param_dict_rev

        if self.parallel:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        self.fix_cosmology = fix_cosmology

        self.print = create_print_function(self.verbose)

        self.like = like
        # number of free parameters to sample
        self.ndim = len(self.like.free_params)

        if self.rank == 0:
            self._setup_chain_folder(rootdir, subfolder)

            # number of walkers
            if nwalkers is not None:
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

                for irank in range(1, self.size):
                    self.comm.send(nsteps, dest=irank, tag=irank * 11)
                    self.comm.send(nwalkers, dest=irank, tag=irank * 13)

        else:
            nsteps = self.comm.recv(source=0, tag=self.rank * 11)
            nwalkers = self.comm.recv(source=0, tag=self.rank * 13)

        self.nwalkers = nwalkers
        self.nsteps = nsteps

        print(
            "rank", self.rank, "nwalkers", self.nwalkers, "nsteps", self.nsteps
        )

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

        # set blinding
        self.set_blinding()

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
        zmask=None,
        timeout=None,
        force_timeout=False,
    ):
        """Set up sampler, run burn in, run chains,
        return chains
            - timeout is the time in hours to run the
              sampler for
            - force_timeout will continue to run the chains
              until timeout, regardless of convergence"""

        if log_func is None:
            _log_func = self.like.log_prob_and_blobs
        else:
            _log_func = log_func

        if zmask is not None:
            log_func = lambda x: _log_func(x, zmask=zmask)
        else:
            log_func = _log_func

        if self.parallel == False:
            ## Get initial walkers
            p0 = self.get_initial_walkers(pini=pini)

            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                log_func,
                blobs_dtype=self.blobs_dtype,
            )
            for sample in sampler.sample(
                p0, iterations=self.burnin_nsteps + self.nsteps
            ):
                if sampler.iteration % 100 == 0:
                    self.print(
                        "Step %d out of %d "
                        % (sampler.iteration, self.burnin_nsteps + self.nsteps)
                    )

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
                # print(chain[-1].shape)
                # print(chain[-1][0])

                for irank in range(1, self.size):
                    self.print("Receiving from rank %d" % irank)
                    lnprob.append(
                        self.comm.recv(source=irank, tag=1000 + irank)
                    )
                    chain.append(self.comm.recv(source=irank, tag=2000 + irank))
                    blobs.append(self.comm.recv(source=irank, tag=3000 + irank))
                    # print(irank, chain[-1].shape)
                    # print(chain[-1][0])

                self.lnprob = np.concatenate(lnprob, axis=1)
                self.chain = np.concatenate(chain, axis=1)
                self.blobs = np.concatenate(blobs, axis=1)

                map_ind = np.argmax(self.lnprob.reshape(-1))
                map_chi2 = -2.0 * self.lnprob.reshape(-1)[map_ind]
                map_chain = self.chain.reshape(-1, self.chain.shape[-1])[
                    map_ind
                ]
                self.set_mle(map_chain, map_chi2)

                # apply masking (only to star parameters)
                self.blobs = self.apply_blinding(self.blobs, sample="chains")

        return sampler

    def run_minimizer(
        self,
        log_func_minimize=None,
        p0=None,
        nsamples=8,
        zmask=None,
        mask_pars=None,
        restart=False,
    ):
        """Minimizer"""

        def set_log_func_minimize(pini, zmask=None, mask_pars=None):
            if mask_pars is None:
                if zmask is not None:
                    fun = lambda x: log_func_minimize(x, zmask=zmask)
                    return fun
                else:
                    return log_func_minimize
            else:
                ind_fix = []
                for ii, par in enumerate(self.like.free_params):
                    if par.fixed:
                        ind_fix.append(ii)
                ind_fix = np.array(ind_fix)
                pfix = pini[ind_fix]
                if zmask is not None:
                    fun = lambda x: log_func_minimize(
                        x, zmask=zmask, ind_fix=ind_fix, pfix=pfix
                    )
                    return fun
                else:
                    fun = lambda x: log_func_minimize(
                        x, ind_fix=ind_fix, pfix=pfix
                    )
                    return fun

        _log_func_minimize = set_log_func_minimize(
            p0, zmask=zmask, mask_pars=mask_pars
        )

        if restart:
            self.mle_chi2 = 1e10

        npars = len(self.like.free_params)

        if p0 is not None:
            # start at the initial value
            mle_cube = p0.copy()
        else:
            # star at the center of the parameter space
            mle_cube = np.ones(npars) * 0.5

        chi2 = self.like.get_chi2(mle_cube, zmask=zmask)
        chi2_ini = chi2 * 1

        # perturbations around starting point
        arr_p0 = lhs(npars, samples=nsamples + 1) - 0.5
        # sigma to search around mle_cube
        sig = 0.1

        print("Starting NM minimization", flush=True)
        rep = 0
        start = time.time()
        for ii in range(nsamples + 1):
            pini = mle_cube.copy()
            if ii != 0:
                pini += arr_p0[ii] * sig

            pini[pini <= 0] = 0.05
            pini[pini >= 1] = 0.95

            res = scipy.optimize.minimize(
                _log_func_minimize,
                pini,
                method="Nelder-Mead",
                bounds=((0.0, 1.0),) * npars,
            )
            _chi2 = self.like.get_chi2(res.x, zmask=zmask)
            # print(ii, res.x)

            if _chi2 < chi2:
                chi2 = _chi2.copy()
                mle_cube = res.x.copy()
                # reduce sigma
                sig *= 0.9
                rep = 0
            else:
                rep += 1

            print("Step took:", np.round(time.time() - start, 2), flush=True)
            print(
                "Minimization improved:",
                np.round(chi2_ini, 4),
                np.round(chi2, 4),
                np.round(chi2_ini - chi2, 4),
                flush=True,
            )
            # if chi2 does not get better after a few it, stop
            if rep > 3:
                break

        self.set_mle(mle_cube, chi2)

    def run_minimizer_da(
        self,
        log_func_minimize=None,
        p0=None,
        zmask=None,
        mask_pars=None,
        restart=True,
    ):
        """Minimizer using dual annealing"""

        def set_log_func_minimize(pini, zmask=None, mask_pars=None):
            if mask_pars is None:
                if zmask is not None:
                    fun = lambda x: log_func_minimize(x, zmask=zmask)
                    return fun
                else:
                    return log_func_minimize
            else:
                ind_fix = []
                for ii, par in enumerate(self.like.free_params):
                    if par.fixed:
                        ind_fix.append(ii)
                ind_fix = np.array(ind_fix)
                pfix = pini[ind_fix]
                if zmask is not None:
                    fun = lambda x: log_func_minimize(
                        x, zmask=zmask, ind_fix=ind_fix, pfix=pfix
                    )
                    return fun
                else:
                    fun = lambda x: log_func_minimize(
                        x, ind_fix=ind_fix, pfix=pfix
                    )
                    return fun

        if restart:
            self.mle_chi2 = 1e10

        _log_func_minimize = set_log_func_minimize(
            p0, zmask=zmask, mask_pars=mask_pars
        )

        npars = len(self.like.free_params)

        if p0 is not None:
            # start at the initial value
            mle_cube = p0.copy()
        else:
            # star at the center of the parameter space
            mle_cube = np.ones(npars) * 0.5

        mle_cube[mle_cube <= 0] = 0.05
        mle_cube[mle_cube >= 1] = 0.95

        chi2 = self.like.get_chi2(mle_cube, zmask=zmask)
        chi2_ini = chi2 * 1

        print("Starting DA minimization", flush=True)

        start = time.time()
        res = scipy.optimize.dual_annealing(
            _log_func_minimize,
            x0=mle_cube,
            bounds=((0.0, 1.0),) * npars,
        )

        _chi2 = self.like.get_chi2(res.x, zmask=zmask)

        if _chi2 < chi2:
            chi2 = _chi2.copy()
            mle_cube = res.x.copy()

        print("Step took:", np.round(time.time() - start, 2), flush=True)
        print(
            "Minimization improved:",
            np.round(chi2_ini, 4),
            np.round(chi2, 4),
            np.round(chi2_ini - chi2, 4),
            flush=True,
        )

        self.set_mle(mle_cube, chi2)

    def run_profile(
        self,
        irank,
        mle_cosmo_cen,
        shift_cosmo,
        input_pars,
        nsamples=1,
        type_minimizer="NM",
    ):
        """Profile likelihood"""

        blind_cosmo = {
            "Delta2_star": mle_cosmo_cen["Delta2_star"]
            + shift_cosmo["Delta2_star"],
            "n_star": mle_cosmo_cen["n_star"] + shift_cosmo["n_star"],
        }
        # unblind internally to apply shift consistently
        target = self.apply_unblinding(mle_cosmo_cen)
        target["Delta2_star"] += shift_cosmo["Delta2_star"]
        target["n_star"] += shift_cosmo["n_star"]

        self.like.theory.rescale_fid_cosmo(target)

        # check whether new fiducial cosmology is within priors
        if np.isfinite(self.like.get_chi2(input_pars)) == False:
            print("skipping", irank, blind_cosmo)
            return

        if type_minimizer == "NM":
            self.run_minimizer(
                self.like.minus_log_prob,
                p0=input_pars,
                restart=True,
                nsamples=nsamples,
            )
        elif type_minimizer == "DA":
            self.run_minimizer_da(
                self.like.minus_log_prob, p0=input_pars, restart=True
            )
        else:
            raise ValueError("type_minimizer must be 'NM' or 'DA'")

        out_dict = {
            "chi2": self.mle_chi2,
            "blind_cosmo": blind_cosmo,
            "mle_cube": self.mle_cube,
            "mle": self.mle,
        }

        file_out = os.path.join(
            os.path.dirname(self.save_directory),
            "profile_" + str(irank) + ".npy",
        )
        np.save(file_out, out_dict)

    def set_mle(self, mle_cube, mle_chi2):
        """Set the maximum likelihood solution"""

        if hasattr(self, "mle_chi2"):
            if mle_chi2 < self.mle_chi2:
                print("updating mle from ", self.mle_chi2, "to", mle_chi2)
                self.mle_chi2 = mle_chi2
            else:
                return
        else:
            self.mle_chi2 = mle_chi2

        self.mle_cube = mle_cube
        mle_no_cube = mle_cube.copy()
        for ii, par_i in enumerate(self.like.free_params):
            scale_i = par_i.max_value - par_i.min_value
            mle_no_cube[ii] = par_i.value_from_cube(mle_cube[ii])

        print("Fit params cube:", self.mle_cube, flush=True)
        print("Fit params no cube:", mle_no_cube, flush=True)

        like_pars = self.like.parameters_from_sampling_point(self.mle_cube)
        star_pars = self.like.theory.get_blob_fixed_background(like_pars)
        self.mle_cosmo = {}
        self.mle_cosmo["Delta2_star"] = star_pars[0]
        self.mle_cosmo["n_star"] = star_pars[1]
        self.mle_cosmo["alpha_star"] = star_pars[2]
        # errors from Hessian do not work
        # self.mle_cosmo = self.get_cosmo_err(log_func_minimize)

        # apply blinding
        self.mle_cosmo = self.apply_blinding(self.mle_cosmo, sample="mle")

        self.lnprop_mle, *blobs = self.like.log_prob_and_blobs(self.mle_cube)

        self.mle = {}
        for ii, par in enumerate(self.paramstrings):
            self.mle[par] = mle_no_cube[ii]
        for par in self.mle_cosmo:
            self.mle[par] = self.mle_cosmo[par]

        if "A_s" not in self.paramstrings[0]:
            return
        self.mle = self.apply_blinding(self.mle, conv=True)

        for key in self.blind:
            if self.blind[key] != 0:
                print("Results are blinded")
            else:
                print("Results are not blinded")

        for par in self.mle_cosmo:
            if par == "Delta2_star":
                if self.like.truth is not None:
                    print("MLE, Truth, MLE/Truth - 1")
                else:
                    print("MLE")

            val = np.round(self.mle_cosmo[par], 5)
            if self.like.truth is not None:
                if par in self.like.truth["like_params"]:
                    true = np.round(self.like.truth["like_params"][par], 5)
                    rat = np.round(
                        self.mle_cosmo[par]
                        / self.like.truth["like_params"][par]
                        - 1,
                        5,
                    )
                    print(par, val, true, rat)
            else:
                print(par, val)

    def get_cosmo_err(self, fun_minimize):
        """Deprecated

        Getting errors from Hessian does not work properly, I tested many methods to get the
        Hessian, including Iminuit, and results very bad

        """

        import numdifftools as nd

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

        cov = hess(self.mle_cube)
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

    def get_initial_walkers(self, pini=None, rms=0.01):
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

    def get_chain(
        self, cube=True, extra_nburn=0, delta_lnprob_cut=None, collapse=True
    ):
        """Figure out whether chain has been read from file, or computed.
        - if cube=True, return values in range [0,1]
        - if delta_lnprob_cut is set, use it to remove low-prob islands"""

        # mask walkers not converged
        if self.explore == False:
            mask, _ = purge_chains(self.lnprob[extra_nburn:, :])
        else:
            mask = np.ones(self.lnprob.shape[1], dtype=bool)

        # step, walker, param

        if collapse:
            lnprob = self.lnprob[extra_nburn:, mask].reshape(-1)
            chain = self.chain[extra_nburn:, mask, :].reshape(
                -1, self.chain.shape[-1]
            )
            blobs = self.blobs[extra_nburn:, mask].reshape(-1)
        else:
            lnprob = self.lnprob[extra_nburn:, mask]
            chain = self.chain[extra_nburn:, mask, :]
            blobs = self.blobs[extra_nburn:, mask]

        if delta_lnprob_cut:
            max_lnprob = np.max(lnprob)
            cut_lnprob = max_lnprob - delta_lnprob_cut
            mask = lnprob > cut_lnprob
            chain = chain[mask]
            lnprob = lnprob[mask]
            blobs = blobs[mask]

        if cube == False:
            cube_values = np.zeros_like(chain)
            for ip in range(chain.shape[-1]):
                cube_values[..., ip] = self.like.free_params[
                    ip
                ].value_from_cube(chain[..., ip])

            return cube_values, lnprob, blobs
        else:
            return chain, lnprob, blobs

    def get_all_params(
        self, delta_lnprob_cut=None, extra_nburn=0, collapse=True
    ):
        """Get a merged array of both sampled and derived parameters
        returns a 2D array of all parameters, and an ordered list of
        the LaTeX strings for each.
            - if delta_lnprob_cut is set, keep only high-prob points"""

        chain, lnprob, blobs = self.get_chain(
            cube=False,
            delta_lnprob_cut=delta_lnprob_cut,
            extra_nburn=extra_nburn,
            collapse=collapse,
        )

        return_all = False
        if collapse:
            if len(blobs[0]) == 6:
                return_all = True
        else:
            if len(blobs[0, 0]) == 6:
                return_all = True

        if return_all:
            # Build an array of chain + blobs
            all_params = np.zeros((*chain.shape[:-1], chain.shape[-1] + 6))

            all_params[..., : chain.shape[-1]] = chain
            for ii in range(6):
                all_params[..., chain.shape[-1] + ii] = blobs[
                    blob_strings_orig[ii]
                ]

            # Ordered strings for all parameters
            all_strings = self.paramstrings + blob_strings
        else:
            all_params = chain
            all_strings = self.paramstrings

        return all_params, all_strings, lnprob

    def _setup_chain_folder(self, rootdir=None, subfolder=None):
        """Set up a directory to save files for this sampler run"""

        if rootdir:
            chain_location = rootdir
        else:
            repo = os.path.dirname(cup1d.__path__[0])
            chain_location = os.path.join(repo, "data", "chains")
        if subfolder:
            # If there is one, check if it exists, if not make it
            subfolder_dir = os.path.join(chain_location, subfolder)
            if not os.path.isdir(subfolder_dir):
                os.makedirs(subfolder_dir)
            base_string = os.path.join(subfolder_dir, "chain_")
        else:
            base_string = os.path.join(chain_location, "chain_")

        # Create a new folder for this chain
        chain_count = 1
        while True:
            sampler_directory = base_string + str(chain_count)
            if os.path.isdir(sampler_directory):
                chain_count += 1
                continue
            else:
                try:
                    os.makedirs(sampler_directory)
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

    def set_blinding(self):
        """Set the blinding parameters"""
        blind_prior = {"Delta2_star": 0.05, "n_star": 0.01, "alpha_star": 0.005}
        if self.like.data.apply_blinding:
            seed = int.from_bytes(
                self.like.data.blinding.encode("utf-8"), byteorder="big"
            )
            rng = np.random.default_rng(seed)
        self.blind = {}
        for key in blind_prior:
            if self.like.data.apply_blinding:
                self.blind[key] = rng.normal(0, blind_prior[key])
            else:
                self.blind[key] = 0

    def apply_blinding(self, dict_cosmo, conv=False, sample=None):
        """Apply blinding to the dict_cosmo"""

        if self.like.data.apply_blinding:
            if sample is not None:
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

    def save_fitter(self, save_chains=False):
        """Write flat chain to file"""

        dict_out = {}

        # ARGS
        dict_out["args"] = {}
        for key in self.like.args:
            dict_out["args"][key] = self.like.args[key]

        # DATA
        dict_out["data"] = {}
        dict_out["data"]["data_label"] = self.like.data.data_label
        dict_out["data"]["zs"] = self.like.data.z
        dict_out["data"]["k_kms"] = self.like.data.k_kms
        dict_out["data"]["Pk_kms"] = self.like.data.Pk_kms
        dict_out["data"]["cov_Pk_kms"] = self.like.data.cov_Pk_kms
        if self.like.data.full_Pk_kms is not None:
            dict_out["data"]["full_Pk_kms"] = self.like.data.full_Pk_kms
            dict_out["data"]["full_cov_Pk_kms"] = self.like.data.full_cov_Pk_kms

        # EMULATOR
        dict_out["emulator"] = {}
        dict_out["emulator"][
            "emulator_label"
        ] = self.like.theory.emulator.emulator_label
        dict_out["emulator"]["kmax_Mpc"] = self.like.theory.emulator.kmax_Mpc

        # LIKELIHOOD
        dict_out["like"] = {}
        dict_out["like"]["cosmo_fid_label"] = self.like.fid
        dict_out["like"]["emu_cov_factor"] = self.like.emu_cov_factor
        dict_out["like"]["free_params"] = self.like.free_param_names

        # SAMPLER
        if save_chains:
            dict_out["sampler"] = {}
            dict_out["sampler"]["nsteps"] = self.nsteps
            dict_out["sampler"]["burnin_nsteps"] = self.burnin_nsteps
            dict_out["sampler"]["nwalkers"] = self.nwalkers

        # TRUTH
        dict_out["truth"] = self.truth

        # IGM
        like_params = self.like.parameters_from_sampling_point(self.mle_cube)
        dict_out["IGM"] = {}
        zs = dict_out["data"]["zs"]
        dict_out["IGM"]["z"] = zs
        dict_out["IGM"][
            "tau_eff"
        ] = self.like.theory.model_igm.F_model.get_tau_eff(
            zs, like_params=like_params
        )
        dict_out["IGM"]["gamma"] = self.like.theory.model_igm.T_model.get_gamma(
            zs, like_params=like_params
        )
        dict_out["IGM"][
            "sigT_kms"
        ] = self.like.theory.model_igm.T_model.get_sigT_kms(
            zs, like_params=like_params
        )
        dict_out["IGM"][
            "kF_kms"
        ] = self.like.theory.model_igm.P_model.get_kF_kms(
            zs, like_params=like_params
        )

        # NUISANCE
        dict_out["nuisance"] = {}
        # dict_out["nuisance"]["z"] = zs
        # # HCD
        # hcd_model = self.like.args["hcd_model_type"]
        # dict_out["nuisance"]["HCD"] = {}
        # dict_out["nuisance"]["HCD"]["hcd_model_type"] = hcd_model
        # dict_out["nuisance"]["HCD"][
        #     "A_damp"
        # ] = self.like.theory.model_cont.hcd_model.get_A_damp(
        #     zs, like_params=like_params
        # )
        # if hcd_model == "new":
        #     dict_out["nuisance"]["HCD"][
        #         "A_scale"
        #     ] = self.like.theory.model_cont.hcd_model.get_A_scale(
        #         zs, like_params=like_params
        #     )
        # # AGN
        # dict_out["nuisance"][
        #     "AGN"
        # ] = self.like.theory.model_cont.agn_model.get_AGN_damp(
        #     zs, like_params=like_params
        # )
        # # Metals
        # metal_models = self.like.theory.model_cont.metal_models
        # for model_name in metal_models:
        #     X_model = metal_models[model_name]
        #     f = X_model.get_amplitude(zs, like_params=like_params)
        #     adamp = X_model.get_damping(zs, like_params=like_params)
        #     alpha = X_model.get_exp_damping(zs, like_params=like_params)
        #     dict_out["nuisance"][X_model.metal_label] = {}
        #     dict_out["nuisance"][X_model.metal_label]["f"] = f
        #     dict_out["nuisance"][X_model.metal_label]["d"] = adamp
        #     dict_out["nuisance"][X_model.metal_label]["a"] = alpha

        # FITTER
        dict_out["fitter"] = {}
        dict_out["fitter"]["mle_cube"] = self.mle_cube
        dict_out["fitter"]["mle_cosmo"] = self.mle_cosmo
        dict_out["fitter"]["mle"] = self.mle
        dict_out["fitter"]["lnprob_mle"] = self.lnprop_mle

        if save_chains:
            dict_out["fitter"]["lnprob"] = self.lnprob
            dict_out["fitter"]["chain"] = self.chain
            dict_out["fitter"]["blobs"] = self.blobs

            dict_out["fitter"]["chain_from_cube"] = {}
            for ip in range(self.chain.shape[-1]):
                param = self.like.free_params[ip]
                dict_out["fitter"]["chain_from_cube"][param.name] = np.zeros(2)
                dict_out["fitter"]["chain_from_cube"][param.name][
                    0
                ] = param.min_value
                dict_out["fitter"]["chain_from_cube"][param.name][
                    1
                ] = param.max_value

            dict_out["fitter"]["chain_names_latex"] = self.paramstrings
            dict_out["fitter"]["blobs_names"] = blob_strings_orig
            dict_out["fitter"]["blobs_names_latex"] = blob_strings
            dict_out["fitter"]["chain_names"] = []
            for key in dict_out["fitter"]["chain_names_latex"]:
                dict_out["fitter"]["chain_names"].append(param_dict_rev[key])

        out_file = self.save_directory + "/fitter_results.npy"
        print("Saving chain to " + out_file)
        np.save(out_file, dict_out)


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
    "H0": "$H_0$",
    "mnu": "$\Sigma m_\\nu$",
    "As": "$A_s$",
    "ns": "$n_s$",
    "nrun": "$n_\mathrm{run}$",
    "ombh2": "$\omega_b$",
    "omch2": "$\omega_c$",
    "cosmomc_theta": "$\\theta_{MC}$",
}

for ii in range(11):
    param_dict["tau_eff_" + str(ii)] = "$\tau_{\rm eff_" + str(ii) + "}$"
    param_dict["sigT_kms_" + str(ii)] = "$\sigma_{\rm T_" + str(ii) + "}$"
    param_dict["gamma_" + str(ii)] = "$\gamma_" + str(ii) + "$"
    param_dict["kF_kms_" + str(ii)] = "$k_F_" + str(ii) + "$"
    param_dict["R_coeff_" + str(ii)] = "$R_" + str(ii) + "$"
    param_dict["ln_SN_" + str(ii)] = "$\log \mathrm{SN}_" + str(ii) + "$"
    param_dict["ln_AGN_" + str(ii)] = "$\log \mathrm{AGN}_" + str(ii) + "$"
    for jj in range(1, 5):
        param_dict["HCD_damp" + str(jj) + "_" + str(ii)] = (
            "$f_{\rm HCD" + str(jj) + "}_" + str(ii) + "$"
        )
    param_dict["HCD_const_" + str(ii)] = "$c_{\rm HCD}_" + str(ii) + "$"


metal_lines = [
    "Lya_SiIII",
    "Lya_SiII",
    "SiIIa_SiIIb",
    "SiIIa_SiIII",
    "SiIIb_SiIII",
    "CIVa_CIVb",
    "MgIIa_MgIIb",
]
metal_lines_latex = {
    "Lya_SiIII": "$\mathrm{Ly}\alpha-\mathrm{SiIII}$",
    "Lya_SiII": "$\mathrm{Ly}\alpha-\mathrm{SiII}$",
    "Lya_SiIIa": "$\mathrm{Ly}\alpha-\mathrm{SiIIa}$",
    "Lya_SiIIb": "$\mathrm{Ly}\alpha-\mathrm{SiIIb}$",
    "Lya_SiIIc": "$\mathrm{Ly}\alpha-\mathrm{SiIIc}$",
    "SiIIa_SiIIb": "$\mathrm{SiIIa}_\mathrm{SiIIb}$",
    "SiIIa_SiIII": "$\mathrm{SiIIa}_\mathrm{SiIII}$",
    "SiIIb_SiIII": "$\mathrm{SiIIb}_\mathrm{SiIII}$",
    "SiII_SiIII": "$\mathrm{SiII}_\mathrm{SiIII}$",
    "SiIIc_SiIII": "$\mathrm{SiIIc}_\mathrm{SiIII}$",
    "CIVa_CIVb": "$\mathrm{CIVa}_\mathrm{CIVb}$",
    "MgIIa_MgIIb": "$\mathrm{MgIIa}_\mathrm{MgIIb}$",
}
for metal_line in metal_lines:
    for ii in range(12):
        param_dict["f_" + metal_line + "_" + str(ii)] = (
            "$\mathrm{ln}\,f("
            + metal_lines_latex[metal_line]
            + "_"
            + str(ii)
            + ")$"
        )
        param_dict["s_" + metal_line + "_" + str(ii)] = (
            "$\mathrm{ln}\,s("
            + metal_lines_latex[metal_line]
            + "_"
            + str(ii)
            + ")$"
        )
        param_dict["p_" + metal_line + "_" + str(ii)] = (
            "$p(" + metal_lines_latex[metal_line] + "_" + str(ii) + ")$"
        )

param_dict_rev = {v: k for k, v in param_dict.items()}


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

conv_strings = {
    "Delta2_star": "$\Delta^2_\star$",
    "n_star": "$n_\star$",
    "alpha_star": "$\\alpha_\star$",
}

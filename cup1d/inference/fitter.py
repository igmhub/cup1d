import copy
import os
from pathlib import Path
import time
from scipy.optimize import minimize
from scipy.linalg import block_diag
import numpy as np
from mpi4py import MPI

from cup1d.utils import blinding
from cup1d.utils.compute_hessian import get_hessian, get_hessian_rows
from cup1d.likelihood import parameter as parameter_space

# our own modules
from cup1d.utils.utils import create_print_function, purge_chains
from cup1d.utils.utils import get_path_repo
from cup1d.utils.various_dicts import (
    param_dict,
    param_dict_rev,
    blob_strings,
    blob_strings_orig,
)


class Fitter(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(
        self,
        like=None,
        nwalkers=1,
        nsteps=1,
        nburn=0,
        thin=1,
        verbose=False,
        subfolder=None,
        rootdir=None,
        parallel=False,
        explore=False,
        fix_cosmology=False,
        random_seed=None,
        create_output=True,
    ):
        """Setup sampler from likelihood, or use default.
        If read_chain_file is provided, read pre-computed chain.
        rootdir allows user to search for saved chains in a different
        location to the code itself."""

        self.parallel = parallel
        self.explore = explore
        self.verbose = verbose

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
        self.random_seed = random_seed
        self.rng = np.random.default_rng(
            np.random.SeedSequence(random_seed, spawn_key=(self.rank,))
        )

        self.print = create_print_function(self.verbose)

        self.like = like
        # number of free parameters to sample
        self.thin = thin
        self.nburn = nburn
        self.nsteps = nsteps
        self.ndim = len(self.like.free_params)

        if nwalkers < 2 * self.ndim:
            self.nwalkers = 2 * self.ndim + 1
        else:
            self.nwalkers = nwalkers

        if self.rank == 0:
            if create_output:
                self._setup_chain_folder(rootdir, subfolder)
            else:
                self.save_directory = None

            # number of walkers
            # if nwalkers is not None:
            # if nwalkers < 2 * self.ndim:
            #     nwalkers = 2 * self.ndim + 1
            #     self.print(
            #         "nwalkers={} ; ndim={}".format(nwalkers, self.ndim)
            #     )
            # else:
            #     max_walkers = 40 * self.ndim
            #     min_walkers = 2 * self.ndim
            #     nwalkers = max_walkers // self.size + 1
            #     combined_steps = max_walkers * (nsteps + self.nburn)

            #     if nwalkers < min_walkers:
            #         nwalkers = min_walkers
            #         nsteps = (
            #             combined_steps // (nwalkers * self.size) - self.nburn
            #         )

            for irank in range(1, self.size):
                # self.comm.send(nsteps, dest=irank, tag=irank * 11)
                # self.comm.send(nwalkers, dest=irank, tag=irank * 13)
                self.comm.send(self.save_directory, dest=irank, tag=irank * 15)

        else:
            # nsteps = self.comm.recv(source=0, tag=self.rank * 11)
            # nwalkers = self.comm.recv(source=0, tag=self.rank * 13)
            self.save_directory = self.comm.recv(source=0, tag=self.rank * 15)

        # self.print(
        #     "rank", self.rank, "nwalkers", self.nwalkers, "nsteps", self.nsteps
        # )

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
            self.nwalkers * self.size * (self.nsteps + self.nburn),
            "useful steps ",
            self.nwalkers * self.size * (self.nsteps),
        )

        ## Set up list of parameter names in tex format for plotting
        self.paramstrings = [
            param_dict[name] for name in self.like.free_params
        ]

        # when running on simulated data, we can store true cosmo values
        self.set_truth()

        # Figure out what extra information will be provided as blobs
        self.blobs_dtype = self.like.theory.get_blobs_dtype()
        self.mle = None

    def sampling_point_from_parameters(self, parameters=None):
        """Convert physical values to the optimizer unit cube."""

        return parameter_space.values_to_cube(self.like.free_params, parameters)

    def parameters_from_sampling_point(self, values):
        """Convert optimizer coordinates to physical values."""

        return parameter_space.values_from_cube(self.like.free_params, values)

    def value_in_cube(self, name, value=None):
        return parameter_space.value_in_cube(self.like.free_params, name, value)

    def value_from_cube(self, name, value):
        return parameter_space.value_from_cube(self.like.free_params, name, value)

    def error_from_cube(self, name, error):
        return parameter_space.error_from_cube(self.like.free_params, name, error)

    def get_mle_latex(self):
        """Return MLE values keyed by presentation-only LaTeX labels."""

        return {
            self.param_dict.get(name, name): value
            for name, value in self.mle.items()
        }

    def get_truth_latex(self):
        """Return truth values keyed by presentation-only LaTeX labels."""

        if self.truth is None:
            return None
        return {
            self.param_dict.get(name, name): value
            for name, value in self.truth.items()
        }

    def _prediction_vector_and_icov(self, values, zmask=None):
        """Return the model vector and matching fixed inverse covariance."""

        parameters = self.parameters_from_sampling_point(values)
        result = self.like.get_p1d_kms(parameters)
        if result is None:
            raise ValueError("cannot estimate errors outside the emulator domain")
        predictions = result[0]
        model_vectors = []
        covariance_blocks = []
        for key, data in self.like.data.items():
            if zmask is None:
                indices = np.arange(len(data.z))
            else:
                indices = np.flatnonzero(
                    np.any(
                        np.isclose(
                            np.asarray(data.z)[:, None],
                            np.atleast_1d(zmask)[None, :],
                            atol=1.0e-3,
                            rtol=0,
                        ),
                        axis=1,
                    )
                )
            if len(indices) == 0:
                continue
            model_vectors.append(
                np.concatenate(
                    [np.asarray(predictions[key][index]).reshape(-1) for index in indices]
                )
            )
            full_icov = self.like.full_icov_Pk_kms[key]
            if full_icov is not None and zmask is None:
                covariance_blocks.append(full_icov)
            else:
                covariance_blocks.append(
                    block_diag(*[self.like.icov_Pk_kms[key][index] for index in indices])
                )
        if not model_vectors:
            raise ValueError("zmask does not select any data bins")
        return np.concatenate(model_vectors), block_diag(*covariance_blocks)

    def _gauss_newton_hessian(self, hessian_step, zmask=None):
        """Approximate posterior curvature from first model derivatives."""

        parameters = list(self.like.free_params.values())
        exponential = np.asarray([
            parameter.get("hessian_transform") == "exp"
            for parameter in parameters
        ])
        hessian_point = self.mle_cube.copy()
        amplitude_min = np.exp([parameter["min_value"] for parameter in parameters])
        amplitude_max = np.exp([parameter["max_value"] for parameter in parameters])
        log_width = np.asarray([
            parameter["max_value"] - parameter["min_value"]
            for parameter in parameters
        ])
        amplitude_range = amplitude_max - amplitude_min
        if np.any(exponential):
            log_values = np.asarray([
                parameter_space.value_from_cube(self.like.free_params, name, value)
                for name, value in zip(self.like.free_params, self.mle_cube)
            ])
            hessian_point[exponential] = (
                np.exp(log_values[exponential]) - amplitude_min[exponential]
            ) / amplitude_range[exponential]

        def hessian_to_cube(point):
            cube = np.asarray(point).copy()
            amplitudes = amplitude_min[exponential] + (
                point[exponential] * amplitude_range[exponential]
            )
            cube[exponential] = (
                np.log(amplitudes)
                - np.asarray([parameter["min_value"] for parameter in parameters])[exponential]
            ) / log_width[exponential]
            return cube

        cube_scale = np.ones(self.ndim)
        if np.any(exponential):
            amplitudes = amplitude_min[exponential] + (
                hessian_point[exponential] * amplitude_range[exponential]
            )
            cube_scale[exponential] = amplitude_range[exponential] / (
                log_width[exponential] * amplitudes
            )

        model, inverse_covariance = self._prediction_vector_and_icov(
            hessian_to_cube(hessian_point), zmask=zmask
        )
        jacobian = np.empty((model.size, self.ndim))
        for index in range(self.ndim):
            step = min(
                hessian_step, hessian_point[index], 1.0 - hessian_point[index]
            )
            direction = np.zeros(self.ndim)
            if step > 0:
                direction[index] = step
                plus, _ = self._prediction_vector_and_icov(
                    hessian_to_cube(hessian_point + direction), zmask=zmask
                )
                minus, _ = self._prediction_vector_and_icov(
                    hessian_to_cube(hessian_point - direction), zmask=zmask
                )
                jacobian[:, index] = (plus - minus) / (2 * step)
            elif self.mle_cube[index] < 1.0:
                step = min(hessian_step, 1.0 - self.mle_cube[index])
                direction[index] = step
                plus, _ = self._prediction_vector_and_icov(
                    hessian_to_cube(hessian_point + direction), zmask=zmask
                )
                jacobian[:, index] = (plus - model) / step
            else:
                step = min(hessian_step, self.mle_cube[index])
                direction[index] = step
                minus, _ = self._prediction_vector_and_icov(
                    hessian_to_cube(hessian_point - direction), zmask=zmask
                )
                jacobian[:, index] = (model - minus) / step
        hessian = jacobian.T @ inverse_covariance @ jacobian
        hessian = hessian * np.outer(cube_scale, cube_scale)
        priors = self.like.Gauss_priors
        if priors is not None:
            scales = np.asarray([
                parameter["max_value"] - parameter["min_value"]
                for parameter in self.like.free_params.values()
            ])
            hessian += np.diag((scales / priors) ** 2)
        return hessian

    def estimate_mle_errors(
        self, hessian_step=1.0e-4, zmask=None, method="finite_difference"
    ):
        """Estimate local MLE errors from the negative-log-posterior Hessian.

        The covariance is evaluated in the unit cube and propagated to
        physical likelihood parameters and compressed linear-power parameters.
        """

        if not hasattr(self, "mle_cube"):
            raise ValueError("run a minimizer or set an MLE before estimating errors")
        if hessian_step <= 0:
            raise ValueError("hessian_step must be positive")

        if method == "finite_difference":
            objective = lambda point: self.minus_log_prob(point, zmask=zmask)
            hessian = get_hessian(objective, self.mle_cube, hh=hessian_step)
        elif method == "gauss_newton":
            hessian = self._gauss_newton_hessian(hessian_step, zmask=zmask)
        elif method == "hybrid":
            hessian = self._gauss_newton_hessian(hessian_step, zmask=zmask)
            cosmology_names = set(
                self.like.theory.fid_cosmo["cosmo"].input_cosmo_params_dict
            )
            indices = [
                index
                for index, name in enumerate(self.like.free_params)
                if name in cosmology_names
            ]
            objective = lambda point: self.minus_log_prob(point, zmask=zmask)
            exact_rows = get_hessian_rows(
                objective, self.mle_cube, indices, hh=hessian_step
            )
            hessian[indices, :] = exact_rows[indices, :]
            hessian[:, indices] = exact_rows[indices, :].T
        else:
            raise ValueError(
                "method must be 'finite_difference', 'gauss_newton', or 'hybrid'"
            )
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError(f"{method} curvature contains non-finite values")
        curvature_scale = max(1.0, np.max(np.abs(eigenvalues)))
        # Distinguish numerical null modes from physically weak directions.
        # The latter matter here: nuisance parameters at a hard prior boundary
        # may vary inward and must remain marginalized in cosmology errors.
        tolerance = np.finfo(float).eps * max(self.ndim, 1) * curvature_scale
        if np.min(eigenvalues) < -tolerance:
            raise ValueError(
                f"{method} curvature has a negative eigenvalue "
                f"({np.min(eigenvalues):.3e}); the MLE is not locally convex."
            )
        null_modes = eigenvalues <= tolerance
        inverse_eigenvalues = np.zeros_like(eigenvalues)
        inverse_eigenvalues[~null_modes] = 1.0 / eigenvalues[~null_modes]

        self.mle_error_method = method
        self.mle_hessian = hessian
        self.mle_hessian_rank = int(np.count_nonzero(~null_modes))
        self.mle_covariance_cube = (
            eigenvectors * inverse_eigenvalues
        ) @ eigenvectors.T
        self.mle_null_modes = eigenvectors[:, null_modes]
        scales = np.asarray([
            parameter["max_value"] - parameter["min_value"]
            for parameter in self.like.free_params.values()
        ])
        self.mle_covariance = self.mle_covariance_cube * np.outer(scales, scales)
        null_weight = np.sum(self.mle_null_modes**2, axis=1)
        self.mle_errors = {
            name: (
                np.inf
                if null_weight[index] > 1.0e-10
                else np.sqrt(self.mle_covariance[index, index])
            )
            for index, name in enumerate(self.like.free_params)
        }

        def star_parameters(point):
            parameters = self.parameters_from_sampling_point(point)
            return np.asarray(self.like.theory.get_blob_for_parameters(parameters)[:3])

        central_star = star_parameters(self.mle_cube)
        jacobian = np.zeros((3, self.ndim))
        cosmology_names = set(
            self.like.theory.fid_cosmo["cosmo"].input_cosmo_params_dict
        )
        for index, name in enumerate(self.like.free_params):
            if name not in cosmology_names:
                continue
            step = min(hessian_step, self.mle_cube[index], 1.0 - self.mle_cube[index])
            direction = np.zeros(self.ndim)
            if step > 0:
                direction[index] = step
                jacobian[:, index] = (
                    star_parameters(self.mle_cube + direction)
                    - star_parameters(self.mle_cube - direction)
                ) / (2 * step)
            elif self.mle_cube[index] < 1.0:
                step = min(hessian_step, 1.0 - self.mle_cube[index])
                direction[index] = step
                jacobian[:, index] = (
                    star_parameters(self.mle_cube + direction) - central_star
                ) / step
            else:
                step = min(hessian_step, self.mle_cube[index])
                direction[index] = step
                jacobian[:, index] = (
                    central_star - star_parameters(self.mle_cube - direction)
                ) / step

        # Null nuisance modes are handled by the pseudoinverse above. Their
        # numerical projections onto cosmology after a coordinate transform do
        # not by themselves invalidate the finite identifiable-subspace error.

        self.mle_cosmo_covariance = (
            jacobian @ self.mle_covariance_cube @ jacobian.T
        )
        errors = np.sqrt(np.diag(self.mle_cosmo_covariance))
        self.mle_cosmo_correlation = self.mle_cosmo_covariance / np.outer(
            errors, errors
        )
        cosmo_names = ("Delta2_star", "n_star", "alpha_star")
        self.mle_cosmo_errors = {
            name: errors[index]
            for index, name in enumerate(cosmo_names)
        }
        return self.mle_errors

    def get_chi2(self, values, **kwargs):
        """Evaluate chi-squared from optimizer coordinates."""

        parameters = self.parameters_from_sampling_point(values)
        return self.like.get_chi2(parameters, **kwargs)

    def log_prob(self, values, **kwargs):
        """Evaluate posterior probability from sampler coordinates."""

        parameters = self.parameters_from_sampling_point(values)
        return self.like.log_prob(parameters, **kwargs)

    def log_prob_and_blobs(self, values, **kwargs):
        """Evaluate posterior and blobs from sampler coordinates."""

        parameters = self.parameters_from_sampling_point(values)
        return self.like.log_prob_and_blobs(parameters, **kwargs)

    def minus_log_prob(self, values, zmask=None, ind_fix=None, pfix=None):
        """Negative posterior in optimizer coordinates."""

        values = np.asarray(values).copy()
        if ind_fix is not None:
            values[ind_fix] = pfix
        return -self.log_prob(values, zmask=zmask)

    def set_truth(self):
        """Set up dictionary with true values of cosmological
        likelihood parameters for plotting purposes"""

        # likelihood contains true parameters, but not in latex names
        like_truth = self.like.truth

        # when running on data, we do not know the truth
        if like_truth is None:
            self.truth = None
            return

        # Keep persisted truth keyed by canonical parameter names.
        self.truth = dict(like_truth["like_params"])

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

        import emcee

        if log_func is None:
            _log_func = self.log_prob_and_blobs
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
            self._seed_sampler(sampler)
            self.print(
                f"Running MCMC with {self.nwalkers} walkers, {self.ndim} dimensions, and {self.nsteps}, {self.nburn}.",
            )
            for sample in sampler.sample(p0, iterations=self.nburn + self.nsteps):
                if sampler.iteration % 100 == 0:
                    self.print(
                        "Step %d out of %d "
                        % (sampler.iteration, self.nburn + self.nsteps)
                    )

            ## Get samples, flat=False to be able to mask not converged chains latter
            self.lnprob = sampler.get_log_prob(
                flat=False, discard=self.nburn, thin=self.thin
            )
            self.chain = sampler.get_chain(
                flat=False, discard=self.nburn, thin=self.thin
            )
            self.blobs = sampler.get_blobs(
                flat=False, discard=self.nburn, thin=self.thin
            )
        else:
            # MPIPool does not work for me in nersc for whatever reason
            # I need to get creative

            p0 = self.get_initial_walkers(pini=pini)
            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, log_func, blobs_dtype=self.blobs_dtype
            )

            self._seed_sampler(sampler)
            for sample in sampler.sample(
                p0,
                iterations=self.nburn + self.nsteps,
                skip_initial_state_check=True,
            ):
                if sampler.iteration % 100 == 0:
                    self.print(
                        "Step %d out of %d "
                        % (sampler.iteration, self.nsteps + self.nburn)
                    )

            self.print(f"Rank {self.rank} done")
            _lnprob = sampler.get_log_prob(
                flat=False, discard=self.nburn, thin=self.thin
            )
            _chain = sampler.get_chain(flat=False, discard=self.nburn, thin=self.thin)
            _blobs = sampler.get_blobs(flat=False, discard=self.nburn, thin=self.thin)

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
                    lnprob.append(self.comm.recv(source=irank, tag=1000 + irank))
                    chain.append(self.comm.recv(source=irank, tag=2000 + irank))
                    blobs.append(self.comm.recv(source=irank, tag=3000 + irank))

                self.lnprob = np.concatenate(lnprob, axis=1)
                self.chain = np.concatenate(chain, axis=1)
                self.blobs = np.concatenate(blobs, axis=1)

        if self.rank == 0:
            map_ind = np.argmax(self.lnprob.reshape(-1))
            map_chi2 = -2.0 * self.lnprob.reshape(-1)[map_ind]
            map_chain = self.chain.reshape(-1, self.chain.shape[-1])[map_ind]
            self.set_mle(map_chain, map_chi2, force=True)
            # Blinding affects only derived star-parameter blobs.
            self.blobs = blinding.apply_blinding(self.like.blind, self.blobs)

        return sampler

    def run_minimizer(
        self,
        log_func_minimize=None,
        p0=None,
        burn_in=False,
        zmask=None,
        mask_pars=False,
        restart=False,
        neval=1000,
        chi2_tol=0.1,
        estimate_errors=False,
        hessian_step=1.0e-4,
        error_method="finite_difference",
    ):
        """Minimizer"""

        def set_log_func_minimize(pini, zmask=None, mask_pars=False):
            if mask_pars == False:
                if zmask is not None:
                    fun = lambda x: log_func_minimize(x, zmask=zmask)
                    return fun
                else:
                    return log_func_minimize
            else:
                ind_fix = []
                for ii, parameter in enumerate(self.like.free_params.values()):
                    if parameter["fixed"]:
                        ind_fix.append(ii)
                ind_fix = np.array(ind_fix)
                pfix = pini[ind_fix]
                if zmask is not None:
                    fun = lambda x: log_func_minimize(
                        x, zmask=zmask, ind_fix=ind_fix, pfix=pfix
                    )
                    return fun
                else:
                    fun = lambda x: log_func_minimize(x, ind_fix=ind_fix, pfix=pfix)
                    return fun

        _log_func_minimize = set_log_func_minimize(p0, zmask=zmask, mask_pars=mask_pars)

        if restart:
            self.mle_chi2 = 1e10

        npars = len(self.like.free_params)

        if p0 is not None:
            # start at the initial value
            mle_cube = p0.copy()
        else:
            # star at the center of the parameter space
            mle_cube = np.ones(npars) * 0.5

        if burn_in:

            from scipy.stats import qmc

            # random starting points
            _chi2 = 1e10
            nsamples = 25
            sig = 0.25
            niter = 1

            lhs_sampler = qmc.LatinHypercube(d=npars, seed=42)
            arr_p0 = lhs_sampler.random(n=nsamples)

            # star minimization at different points, keep best
            # we hope it is easier to get to the local minima
            for it in range(niter):
                if it == 0:
                    pnext0 = mle_cube.copy()
                self.print("it, pnext0", it, pnext0[:2])
                for ii in range(nsamples):
                    pini = pnext0 + (arr_p0[ii, :] - 0.5) * sig / (ii + 1)
                    pini[pini <= 0] = 0.05
                    pini[pini >= 1] = 0.95

                    res = minimize(
                        _log_func_minimize,
                        pini,
                        method="Nelder-Mead",
                        bounds=((0.0, 1.0),) * npars,
                        options={
                            "fatol": chi2_tol,  # fatol and xatol are both evaluated
                            "xatol": 1e-6,  # needed to reach the good convergence
                            "maxiter": neval,
                            "maxfev": neval,
                        },
                    )
                    self.print("ITER", it, ii, res.fun, pini[:2], res.x[:2])
                    if res.fun < _chi2:
                        _chi2 = res.fun
                        pnext = res.x
                pnext0 = pnext.copy()
            mle_cube = pnext0

        chi2 = self.get_chi2(mle_cube, zmask=zmask)
        chi2_ini = chi2 * 1

        self.print("Starting NM minimization, chi2=", chi2)
        keep = True
        ii = 0
        rep = 0
        start = time.time()
        while keep:
            start1 = time.time()
            pini = mle_cube.copy()

            res = minimize(
                _log_func_minimize,
                pini,
                method="Nelder-Mead",
                bounds=((0.0, 1.0),) * npars,
                options={
                    "fatol": chi2_tol,  # fatol and xatol are both evaluated
                    "xatol": 1e-6,  # needed to reach the good convergence
                    "maxiter": neval,
                    "maxfev": neval,
                },
            )
            # self.print(res)

            _chi2 = self.get_chi2(res.x, zmask=zmask)
            diff_chi = _chi2 - chi2

            self.print(
                "Step, rep, time",
                ii,
                rep,
                np.round(time.time() - start1, 2),
                np.round(time.time() - start, 2),
            )
            self.print(
                "Minimization improved (ini, last, now, diff):",
                np.round(chi2_ini, 4),
                np.round(chi2, 4),
                np.round(_chi2, 4),
                np.round(diff_chi, 4),
            )

            if res.success:
                keep = False
            else:
                if -diff_chi > chi2_tol:
                    chi2 = _chi2.copy()
                    mle_cube = res.x.copy()
                    rep = 0
                elif diff_chi < 0:
                    chi2 = _chi2.copy()
                    mle_cube = res.x.copy()
                    rep += 1
                else:
                    rep += 1

            if rep >= 2:
                # multiple times to ensure that it is real
                keep = False

            ii += 1

        mle_cube = res.x
        chi2 = self.get_chi2(mle_cube, zmask=zmask)
        self.print("Passed out:", chi2)
        _ = (mle_cube > 0.95) | (mle_cube < 0.05)
        if np.sum(_) > 0:
            self.print(
                "Almost out of bounds:",
            )
            _ = np.argwhere((mle_cube > 0.95) | (mle_cube < 0.05))[:, 0]
            for ii in range(len(_)):
                ind = _[ii]
                self.print(
                    self.like.free_param_names[ind],
                    mle_cube[ind],
                    self.value_from_cube(
                        self.like.free_param_names[ind], mle_cube[ind]
                    ),
                )
        self.set_mle(mle_cube, chi2)
        if estimate_errors:
            self.estimate_mle_errors(
                hessian_step=hessian_step, zmask=zmask, method=error_method
            )

    def run_minimizer_da(
        self,
        log_func_minimize=None,
        p0=None,
        zmask=None,
        mask_pars=None,
        restart=True,
        estimate_errors=False,
        hessian_step=1.0e-4,
    ):
        """Minimizer using dual annealing"""

        from scipy.optimize import dual_annealing

        def set_log_func_minimize(pini, zmask=None, mask_pars=None):
            if mask_pars is None:
                if zmask is not None:
                    fun = lambda x: log_func_minimize(x, zmask=zmask)
                    return fun
                else:
                    return log_func_minimize
            else:
                ind_fix = []
                for ii, parameter in enumerate(self.like.free_params.values()):
                    if parameter["fixed"]:
                        ind_fix.append(ii)
                ind_fix = np.array(ind_fix)
                pfix = pini[ind_fix]
                if zmask is not None:
                    fun = lambda x: log_func_minimize(
                        x, zmask=zmask, ind_fix=ind_fix, pfix=pfix
                    )
                    return fun
                else:
                    fun = lambda x: log_func_minimize(x, ind_fix=ind_fix, pfix=pfix)
                    return fun

        if restart:
            self.mle_chi2 = 1e10

        _log_func_minimize = set_log_func_minimize(p0, zmask=zmask, mask_pars=mask_pars)

        npars = len(self.like.free_params)

        if p0 is not None:
            # start at the initial value
            mle_cube = p0.copy()
        else:
            # star at the center of the parameter space
            mle_cube = np.ones(npars) * 0.5

        mle_cube[mle_cube <= 0] = 0.05
        mle_cube[mle_cube >= 1] = 0.95

        chi2 = self.get_chi2(mle_cube, zmask=zmask)
        chi2_ini = chi2 * 1

        self.print("Starting DA minimization")

        start = time.time()
        res = dual_annealing(
            _log_func_minimize,
            maxiter=10000,
            x0=mle_cube,
            bounds=((0.0, 1.0),) * npars,
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "bounds": ((0.0, 1.0),) * npars,
                "options": {"fatol": 0.1, "xatol": 0},
            },
        )
        self.print(res)

        _chi2 = self.get_chi2(res.x, zmask=zmask)

        if _chi2 < chi2:
            chi2 = _chi2.copy()
            mle_cube = res.x.copy()

        self.print("Step took:", np.round(time.time() - start, 2))
        self.print(
            "Minimization improved:",
            np.round(chi2_ini, 4),
            np.round(chi2, 4),
            np.round(chi2_ini - chi2, 4),
        )

        self.set_mle(mle_cube, chi2)
        if estimate_errors:
            self.estimate_mle_errors(
                hessian_step=hessian_step, zmask=zmask, method=error_method
            )

    def set_mle(self, mle_cube, mle_chi2, force=False):
        """Set the maximum likelihood solution"""

        if hasattr(self, "mle_chi2") and not force:
            if mle_chi2 < self.mle_chi2:
                self.print("updating mle from ", self.mle_chi2, "to", mle_chi2)
                self.mle_chi2 = mle_chi2
            else:
                return
        else:
            self.mle_chi2 = mle_chi2

        # Error estimates belong to the previous MLE and must not survive a move.
        for name in self._PERSISTED_FIT_ATTRIBUTES:
            if hasattr(self, name):
                delattr(self, name)

        self.mle_cube = mle_cube
        like_pars = self.parameters_from_sampling_point(self.mle_cube)
        mle_no_cube = np.asarray(list(like_pars.values()))

        self.print("Fit params cube:", self.mle_cube)
        self.print("Fit params no cube:", mle_no_cube)

        star_pars = self.like.theory.get_blob_for_parameters(like_pars)
        self.mle_cosmo = {}
        self.mle_cosmo["Delta2_star"] = star_pars[0]
        self.mle_cosmo["n_star"] = star_pars[1]
        self.mle_cosmo["alpha_star"] = star_pars[2]
        # apply blinding
        self.mle_cosmo = blinding.apply_blinding(self.like.blind, self.mle_cosmo)

        self.lnprop_mle, *blobs = self.log_prob_and_blobs(self.mle_cube)

        self.mle = dict(like_pars)
        self.mle.update(self.mle_cosmo)

        if "As" not in self.like.free_params:
            return

        for key in self.like.blind:
            if self.like.blind[key] != 0:
                self.print("Results are blinded")
            else:
                self.print("Results are not blinded")

        for par in self.mle_cosmo:
            if par == "Delta2_star":
                if self.like.truth is not None:
                    self.print("MLE, Truth, MLE/Truth - 1")
                else:
                    self.print("MLE")

            val = np.round(self.mle_cosmo[par], 5)
            if self.like.truth is not None:
                if par in self.like.truth["like_params"]:
                    true = np.round(self.like.truth["like_params"][par], 5)
                    rat = np.round(
                        self.mle_cosmo[par] / self.like.truth["like_params"][par] - 1,
                        5,
                    )
                    self.print(par, val, true, rat)
            else:
                self.print(par, val)

    def _seed_sampler(self, sampler):
        """Seed emcee proposal moves from this fitter’s local generator."""

        seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
        sampler.random_state = np.random.RandomState(seed).get_state()

    def get_initial_walkers(self, pini=None, rms=0.01):
        """Setup initial states of walkers in sensible points
        -- initial will set a range within unit volume around the
           fiducial values to initialise walkers (if no prior is used)"""

        ndim = self.ndim
        nwalkers = self.nwalkers

        self.print("set %d walkers with %d dimensions" % (nwalkers, ndim))

        p0 = self.rng.random(ndim * nwalkers).reshape((nwalkers, ndim))
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

        from scipy.stats import truncnorm

        rms = self.like.prior_Gauss_rms
        values = truncnorm.rvs(
            (0.0 - mean) / rms,
            (1.0 - mean) / rms,
            scale=rms,
            loc=mean,
            size=n_samples,
        )

        return values

    def get_chain(self, cube=True, extra_nburn=0, delta_lnprob_cut=None, collapse=True):
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
            chain = self.chain[extra_nburn:, mask, :].reshape(-1, self.chain.shape[-1])
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
                name = self.like.free_param_names[ip]
                cube_values[..., ip] = self.value_from_cube(
                    name, chain[..., ip]
                )

            return cube_values, lnprob, blobs
        else:
            return chain, lnprob, blobs

    def get_all_params(self, delta_lnprob_cut=None, extra_nburn=0, collapse=True):
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
                all_params[..., chain.shape[-1] + ii] = blobs[blob_strings_orig[ii]]

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
            chain_location = os.path.join(get_path_repo("cup1d"), "data", "chains")
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

    def get_best_fit(self, delta_lnprob_cut=None, stat_best_fit="mean"):
        """Return an array of best fit values (mean) from the MCMC chain,
        in unit likelihood space.
            - if delta_lnprob_cut is set, use only high-prob points"""

        if stat_best_fit == "mean":
            chain, lnprob, blobs = self.get_chain(delta_lnprob_cut=delta_lnprob_cut)
            best_values = np.mean(chain, axis=0)
        elif stat_best_fit == "median":
            chain, lnprob, blobs = self.get_chain(delta_lnprob_cut=delta_lnprob_cut)
            best_values = np.median(chain, axis=0)
        elif stat_best_fit == "mle":
            best_values = self.mle_cube
        else:
            raise ValueError(stat_best_fit + " not implemented")

        return best_values

    _PERSISTED_FIT_ATTRIBUTES = (
        "mle_error_method",
        "mle_hessian",
        "mle_hessian_rank",
        "mle_covariance_cube",
        "mle_null_modes",
        "mle_covariance",
        "mle_errors",
        "mle_cosmo_errors",
        "mle_cosmo_covariance",
        "mle_cosmo_correlation",
    )

    def _configuration_reference(self):
        """Return the YAML information required to reconstruct this fit."""

        config_path = getattr(self.like.args, "config_path", None)
        if config_path is None:
            raise ValueError(
                "Cannot save reconstructable results because Args was not "
                "created from a YAML file"
            )
        return {
            "config_path": str(Path(config_path).expanduser().resolve()),
            "config_loader": getattr(self.like.args, "config_loader", "yaml"),
            "synthetic": bool(getattr(self.like.args, "synthetic", False)),
        }

    def _fit_result(self):
        """Return only state produced by fitting, not YAML-derived inputs."""

        if self.mle is None or not hasattr(self, "mle_cube"):
            raise ValueError("No fitted result is available to save")
        result = {
            "mle_cube": np.asarray(self.mle_cube).copy(),
            "mle_chi2": float(self.mle_chi2),
        }
        for name in self._PERSISTED_FIT_ATTRIBUTES:
            if hasattr(self, name):
                result[name] = copy.deepcopy(getattr(self, name))
        return result

    def _result_payload(self, result_type):
        return {
            "format_version": 1,
            "result_type": result_type,
            **self._configuration_reference(),
            "parameter_names": list(self.like.free_param_names),
            "fit": self._fit_result(),
        }

    def save_minimizer_results(self):
        """Save a compact, YAML-backed standalone minimizer result."""

        if self.save_directory is None:
            raise ValueError("This fitter has no output directory")
        if not hasattr(self, "mle_cosmo_errors"):
            self.estimate_mle_errors(method="gauss_newton")
        out_file = Path(self.save_directory) / "minimizer_results.npy"
        payload = self._result_payload("minimizer")
        self.print(f"Saving data to {out_file}")
        np.save(out_file, payload)
        return out_file

    def save_sampler_results(self):
        """Save sampler arrays separately and reference them from its result."""

        if self.save_directory is None:
            raise ValueError("This fitter has no output directory")
        missing = [
            name for name in ("chain", "blobs", "lnprob")
            if not hasattr(self, name)
        ]
        if missing:
            raise ValueError(
                "Cannot save sampler results without " + ", ".join(missing)
            )

        directory = Path(self.save_directory).resolve()
        paths = {
            "chain_path": directory / "chain.npy",
            "blobs_path": directory / "blobs.npy",
            "lnprob_path": directory / "lnprob.npy",
        }
        np.save(paths["chain_path"], self.chain)
        np.save(paths["blobs_path"], self.blobs)
        np.save(paths["lnprob_path"], self.lnprob)

        payload = self._result_payload("sampler")
        payload.update({name: str(path) for name, path in paths.items()})
        out_file = directory / "sampler_results.npy"
        self.print(f"Saving data to {out_file}")
        np.save(out_file, payload)
        return out_file

    def restore_results(self, payload, result_path):
        """Restore fitter state from a validated result payload."""

        expected_names = list(self.like.free_param_names)
        if payload.get("parameter_names") != expected_names:
            raise ValueError(
                "Saved parameter names do not match those reconstructed "
                "from the YAML configuration"
            )
        fit = payload["fit"]
        cube = np.asarray(fit["mle_cube"], dtype=float)
        if cube.shape != (self.ndim,):
            raise ValueError(
                f"Saved MLE has shape {cube.shape}; expected {(self.ndim,)}"
            )
        self.set_mle(cube, float(fit["mle_chi2"]))
        for name in self._PERSISTED_FIT_ATTRIBUTES:
            if name in fit:
                setattr(self, name, copy.deepcopy(fit[name]))

        result_path = Path(result_path).expanduser().resolve()
        self.save_directory = str(result_path.parent)
        self.results_path = str(result_path)
        self.result_type = payload["result_type"]
        if self.result_type == "sampler":
            for attribute, key in (
                ("chain", "chain_path"),
                ("blobs", "blobs_path"),
                ("lnprob", "lnprob_path"),
            ):
                array_path = Path(payload[key]).expanduser()
                if not array_path.is_absolute():
                    array_path = result_path.parent / array_path
                setattr(self, attribute, np.load(array_path, allow_pickle=False))
        return self

    def save_fitter(self, save_chains=False):
        """Compatibility wrapper for the split result-file interface."""

        if save_chains:
            return self.save_sampler_results()
        return self.save_minimizer_results()

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline inference: minimization, PSO, and sampling
#
# This notebook uses the CM2026 baseline likelihood to compare its three
# inference routes: local Nelder--Mead minimization, particle-swarm
# optimization (PSO), and ensemble sampling with ``emcee``.
#
# PSO and ``emcee`` can evaluate a group of likelihood points at once.  The
# ``vectorize`` switch below compares that batched route to evaluating exactly
# the same kind of points one at a time.  Nelder--Mead is intrinsically scalar,
# so it is shown once.
#
# The default ``test=True`` runs are deliberately tiny demonstrations, not
# converged fits or chains.  They are intended to make timing comparisons quick.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from copy import deepcopy

from cup1d import Analysis, Args

# %% [markdown]
# ## Build one test-sized baseline analysis
#
# The baseline YAML supplies the data, theory, priors, and emulator.  They are
# loaded only once: every comparison below reuses this same ``Analysis`` and
# explicitly starts again from ``initial_point``.  We replace only sampler
# runtime settings. ``Fitter`` enforces the minimum ensemble size required by
# ``emcee``.

# %%
test = True

args = Args.from_baseline(verbose=False)
args.out_folder = str(Path("/tmp") / "cup1d_inference_methods")
args.mcmc = dict(args.mcmc)
args.mcmc.update(
    {
        "parallel": False,
        "seed": 1234,
        "n_burn_in": 0,
        "n_steps": 10 if test else 2000,
        "thin": 1,
        # Fitter raises this to 2 * ndim + 1 when necessary.
        "n_walkers": 1 if test else args.mcmc["n_walkers"],
    }
)
analysis = Analysis(args)
# This named dictionary is the public initial point: it includes each physical
# value, prior bounds, and other parameter metadata. Optimizers convert it to
# their private unit-cube array internally.
initial_point = analysis.fitter.initial_parameters()
initial_table = pd.DataFrame(initial_point).T[
    ["value", "min_value", "max_value", "Gauss_priors_width", "fixed"]
]
print(f"Free parameters: {analysis.fitter.ndim}")
print(f"Initial chi2: {analysis.fitter.get_chi2(initial_point):.3f}")

# %% [markdown]
# ## Local minimization (scalar)
#
# Nelder--Mead proposes a single point at every function evaluation; it has no
# batched counterpart.  The small evaluation budget is only for this tutorial.

# %%
start = perf_counter()
analysis.fitter.run_minimizer(
    log_func_minimize=analysis.fitter.minus_log_prob,
    p0=initial_point,
    restart=True,
    neval=20 if test else 1000,
)
nm_seconds = perf_counter() - start
nm_chi2 = analysis.fitter.mle_chi2
print(f"Nelder--Mead chi2: {nm_chi2:.3f} ({nm_seconds:.1f} s)")

# %% [markdown]
# ## PSO with and without batched likelihood calls
#
# Each PSO iteration evaluates every particle.  With ``vectorize=True`` those
# particles are passed to one batched likelihood call; with ``False`` the same
# particles are evaluated one by one. Both runs reuse the already-loaded model,
# start from ``initial_point``, and use the same deterministic swarm setup.

# %%
def run_pso(vectorize, chi2_tol=0.1, pso_type="global"):
    start = perf_counter()
    # ``initial_point`` is reused unchanged for each independent PSO run.
    analysis.fitter.run_minimizer_pso(
        p0=initial_point,
        n_particles=100 if test else 300,
        iters=10 if test else 100,
        vectorize=vectorize,
        restart=True,
        chi2_tol=chi2_tol,
        pso_type=pso_type,
    )
    return analysis.fitter.mle_chi2, analysis.fitter.mle_cube.copy(), perf_counter() - start


pso_scalar_chi2, _, pso_scalar_seconds = run_pso(vectorize=False)
print(f"PSO scalar chi2: {pso_scalar_chi2:.3f} ({pso_scalar_seconds:.1f} s)")

pso_batched_chi2, pso_batched_point, pso_batched_seconds = run_pso(vectorize=True)
print(f"PSO batched chi2: {pso_batched_chi2:.3f} ({pso_batched_seconds:.1f} s)")

# %% [markdown]
# ## emcee with and without batched likelihood calls
#
# Both chains reuse the same likelihood and start from the same PSO point.
# Resetting the fitter random generator gives them identical initial walkers.
# ``vectorize=False`` calls the scalar likelihood per walker; ``vectorize=True``
# evaluates walker groups together.

# %%
def run_sampler(vectorize):
    analysis.fitter.rng = np.random.default_rng(1234)
    start = perf_counter()
    analysis.fitter.run_sampler(pini=pso_batched_point, vectorize=vectorize)
    seconds = perf_counter() - start
    best_chi2 = -2.0 * np.max(analysis.fitter.lnprob)
    return best_chi2, seconds


sampler_scalar_chi2, sampler_scalar_seconds = run_sampler(vectorize=False)
sampler_batched_chi2, sampler_batched_seconds = run_sampler(vectorize=True)

# %% [markdown]
# ## Compare timings
#
# The numerical trajectories are not expected to match exactly between scalar
# and batched emcee, but both use the same likelihood. Timings depend on the
# emulator, batch size, CPU, and the small test budget; repeat with production
# settings before using them for resource planning.

# %%
summary = pd.DataFrame(
    {
        "method": [
            "Nelder--Mead (scalar)",
            "PSO (scalar)",
            "PSO (batched)",
            "emcee (scalar)",
            "emcee (batched)",
        ],
        "seconds": [
            nm_seconds,
            pso_scalar_seconds,
            pso_batched_seconds,
            sampler_scalar_seconds,
            sampler_batched_seconds,
        ],
        "best chi2": [
            nm_chi2,
            pso_scalar_chi2,
            pso_batched_chi2,
            sampler_scalar_chi2,
            sampler_batched_chi2,
        ],
    }
)
summary["speedup relative to scalar counterpart"] = [
    np.nan,
    1.0,
    pso_scalar_seconds / pso_batched_seconds,
    1.0,
    sampler_scalar_seconds / sampler_batched_seconds,
]
summary

# %% [markdown]
# For a scientific run, first obtain a converged minimizer or PSO solution,
# then set ``test = False`` and choose appropriate PSO and MCMC budgets. Keep
# ``vectorize=True`` for PSO and emcee unless profiling the selected emulator
# and hardware shows otherwise.

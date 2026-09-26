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
#     display_name: cup1d
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
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cup1d import Analysis, Args

# %% [markdown]
# ## Build test-sized baseline analyses
#
# The baseline YAML still supplies the data, theory, priors, and emulator.  We
# only replace the sampler runtime settings. ``Fitter`` enforces the minimum
# ensemble size required by ``emcee``.

# %%
test = True


def make_baseline_analysis(label):
    """Create an independent baseline analysis with a short sampler budget."""
    args = Args.from_baseline(verbose=False)
    args.out_folder = str(Path("/tmp") / f"cup1d_inference_methods_{label}")
    args.mcmc = dict(args.mcmc)
    args.mcmc.update(
        {
            "parallel": False,
            "seed": 1234,
            "n_burn_in": 0,
            "n_steps": 2 if test else 2000,
            "thin": 1,
            # Fitter raises this to 2 * ndim + 1 when necessary.
            "n_walkers": 1 if test else args.mcmc["n_walkers"],
        }
    )
    return Analysis(args)


reference = make_baseline_analysis("reference")
initial_point = reference.fitter.sampling_point_from_parameters().copy()
print(f"Free parameters: {reference.fitter.ndim}")
print(f"Initial chi2: {reference.fitter.get_chi2(initial_point):.3f}")

# %% [markdown]
# ## Local minimization (scalar)
#
# Nelder--Mead proposes a single point at every function evaluation; it has no
# batched counterpart.  The small evaluation budget is only for this tutorial.

# %%
nm_analysis = make_baseline_analysis("nelder_mead")
start = perf_counter()
nm_analysis.fitter.run_minimizer(
    log_func_minimize=nm_analysis.fitter.minus_log_prob,
    p0=initial_point,
    restart=True,
    neval=20 if test else 1000,
)
nm_seconds = perf_counter() - start
print(f"Nelder--Mead chi2: {nm_analysis.fitter.mle_chi2:.3f} ({nm_seconds:.1f} s)")

# %% [markdown]
# ## PSO with and without batched likelihood calls
#
# Each PSO iteration evaluates every particle.  With ``vectorize=True`` those
# particles are passed to one batched likelihood call; with ``False`` the same
# particles are evaluated one by one.  Both runs use the same deterministic
# swarm initialization.

# %%
def run_pso(vectorize):
    label = "pso_batched" if vectorize else "pso_scalar"
    analysis = make_baseline_analysis(label)
    start = perf_counter()
    analysis.fitter.run_minimizer_pso(
        p0=initial_point,
        n_particles=8 if test else 32,
        iters=3 if test else 100,
        vectorize=vectorize,
    )
    return analysis, perf_counter() - start


pso_scalar, pso_scalar_seconds = run_pso(vectorize=False)
pso_batched, pso_batched_seconds = run_pso(vectorize=True)

# %% [markdown]
# ## emcee with and without batched likelihood calls
#
# Both chains start from the same PSO solution. ``vectorize=False`` calls the
# scalar likelihood for each walker; ``vectorize=True`` receives walker groups
# from emcee and evaluates them together.

# %%
def run_sampler(vectorize):
    label = "sampler_batched" if vectorize else "sampler_scalar"
    analysis = make_baseline_analysis(label)
    start = perf_counter()
    analysis.run_sampler(pini=pso_batched.fitter.mle_cube, vectorize=vectorize)
    return analysis, perf_counter() - start


sampler_scalar, sampler_scalar_seconds = run_sampler(vectorize=False)
sampler_batched, sampler_batched_seconds = run_sampler(vectorize=True)

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
            nm_analysis.fitter.mle_chi2,
            pso_scalar.fitter.mle_chi2,
            pso_batched.fitter.mle_chi2,
            -2.0 * np.max(sampler_scalar.fitter.lnprob),
            -2.0 * np.max(sampler_batched.fitter.lnprob),
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

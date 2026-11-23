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
# # ForestFlow fiducial linear-power cache: accuracy and speed
#
# With a fixed background only `As`, `ns`, `nrun` vary, so LaCE represents the
# cosmology as a `RescaledCosmology` (fiducial linear power times an analytic
# primordial scaling) and the P3D→P1D quadrature grid is fixed. With
# `cache_fiducial_linP=True` (default) ForestFlow caches the fiducial linear
# power on that grid and only multiplies by the scaling on each call.
#
# Here we set up the DESI DR1 + Karaçaylı et al. (2022) analysis with
# `forest_mpg_gp` and evaluate the likelihood at the same 20 points first with
# the cache and then without it, comparing χ² and timing.

# %%
# %load_ext autoreload
# %autoreload 2

import os
import time

import numpy as np

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo

# %% [markdown]
# ## Set up the analysis

# %%
config_file = os.path.join(
    get_path_repo("cup1d"), "configs", "hires", "hires_forestflow.yaml"
)
args = Args.from_yaml(config_file, verbose=False)
analysis = Analysis(args)
linear_theory = analysis.emulator.model_Arinyo.linear

# %% [markdown]
# ## Evaluation points
#
# The starting point plus 19 points displaced by up to ±0.1 of the prior range
# in every free parameter (unit-cube coordinates).

# %%
n_points = 20
rng = np.random.default_rng(0)
cube0 = analysis.fitter.sampling_point_from_parameters(
    analysis.fitter.initial_parameters()
)
cubes = [cube0] + [
    np.clip(cube0 + rng.uniform(-0.1, 0.1, size=cube0.size), 0.01, 0.99)
    for _ in range(n_points - 1)
]


def evaluate(use_cache):
    """Return chi2 and wall time (s) of every point with/without the cache."""
    linear_theory.cache_fiducial_linP = use_cache
    linear_theory.clear_fiducial_linP_cache()
    chi2 = np.zeros(n_points)
    times = np.zeros(n_points)
    for ii, cube in enumerate(cubes):
        start = time.perf_counter()
        chi2[ii] = analysis.fitter.get_chi2(cube)
        times[ii] = time.perf_counter() - start
    return chi2, times


# %% [markdown]
# ## With the cache

# %%
chi2_cache, time_cache = evaluate(use_cache=True)

# %% [markdown]
# ## Without the cache

# %%
chi2_nocache, time_nocache = evaluate(use_cache=False)
linear_theory.cache_fiducial_linP = True

# %% [markdown]
# ## Comparison
#
# The first call with the cache fills it, so timings exclude point 0.

# %%
for ii in range(n_points):
    print(
        f"point {ii:2d}: chi2 cache={chi2_cache[ii]:11.4f}  "
        f"no cache={chi2_nocache[ii]:11.4f}  "
        f"diff={chi2_cache[ii] - chi2_nocache[ii]:+.2e}"
    )
print(f"\nmax |Delta chi2| = {np.max(np.abs(chi2_cache - chi2_nocache)):.2e}")
t_cache = 1e3 * np.median(time_cache[1:])
t_nocache = 1e3 * np.median(time_nocache[1:])
print(
    f"median time per likelihood call: cache {t_cache:.0f} ms, "
    f"no cache {t_nocache:.0f} ms, speed-up {t_nocache / t_cache:.2f}x"
)

# %%

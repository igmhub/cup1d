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
# # Nested sampling with Nautilus
#
# `Analysis.run_sampler_nautilus` runs the
# [Nautilus](https://nautilus-sampler.readthedocs.io) nested sampler on the
# same posterior as emcee: flat priors (identity on the unit cube) and
# `like.log_prob` with blobs. Results are kept in Nautilus' native form,
# weighted points, and saved as `nautilus_*.npy` in the output folder.
#
# Requires `pip install nautilus-sampler h5py` (the PyPI package `nautilus`
# is an unrelated library with the same import name).

# %%
# %load_ext autoreload
# %autoreload 2

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from cup1d import Analysis, Args
from cup1d.inference.fitter import default_n_threads
from cup1d.utils.utils import get_path_repo

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ## Set up the analysis
#
# Any YAML works; ICs are read automatically when available.

# %%
# config_file = os.path.join(get_path_repo("cup1d"), "configs", "hires", "hires.yaml")
config_file = os.path.join(get_path_repo("cup1d"), "configs", "hires", "hires_forestflow.yaml")
# config_file = os.path.join(get_path_repo("cup1d"), "configs", "cm2026", "cm2026_base.yaml")

args = Args.from_yaml(config_file, verbose=False)
analysis = Analysis(args)
print("free parameters:", analysis.fitter.ndim)
print("output folder:", analysis.fitter.save_directory)

# %% [markdown]
# Optional: a quick minimization first, to compare its best fit with the
# best sampled point.

# %%
# # %%time
# free_params = analysis.fitter.initial_parameters()
# analysis.run_minimizer(free_params)

# %% [markdown]
# ## Run Nautilus
#
# Settings for a quick test; production runs typically use the defaults
# (`n_live=2000`, `n_eff=10000`) or more live points.
#
# * `pool`: number of worker processes (leave 2 cores free), or `None`.
# * `vectorize`: batched likelihood; default true except for ForestFlow.
# * `filepath`: checkpoint, by default `nautilus.hdf5` in the output folder;
#   pass an existing file to resume a run.

# %%
tighten prior??
f_live < 0.01 (exploration)
N_eff > n_eff (sampling)

# %%
# %%time
sampler = analysis.run_sampler_nautilus(
    n_live=1000,
    n_eff=5000,
    # n_eff=2,
    # pool=None,
    pool=default_n_threads(),
    # run_kwargs={"n_like_max": 3000},
    seed=0,
)

# %%
res = analysis.fitter.nautilus_results
weights = np.exp(res["log_w"] - res["log_w"].max())
weights /= weights.sum()
print(f"log Z = {res['log_z']:.3f}")
print(f"n_eff = {res['n_eff']:.0f}, likelihood calls = {res['n_like']}")
print(f"weighted points stored: {len(weights)}")

# %% [markdown]
# ## Parameter summaries
#
# Weighted mean and standard deviation in physical units, flagging
# posteriors that pile up near a flat prior bound.

# %%
fitter = analysis.fitter
physical = np.zeros_like(res["points"])
for ip, name in enumerate(fitter.like.free_param_names):
    physical[:, ip] = fitter.value_from_cube(name, res["points"][:, ip])

mean = weights @ physical
std = np.sqrt(weights @ (physical - mean) ** 2)
cube_mean = weights @ res["points"]
for ip, name in enumerate(fitter.like.free_param_names):
    flag = "  <-- near prior bound" if (cube_mean[ip] < 0.05 or cube_mean[ip] > 0.95) else ""
    print(f"{name:20s} {mean[ip]: .4e} +- {std[ip]:.2e}{flag}")

# %% [markdown]
# ## Compressed cosmology (blinded blobs)

# %%
blob_names = fitter.blob_names
blobs = res["blobs"]
i_d2 = blob_names.index("Delta2_star")
i_ns = blob_names.index("n_star")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].hist(blobs[:, i_d2], bins=50, weights=weights, histtype="step")
ax[0].set_xlabel(r"$\Delta^2_\star$")
ax[1].hist(blobs[:, i_ns], bins=50, weights=weights, histtype="step")
ax[1].set_xlabel(r"$n_\star$")
ax[2].hist2d(blobs[:, i_d2], blobs[:, i_ns], bins=50, weights=weights, cmap="Blues")
ax[2].set_xlabel(r"$\Delta^2_\star$")
ax[2].set_ylabel(r"$n_\star$")
fig.tight_layout()

# %% [markdown]
# ## Corner plot with getdist (weighted samples)

# %%
from getdist import MCSamples, plots

samples = MCSamples(
    samples=np.column_stack([blobs[:, i_d2], blobs[:, i_ns]]),
    weights=weights,
    names=["Delta2_star", "n_star"],
    labels=[r"\Delta^2_\star", r"n_\star"],
)
g = plots.get_subplot_plotter()
g.triangle_plot([samples], filled=True)

# %% [markdown]
# ## Reload saved results
#
# The manifest and arrays written by `save_nautilus_results`.

# %%
manifest = np.load(
    os.path.join(fitter.save_directory, "nautilus_results.npy"), allow_pickle=True
).item()
print({key: manifest[key] for key in ("log_z", "n_eff", "n_like", "blob_names")})
points = np.load(manifest["points_path"])
log_w = np.load(manifest["log_w_path"])
print(points.shape, log_w.shape)

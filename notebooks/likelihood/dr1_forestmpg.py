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
# # Tutorial for DR1 data with Forest-MPG
#
# This is the DR1 tutorial using the ForestFlow MPG emulator.  It is a
# like-for-like counterpart of ``notebooks/tutorials/dr1.py``: the only
# likelihood configuration change is ``emulator_label: forest_mpg``.

# %%
# %load_ext autoreload
# %autoreload 2

from copy import deepcopy

import numpy as np
import os
import matplotlib.pyplot as plt

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Load P1D measurements and set likelihood

# %%
config_file = os.path.join(
    get_path_repo("cup1d"), "configs", "cm2026", "cm2026_base_forestmpg.yaml"
)
args = Args.from_yaml(config_file, verbose=False)
analysis = Analysis(args)

# %% [markdown]
# ## Plot P1D data
#
# Get parameters from a point of the parameter space close to the best fit.

# %%
initial_point = analysis.fitter.initial_parameters()
analysis.fitter.get_chi2(initial_point)

# %% [markdown]
# Plot model for these parameters.

# %%
analysis.like.plot_p1d(initial_point)

# %% [markdown]
# ### Get predictions from the model

# %%
# List of model parameters.
for par, parameter in initial_point.items():
    print(
        par,
        f"{parameter['value']:.4e}",
        f"{parameter['min_value']:.4e}",
        f"{parameter['max_value']:.4e}",
    )

# %% [markdown]
# ## Run minimizer

# %% [markdown]
# Get a point close to the best fit again.

# %%
initial_point = analysis.fitter.initial_parameters()
analysis.fitter.get_chi2(initial_point)

# %% [markdown]
# Run the minimizer starting from this point.  For a production fit, set the
# desired minimizer settings in the YAML or on ``args`` before constructing the
# analysis.

# %%
analysis.run_minimizer(initial_point)

# %% [markdown]
# Evaluate the new best fit.

# %%
best_fit_point = analysis.fitter.mle
analysis.like.plot_p1d(best_fit_point)

# %%
analysis.fitter.save_directory

# %%
from pathlib import Path

minimizer_results = Path(analysis.fitter.save_directory) / "minimizer_results.npy"
print(minimizer_results)

# %% [markdown]
# ## Read chain
#
# Set this path to the chain produced by a Forest-MPG sampling run.

# %%
chain_path = "PATH_TO_FOREST_MPG_BLOBS.npy"
# base_chain = np.load(chain_path)
# results = {
#     "Delta2_star": base_chain["Delta2_star"].reshape(-1),
#     "n_star": base_chain["n_star"].reshape(-1),
# }
# for par, values in results.items():
#     print(par, np.median(values))

# %% [markdown]
# ## Apply unblinding and compare sampler with minimizer
#
# The following cells become applicable after a Forest-MPG chain is available.

# %%
# from cup1d.utils.blinding import apply_unblinding
#
# results_unblind = apply_unblinding(analysis.like.blind, results)
# analysis.fitter.estimate_mle_errors(method="gauss_newton")
# mle_cosmo_unblind = apply_unblinding(
#     analysis.like.blind, analysis.fitter.mle_cosmo.copy()
# )
# from cup1d.postprocessing import plot_cosmo_sampler_and_fit
#
# samples = np.column_stack(
#     [results_unblind["Delta2_star"], results_unblind["n_star"]]
# )
# fig = plot_cosmo_sampler_and_fit(
#     samples,
#     mle_cosmo_unblind,
#     analysis.fitter.mle_cosmo_covariance[:2, :2],
#     sampler_label="Sampler contours",
#     fit_label="Minimizer",
# )
# plt.show()

# %% [markdown]
# ## Reload the saved minimizer result
#
# The YAML configuration rebuilds the Forest-MPG likelihood, after which the
# saved fit state is restored. Loading does not create another output directory.

# %%
# minimizer_results = os.path.join(
#     analysis.fitter.save_directory, "minimizer_results.npy"
# )
# restored_analysis = Analysis.from_results(minimizer_results)
# print(restored_analysis.fitter.mle_chi2)
# print(restored_analysis.fitter.mle)

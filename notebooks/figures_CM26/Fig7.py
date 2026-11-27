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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fig. 7
#
# Contribution of different contaminants

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np

# our own modules
from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.fit.illustrate import (
    plot_contaminant_residuals, residual_scenarios, residual_scenarios_data,
)
from cup1d.postprocessing.io import save_figure_data
from cup1d.utils.utils import get_path_repo

# %% [markdown]
# Local (one redshift at a time) fits to DESI DR1 QMLE3 with lace-mpg, fixed
# cosmology and without inflating the statistical errors (the
# `at_a_time_global` setup with the `no_inflate` covariance of the paper).

# %%
config_path = (Path(get_path_repo("cup1d")) / "configs" / "cm2026" / "variations"
               / "at_a_time_global_QMLE3.yaml")
args = Args.from_yaml(config_path, verbose=False)
pip = Analysis(args, create_output=False)
data = next(iter(pip.like.data.values()))

# %%
npoints = np.array([len(k) for k in data.k_kms])
npoints

# %% [markdown]
# Fit the first redshift bin (z = 2.2)

# %%
iz = 0
zmask = np.array([data.z[iz]])
p0 = pip.fitter.initial_parameters()
pip.fitter.run_minimizer(log_func_minimize=pip.fitter.minus_log_prob, p0=p0, zmask=zmask, restart=True)
print("chi2 at z =", zmask[0], ":", pip.fitter.get_chi2(pip.fitter.mle_cube, zmask=zmask))
best = pip.fitter.parameters_from_sampling_point(pip.fitter.mle_cube)

# %% [markdown]
# Residuals after switching off each contaminant term

# %%
scenarios = residual_scenarios(pip.like, best, zmask[0])
for name, chi2 in scenarios["chi2"].items():
    print(f"{name:12s} chi2 = {chi2:7.1f}, delta chi2 = {chi2 - scenarios['chi2']['all']:6.1f}")

# %%
fig = plot_contaminant_residuals(scenarios, fontsize=22)
# fig.savefig("figs/cont_illustrate_each2.2.pdf")

# %%
save_figure_data(residual_scenarios_data(scenarios), "fig_7.npy")

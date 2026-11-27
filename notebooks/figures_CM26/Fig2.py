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
# # Figure 2
#
# Diagonal terms of the covariance matrix for DESI DR1 analysis

# %%
# %load_ext autoreload
# %autoreload 2

from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.io import save_figure_data
from cup1d.postprocessing.p1d.covariance import (
    error_budget_data, p1d_error_budget, plot_error_budget, plot_p1d_correlation,
)

# %%
args = Args.from_baseline()
pip = Analysis(args, create_output=False)

# %%
budget = p1d_error_budget(pip.fitter.like, use_pk_smooth=False)
fig = plot_error_budget(budget)

# %%
save_figure_data(error_budget_data(budget), "fig_2.npy")

# %% [markdown]
# ### Plot correlation matrix

# %%
fig = plot_p1d_correlation(pip.fitter.like)

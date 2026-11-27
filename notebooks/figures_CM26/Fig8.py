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
# # Fig. 8
#
# One redshift at a time fits

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np

# our own modules
from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.io import save_figure_data
from cup1d.postprocessing.p1d import P1DPlotter, p1d_figure_data, plot_p1d_bins
from cup1d.utils.utils import get_path_repo

# %% [markdown]
# One free node per redshift for every IGM and contaminant parameter, starting
# from the one-redshift-at-a-time fits (fixed cosmology, no inflation of the
# statistical errors): the `global_all` setup of the paper.

# %%
config_path = (Path(get_path_repo("cup1d")) / "configs" / "cm2026" / "variations"
               / "global_all_QMLE3.yaml")
args = Args.from_yaml(config_path, verbose=False)
pip = Analysis(args, create_output=False)

# %%
p0 = pip.fitter.initial_parameters()
free_params = p0
pip.fitter.like.get_chi2(free_params, zmask=[2.2])

# %%
# Fig. 8

n_param_glob_full = 17

ylims = np.array([
    [0.85, 1.15],
    [0.85, 1.15],
    [0.7, 1.3],
    [0., 2.0],
])

p1d_plotter = P1DPlotter(pip.fitter.like)
bins, chi2, dof = p1d_plotter.prepare(
    free_params, plot_panels=True, glob_full=True, n_param_glob_full=n_param_glob_full,
    chi2_nozcov=True,
)
fig = plot_p1d_bins(bins, chi2, dof, residuals=True, panels=True, ndata=p1d_plotter.ndata,
                    print_chi2=True, ylims=ylims)
# fig.savefig("figs/residual_full_global.pdf")

# %%
save_figure_data(p1d_figure_data(bins, residuals=True), "fig_8.npy")

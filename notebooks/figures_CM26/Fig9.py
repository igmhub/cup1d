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
# # Figure 9
#
# Comparison between DESI DR1 data and best-fitting model to it

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np

from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.chains.products import load_parameter_bounds
from cup1d.postprocessing.io import save_figure_data
from cup1d.postprocessing.literature import dr1_chain_directory
from cup1d.postprocessing.p1d import P1DPlotter, p1d_figure_data, plot_p1d_bins

# %%
args = Args.from_baseline()
pip = Analysis(args, create_output=False)

# %% [markdown]
# The best fit of the paper chain is stored in unit-cube coordinates of the
# chain's priors (the `ns` prior changed since), so it is converted with them.

# %%
# my local machine
folder = dr1_chain_directory()
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"

data = np.load(folder / "fitter_results.npy", allow_pickle=True).item()
names, bounds = load_parameter_bounds(folder)
free_params = {name: bounds[name][0] + (bounds[name][1] - bounds[name][0]) * cube
               for name, cube in zip(names, data["fitter"]["mle_cube"])}
print("chi2 saved with the chain:", -2 * data["fitter"]["lnprob_mle"])
print("chi2 now:", pip.fitter.like.get_chi2(free_params))

# %%
ylims = np.array([
    [0.85, 1.15],
    [0.85, 1.15],
    [0.7, 1.3],
    [0., 2.0],
])

p1d_plotter = P1DPlotter(pip.fitter.like)
bins, chi2, dof = p1d_plotter.prepare(free_params, plot_panels=True)
fig = plot_p1d_bins(bins, chi2, dof, residuals=True, panels=True, ndata=p1d_plotter.ndata,
                    print_chi2=False, ylims=ylims)
# fig.savefig("figs/residual_fid_opt_global.pdf")

# %%
save_figure_data(p1d_figure_data(bins, residuals=True), "fig_9.npy")

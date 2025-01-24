# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Results of challenge

# pair notebook
#
# jupytext --set-formats ipynb,py notebook.ipynb

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import glob
import matplotlib.pyplot as plt
from cup1d.likelihood.plotter import Plotter
from corner import corner
# -

path_out_challenge = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/v9fx/"

emulator = "Nyx_alphap_cov"

# ## Fiducial

# Find files

search = os.path.join(path_out_challenge, emulator, "*noise-42-*fiducial*")
files = np.sort(glob.glob(search))
# for ii in range(len(files)):
    # print(ii, files[ii])

# +

cosmo_keys = ['$\\Delta^2_\\star$', '$n_\\star$', '$\\alpha_\\star$']
best_fit = np.zeros((len(files), len(cosmo_keys)))
truth = np.zeros(len(cosmo_keys))

for ii in range(len(files)):
    file = os.path.join(files[ii], "chain_1", "fitter_results.npy")
    data = np.load(file, allow_pickle=True).item()
    
    for jj, key in enumerate(cosmo_keys):
        if ii == 0:
            truth[jj] = data["truth"][key]            
        best_fit[ii, jj] = data["fitter"]["mle"][key]

# +

plotter.plot_corner(only_cosmo=True, only_cosmo_lims=False, extra_data=best_fit)
# -

fname_chain = os.path.join(path_out_challenge, emulator, "mockchallenge-0.9fx_nonoise_fiducial/chain_1/fitter_results.npy")
plotter = Plotter(save_directory="test", fname_chain=fname_chain)



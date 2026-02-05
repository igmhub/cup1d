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

# # Figure 20c
#
# Best-fitting HCD model

# +
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import cup1d
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
# -

args = Args(pre_defined="CM2026", system="local")
pip = Pipeline(args, out_folder=None)

# +
# my local machine
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"

data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]

chain = np.load(folder + "chain.npy")
# -

# Fig. 20c
out_data = pip.fitter.like.plot_hcd_cont(p0=p0, chain=chain, save_directory=None, store_data=True)

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20c.npy")
np.save(fname, out_data)
# -



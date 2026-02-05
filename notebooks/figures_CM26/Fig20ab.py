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

# # Figure 20ab
#
# Best-fitting metal models

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

chain = np.load(folder + "chain.npy")
# -

# Fig. 20a
out_data = pip.fitter.like.plot_metal_cont_mult(chain=chain, save_directory=None, store_data=True)

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20a.npy")
np.save(fname, out_data)
# -

# Fig. 20b
out_data = pip.fitter.like.plot_metal_cont_add(free_params=free_params, chain=chain, save_directory=None, store_data=True)

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20b.npy")
np.save(fname, out_data)

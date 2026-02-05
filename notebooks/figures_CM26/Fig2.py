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

# # Figure 2
#
# Diagonal terms of the covariance matrix for DESI DR1 analysis

# +
# %load_ext autoreload
# %autoreload 2

import os
import cup1d
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
# -

args = Args(pre_defined="CM2026", system="local")
# pip = Pipeline(args, out_folder=args.out_folder)
pip = Pipeline(args, out_folder=None)

# store_data = pip.fitter.like.data.plot_p1d(store_data=True, fname=None)
store_data = pip.fitter.like.plot_cov_to_pk(use_pk_smooth=False, store_data=True)

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_2.npy")
np.save(fname, store_data)

# ### Plot correlation matrix

pip.fitter.like.plot_correlation_matrix()

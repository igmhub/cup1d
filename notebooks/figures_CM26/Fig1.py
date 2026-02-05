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

# # Figure 1
#
# P1D from DESI DR1 data

# +
# %load_ext autoreload
# %autoreload 2

import os
import cup1d
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
# -

args = Args(pre_defined="CM2026", system="local")
pip = Pipeline(args, out_folder=None)

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_1.npy")
np.save(fname, store_data)



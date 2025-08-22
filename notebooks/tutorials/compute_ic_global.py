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

# # Compute IC from global fit

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo


# +
emu = "nyx"

args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="global_all", fix_cosmo=True)
pip = Pipeline(args, out_folder=None)
# -

p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

pip.fitter.like.plot_p1d(residuals=True, plot_panels=True)

pip.run_minimizer(p0, restart=True)

fname = emu + "_ic_global_orig.npy"
pip.save_global_ic(fname)



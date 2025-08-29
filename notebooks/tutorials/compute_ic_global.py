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
# emu = "nyx"
emu = "mpg"

data_label = "DESIY1_QMLE3"

args = Args(data_label=data_label, emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="global_all", fix_cosmo=True, P1D_type=data_label)
pip = Pipeline(args, out_folder=None)
# -

p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

pip.fitter.like.plot_p1d(residuals=True, plot_panels=True, glob_full=True, fontsize=18, plot_fname="residual_full_global")

pip.run_minimizer(p0, restart=True)

fname = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", emu + "_ic_global_orig.npy"
)
pip.save_global_ic(fname)



# ### For reduced

# +
emu = "nyx"
# emu = "mpg"

data_label = "DESIY1_QMLE3"

args = Args(data_label=data_label, emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="global_opt", fix_cosmo=True, P1D_type=data_label)
args.file_ic = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", emu + "_ic_global_orig.npy"
)
pip = Pipeline(args, out_folder=None)
# -

p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

pip.run_minimizer(p0, restart=True)

fname = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", emu + "_ic_global_red.npy"
)
pip.save_global_ic(fname)



# ### Check all good

# +
# emu = "nyx"
emu = "mpg"

data_label = "DESIY1_QMLE3"

args = Args(data_label=data_label, emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="global_opt", fix_cosmo=True, P1D_type=data_label)
pip = Pipeline(args, out_folder=None)
p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)
# -

pip.fitter.like.plot_p1d(residuals=True, plot_panels=True)

pip.fitter.like.plot_igm(cloud=True)

# +
# pip.run_minimizer(p0, restart=True)
# -



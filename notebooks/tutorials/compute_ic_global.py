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
data_label = "DESIY1_QMLE3"
name_variation = None
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"
# name_variation = "no_inflate"

args = Args(data_label=data_label, emulator_label=emulator_label)
args.set_baseline(
    fit_type="global_all", 
    fix_cosmo=True, 
    P1D_type=data_label, 
    name_variation=name_variation, 
)

pip = Pipeline(args, out_folder=None)
# -

p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0, zmask=[2.2])
# pip.fitter.like.get_chi2(p0)

# +

pip.fitter.like.get_chi2(p0)

# +
n_param_glob_full = 15 # nparams each z, cheeeeeeck!!!!

pname = None
# pname = "figs/residual_full_global"
pip.fitter.like.plot_p1d(
    p0,
    residuals=True,
    plot_panels=True,
    glob_full=True,
    n_param_glob_full=n_param_glob_full,
    fontsize=18,
    chi2_nozcov=True,
    plot_fname=pname,
)
# -

176 - 53

# ### IC for reduced

# +
data_label = "DESIY1_QMLE3"
name_variation = None
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"
name_variation = "no_inflate"

args = Args(data_label=data_label, emulator_label=emulator_label)
args.set_baseline(
    fit_type="global_opt", 
    fix_cosmo=True, 
    P1D_type=data_label, 
    name_variation=name_variation,
    ic_global=False
)

pip = Pipeline(args, out_folder=None)
# -

p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

pip.run_minimizer(p0, restart=True)

753.086178953991

p0 = pip.fitter.mle_cube

pname = None
# pname = "figs/residual_full_global"
pip.fitter.like.plot_p1d(
    p0,
    residuals=True,
    plot_panels=True,
    # glob_full=True,
    # n_param_glob_full=16,
    fontsize=18,
    # chi2_nozcov=True,
    plot_fname=pname,
)

fname = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", "mpg_ic_global_red.npy"
)
pip.save_global_ic(fname)



# ### Check all good

# +
data_label = "DESIY1_QMLE3"
name_variation = None

args = Args(data_label=data_label, emulator_label="CH24_mpgcen_gpr")
args.set_baseline(
    fit_type="global_opt", 
    fix_cosmo=True, 
    P1D_type=data_label, 
    name_variation=name_variation, 
)

pip = Pipeline(args, out_folder=None)
p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)
# -

pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True)

pip.fitter.like.plot_igm(cloud=True)

# +
# pip.run_minimizer(p0, restart=True)
# -



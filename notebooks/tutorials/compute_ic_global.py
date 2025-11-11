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

# pip.fitter.like.get_chi2(p0)
17 * 11

# +
n_param_glob_full = 17

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

$z$ & $\chi^2$ & ndeg & prob\ \hline
2.2 & 29.34 & 32 & 60.18 \\
2.4 & 43.03 & 35 & 16.52 \\
2.6 & 51.28 & 38 & 7.35 \\
2.8 & 50.5 & 41 & 14.68 \\
3.0 & 70.71 & 44 & 0.65 \\
3.2 & 53.47 & 46 & 20.93 \\
3.4 & 47.78 & 48 & 48.17 \\
3.6 & 81.37 & 50 & 0.33 \\
3.8 & 62.14 & 52 & 15.86 \\
4.0 & 81.16 & 53 & 0.77 \\
4.2 & 64.96 & 55 & 16.85 \\
\hline
All & 635.75 & 494 & 0.0 \\ \hline
Prob 0.0015954647003453976

# ### IC for reduced

# +
data_label = "DESIY1_QMLE3"
name_variation = None
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"
# name_variation = "no_inflate"

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

mpg 753.086178953991
nyx 643.902238250646

p0 = pip.fitter.mle_cube

# pname = None
pname = "figs/residual_full_global"
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
    # os.path.dirname(get_path_repo("cup1d")), "data", "ics", "nyx_ic_global_red.npy"
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



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

# # Compute IC from fits at a time

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.likelihood.plotter import Plotter
from cup1d.utils.utils import get_path_repo


# +

data_label = "DESIY1_QMLE3"
name_variation = None
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"

# emu_cov_type = "block"
# emu_cov_type = "diagonal"
emu_cov_type = "full"
# name_variation = "Ma2025"
name_variation = "no_inflate"

args = Args(data_label=data_label, emulator_label=emulator_label, emu_cov_type=emu_cov_type)
args.set_baseline(
    fit_type="at_a_time_global", 
    fix_cosmo=True, 
    P1D_type=data_label, 
    name_variation=name_variation, 
)

pip = Pipeline(args, out_folder=None)
# -

npoints = []
for ii in range(len(pip.fitter.like.data.z)):
    npoints.append(len(pip.fitter.like.data.k_kms[ii]))
npoints = np.array(npoints)
npoints

# ### Do fits

out_mle = []
out_mle_cube = []
out_chi2 = []
out_pnames = []
# for ii in range(len(pip.fitter.like.data.z)):
for ii in range(1):
    zmask = np.array([pip.fitter.like.data.z[ii]])

    pip = Pipeline(args, out_folder=None)
    
    print()
    
    f_space_len = 14
    s_space_len = 5
    for p in pip.fitter.like.free_params:            
        print(
            p.name, (f_space_len-len(p.name)) * " ", "\t", 
            np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
            np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
            np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
            p.Gauss_priors_width
        )

    
    print()
    
    print(ii, zmask)
    p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    pip.fitter.run_minimizer(log_func_minimize=pip.fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True)
    out_pnames.append(pip.fitter.like.free_param_names)
    out_mle.append(pip.fitter.mle)
    out_mle_cube.append(pip.fitter.mle_cube)
    out_chi2.append(pip.fitter.mle_chi2)

# +
# pip.fitter.like.theory.model_cont.metal_models["Si_mult"].fid_vals
# -

pip.fitter.like.get_chi2(pip.fitter.mle_cube, zmask=zmask)

diru = 'figs'
# diru=None
plotter = Plotter(pip.fitter, save_directory=diru, zmask=zmask)

# +

plotter.plot_illustrate_contaminants_cum(out_mle_cube[0].copy(), zmask, fontsize=20)
# +

plotter.plot_illustrate_contaminants_each(out_mle_cube[0].copy(), zmask, fontsize=20)
# -






fname = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", "mpg_ic_at_a_time.npy"
    # os.path.dirname(get_path_repo("cup1d")), "data", "ics", "nyx_ic_at_a_time.npy"
)
dir_out = {
    "z":pip.fitter.like.data.z,
    "pnames":out_pnames,
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
}
np.save(fname, dir_out)

# inflate 5%
from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)

# no inflate
from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)



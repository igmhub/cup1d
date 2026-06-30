# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute IC from fits at a time

# %%
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


# %%
data_label = ["DESIY1_QMLE3"]
name_variation = None
p1d_fname = None


data_label = ["DESIY1_FFT3_dir"]
name_variation = None
p1d_fname = None
# name_variation = "DLA_TAN"
# p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/in_DESI_DR1/ting_tan/p1d_fft_y1_measurement_kms_tingdla_nocrossexp_snr3noweights_directmetalsubtraction.fits"


emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"

# emu_cov_type = "block"
# emu_cov_type = "diagonal"
emu_cov_type = "full"
# name_variation = "Ma2025"
name_variation = "no_inflate"

args = Args(
    data_label=data_label,
    emulator_label=emulator_label,
    emu_cov_type=emu_cov_type,
    p1d_fname=p1d_fname,
)

args.set_baseline(
    fit_type="at_a_time_global",
    fix_cosmo=True,
    P1D_type=data_label,
    name_variation=name_variation,
)

pip = Pipeline(args, out_folder=None)

# %%
key = "DESIY1_FFT3_dir"

npoints = []
for ii in range(len(pip.fitter.like.data[key].z)):
    npoints.append(len(pip.fitter.like.data[key].k_kms[ii]))
npoints = np.array(npoints)
npoints

# %% [markdown]
# ### Do fits

# %%
out_mle = []
out_mle_cube = []
out_chi2 = []
out_pnames = []
# for ii in range(len(pip.fitter.like.data.z)):
for ii in range(1):
    zmask = np.array([pip.fitter.like.data[key].z[ii]])

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
    # p0 = np.array(list(pip.fitter.like.fid["fit_cube"].values()))
    pip.fitter.run_minimizer(log_func_minimize=pip.fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True)
    out_pnames.append(pip.fitter.like.free_param_names)
    out_mle.append(pip.fitter.mle)
    out_mle_cube.append(pip.fitter.mle_cube)
    out_chi2.append(pip.fitter.mle_chi2)

# %%
# pip.fitter.like.theory.model_cont.metal_models["Si_mult"].fid_vals

# %%
p0 = pip.fitter.mle_cube

# %%
pip.fitter.like.get_chi2(pip.fitter.mle_cube, zmask=zmask)

# %%
chi2_z22 = {
    "full": 29.34193443427515,
    "no HCD": 50.562410331579215,
    "no SiII-SiIII": 40.0704405056536,
    "no SiII-SiII": 58.36565506912524,
    "no Lya-SiII": 75.86759701563147,
    "no Lya-SiIII": 661.8034659063873,
    "no cont": 764.4559241510051,
}
for key in chi2_z22:
    print(key, np.round(chi2_z22[key] - chi2_z22["full"], 1))

# %%
# diru = 'figs'
diru=None
plotter = Plotter(pip.fitter, save_directory=diru, zmask=zmask)

# %%
pip.fitter.like.data

# %%

plotter.plot_illustrate_contaminants_cum(out_mle_cube[0].copy(), zmask, fontsize=20)
# %%
store_data = plotter.plot_illustrate_contaminants_each(out_mle_cube[0].copy(), zmask, fontsize=22, store_data=True)
# %%
# store_data_ting = store_data
store_data_orig = store_data

# %%
for ii in range(2):
    if ii == 0:
        store_data = store_data_ting
        lab = "Ting"
        mar = "s"
    else:
        store_data = store_data_orig
        lab = "Corentin"
        mar = "o"

    plt.errorbar(
        store_data["x"],
        store_data["y4_blue"],
        store_data["yerr4_blue"],
        ls=":",
        marker=mar,
        label="Residual w/o HCD term: " + lab,
        alpha=0.8,
    )
    plt.plot(
        store_data["x"], store_data["y4_orange"], label="HCD term: " + lab, alpha=0.8
    )


plt.xlabel("k [s/km]")
plt.ylabel("Residual")

plt.legend()

plt.axhline(ls=":", color="k")

plt.savefig("residual_ting.png")

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_7.npy")
np.save(fname, store_data)

# %%

# %%

# %%
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

# %%
# inflate 5%
from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)

# %%
# no inflate
from cup1d.optimize.show_results import print_results
print_results(pip.fitter.like, out_chi2, out_mle_cube)

# %%

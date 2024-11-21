# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Does the fiducial igm history matters?
#
# This test examines whether choosing a particular fiducial IGM history matters for cup1d constraints.
#
# - These are mocks drawn from emulators
# - The true IGM history is always the same. We assume those from other simulations in the LH.
# - The fiducial and true cosmology are the same.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import glob
import matplotlib.pyplot as plt

# %%
folder_out = "/home/jchaves/Proyectos/projects/lya/data/cup1d/validate_igm/"
# folder_cov = "Chabanier2019"
folder_cov = "DESIY1"
iemu = 1
arr_folder_emu = ["Pedersen23_ext", "Cabayol23+", "Nyx_alphap_cov"]
pars = ['Delta2_star', 'n_star', "alpha_star"]
nIGM = 2

sim_labels = []
if "Nyx" in arr_folder_emu[iemu]:
    for ii in range(14):
        sim_labels.append("nyx_{}".format(ii))
else:
    for ii in range(30):
        sim_labels.append("mpg_{}".format(ii))
nsims = len(sim_labels)

true_star = np.zeros((nsims, 3))
fid_star = np.zeros((nsims, 3))
best_star = np.zeros((nsims, 3))

for ii in range(nsims):
    sim_label = sim_labels[ii]
    # print(sim_label)

    for kk in range(1):
        file = (
            folder_out
            + "/"
            + folder_cov
            + "/"
            + arr_folder_emu[iemu]
            + "/nIGM"
            + str(nIGM)
            + "/"
            + sim_label
            + "/chain_"
            + str(kk + 1)
            + "/minimizer_results.npy"
        )
        res = np.load(file, allow_pickle=True).item()
    #     print(res['lnprob_mle'], res['cosmo_best'])
    # print(res['cosmo_true'])

    for jj in range(3):        
        if (pars[jj] in res['cosmo_best']):
            true_star[ii, jj] = res['cosmo_true'][pars[jj]]
            fid_star[ii, jj] = res['cosmo_fid'][pars[jj]]
            best_star[ii, jj] = res['cosmo_best'][pars[jj]]


# %% [markdown]
# ### Difference between truth and best fit

# %%
sep_x = 0.01
fontsize =16
if "Nyx" in arr_folder_emu[iemu]:
    nax = 3
    fig, ax = plt.subplots(1, 3, figsize=(14, 6)) 
else:
    nax = 1
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax = [ax]
    
for ii in range(nax):
    if ii == 0:
        jj0 = 0
        jj1 = 1
    elif ii == 1:
        jj0 = 0
        jj1 = 2
    elif ii == 2:
        jj0 = 1
        jj1 = 2

    x0 = (best_star[:, jj0]/true_star[:, jj0]-1) * 100
    y0 = (best_star[:, jj1]/true_star[:, jj1]-1) * 100
    ax[ii].scatter(x0, y0, marker="o", color="C0")
    for kk in range(len(sim_labels)):
        ax[ii].annotate(sim_labels[kk], (x0[kk] + sep_x, y0[kk]), fontsize=6)

    ax[ii].axhline(color="black", linestyle=":")
    ax[ii].axvline(color="black", linestyle=":")

if "Nyx" in arr_folder_emu[iemu]:
    ax[0].set_xlabel(r"$\Delta(\Delta^2_\star)$ [%]", fontsize=fontsize)
    ax[0].set_ylabel(r"$\Delta(n_\star)$ [%]", fontsize=fontsize)
    ax[1].set_xlabel(r"$\Delta(\Delta^2_\star)$ [%]", fontsize=fontsize)
    ax[1].set_ylabel(r"$\Delta(\alpha_\star)$ [%]", fontsize=fontsize)
    ax[2].set_xlabel(r"$\Delta(n_\star)$ [%]", fontsize=fontsize)
    ax[2].set_ylabel(r"$\Delta(\alpha_\star)$ [%]", fontsize=fontsize)
    ax[0].set_xlim(-40, 40)
    ax[0].set_ylim(-3, 3)
    ax[1].set_xlim(-40, 40)
    ax[1].set_ylim(-25, 25)
    ax[2].set_xlim(-3, 3)
    ax[2].set_ylim(-25, 25)
else:
    ax[0].set_xlim(-40, 40)
    ax[0].set_ylim(-3, 3)
    ax[0].set_xlabel(r"$\Delta(\Delta^2_\star)$ [%]", fontsize=fontsize)
    ax[0].set_ylabel(r"$\Delta(n_\star)$ [%]", fontsize=fontsize)

plt.tight_layout()
plt.savefig("nigm"+str(nIGM)+"_"+arr_folder_emu[iemu] + ".png")

# %%

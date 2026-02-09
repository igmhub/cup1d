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

# # Figure 3a
#
# Cosmic variance in mpg-gadget

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import lace
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_multi import GPEmulator
from matplotlib.ticker import FormatStrFormatter


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# -

archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

emulator_label = "CH24_mpgcen_gpr"
emulator = GPEmulator(
    emulator_label=emulator_label, 
    train=False,
)

for kk in range(2):
    if kk == 0:
        sim_lab = "mpg_seed"
    else:
        sim_lab = "mpg_central"

    testing_data = archive.get_testing_data(sim_lab)

    _k_Mpc = testing_data[0]['k_Mpc']
    ind = (_k_Mpc < emulator.kmax_Mpc) & (_k_Mpc > 0)
    k_Mpc = _k_Mpc[ind]
    k_fit = k_Mpc/emulator.kmax_Mpc
    jj = 0
    if jj == 0:
        k_Mpc_0 = k_Mpc.copy()
    else:
        if np.allclose(k_Mpc_0, k_Mpc) == False:
            print(jj, k_Mpc)
            
    
    nz = len(testing_data)
    
    p1d_Mpc_sim = np.zeros((nz, k_Mpc.shape[0]))
    p1d_Mpc_emu = np.zeros((nz, k_Mpc.shape[0]))
    p1d_Mpc_sm = np.zeros((nz, k_Mpc.shape[0]))
    zz_full = np.zeros((nz, k_Mpc.shape[0]))
    k_Mpc_full = np.zeros((nz, k_Mpc.shape[0]))
    
    for ii in range(nz):
    
        zz_full[ii] = testing_data[ii]["z"]
        k_Mpc_full[ii] = k_Mpc_0
    
        p1d_Mpc_sim[ii] = testing_data[ii]['p1d_Mpc'][ind]
        norm = np.interp(
            k_Mpc, emulator.input_norm["k_Mpc"], emulator.norm_imF(testing_data[ii]["mF"])
        )
        yfit = np.log(testing_data[ii]["p1d_Mpc"][ind]/norm)
        popt, _ = curve_fit(emulator.func_poly, k_fit, yfit)
        p1d_Mpc_sm[ii] = norm * np.exp(emulator.func_poly(k_fit, *popt))

    if kk == 0:
        zz_full_seed = zz_full.copy()
        p1d_Mpc_sim_seed = p1d_Mpc_sim.copy()
        p1d_Mpc_sm_seed = p1d_Mpc_sm.copy()



# +
store_data = {}

fig, ax = plt.subplots(1, figsize=(8, 6))
ftsize = 24

ztar = 3
ii = np.argwhere(zz_full == ztar)[0,0]
jj = np.argwhere(zz_full_seed == ztar)[0,0] 
    
psm = 0.5 * (p1d_Mpc_sm_seed[jj] + p1d_Mpc_sm[ii])

ax.plot(k_Mpc_full[ii], p1d_Mpc_sim[ii]/psm-1, "C0-", lw=2, label=r"$P_\mathrm{1D}^x=P_\mathrm{1D}^\mathrm{central}$")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sim_seed[jj]/psm-1, "C1-", lw=2, label=r"$P_\mathrm{1D}^x=P_\mathrm{1D}^\mathrm{seed}$")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sm[ii]/psm-1, "C0--", lw=2, label=r"$P_\mathrm{1D}^x=P_\mathrm{1D}^\mathrm{sm,\,central}$")
ax.plot(k_Mpc_full[ii], p1d_Mpc_sm_seed[jj]/psm-1, "C1--", lw=2, label=r"$P_\mathrm{1D}^x=P_\mathrm{1D}^\mathrm{sm,\,seed}$")

store_data["x"] = k_Mpc_full[ii]
store_data["y_blue_solid"] = p1d_Mpc_sim[ii]/psm-1
store_data["y_orange_solid"] = p1d_Mpc_sim_seed[ii]/psm-1
store_data["y_blue_dashed"] = p1d_Mpc_sm[ii]/psm-1
store_data["y_orange_dashed"] = p1d_Mpc_sm_seed[ii]/psm-1


ax.axhline(color="k", linestyle=":")
ax.axhline(0.01, color="k", linestyle="--")
ax.axhline(-0.01, color="k", linestyle="--")


ax.set_ylim(-0.022, 0.028)
# ax.set_ylim(-0.06, 0.1)


ax.set_xscale("log")
ax.set_ylabel(r"$P_\mathrm{1D}^x/P_\mathrm{1D}^\mathrm{sm, average}-1$", fontsize=ftsize)
ax.set_xlabel(r"$k_\parallel\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
ax.tick_params(axis="both", which="major", labelsize=ftsize)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.legend(fontsize=ftsize-2, loc="upper right", ncol=2)
plt.tight_layout()
# plt.savefig("figs/smooth_cen_seed.png")
# plt.savefig("figs/smooth_cen_seed.pdf")

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_3a.npy")
np.save(fname, store_data)

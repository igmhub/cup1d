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

# # Fig B2a
#
# Performance of lace-lyssa in reproducing P1D predictions from lyssa simulations

# +
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_multi import GPEmulator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
archive = nyx_archive.NyxArchive(nyx_version="models_Nyx_Sept2025_include_Nyx_fid_rseed")
central = archive.get_testing_data("nyx_central")
seed = archive.get_testing_data("nyx_seed")
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data = archive.get_training_data(emu_params=emu_params)

train = False
# emulator_label = "CH24_mpg_gpr"
emulator_label = "CH24_nyxcen_gpr"
emulator = GPEmulator(emulator_label=emulator_label, archive=archive, train=train, drop_sim=None)

# +
kmax_Mpc = emulator.kmax_Mpc
kmax_Mpc_use = 4

# sim_test = central.copy()
sim_test = seed.copy()

_k_Mpc = sim_test[0]['k_Mpc']
ind = (_k_Mpc < kmax_Mpc_use) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/kmax_Mpc

nsam = len(sim_test)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))

for ii in range(nsam):

    if ("kF_Mpc" not in sim_test[ii]) | (np.isfinite(sim_test[ii]['kF_Mpc']) == False):
        for jj in range(len(central)):
            if np.abs(sim_test[ii]["z"] - central[jj]["z"]) < 0.05:
                sim_test[ii]['kF_Mpc'] = central[jj]['kF_Mpc']
        if ("kF_Mpc" not in sim_test[ii]) | (np.isfinite(sim_test[ii]['kF_Mpc']) == False):
            continue
    
    p1d_Mpc_sim[ii] = sim_test[ii]['p1d_Mpc'][ind]
    p1d_Mpc_emu[ii] = emulator.emulate_p1d_Mpc(
        sim_test[ii], 
        k_Mpc
    )

    norm = np.interp(
        k_Mpc, emulator.input_norm["k_Mpc"], emulator.norm_imF(sim_test[ii]["mF"])
    )
    yfit = np.log(sim_test[ii]["p1d_Mpc"][ind]/norm)

    popt, _ = curve_fit(emulator.func_poly, k_fit, yfit)
    p1d_Mpc_sm[ii] = norm * np.exp(emulator.func_poly(k_fit, *popt))

# +
store_data = {}

fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 24

for ii in range(0, nsam-2):
    lab = r"$z=$"+str(np.round(sim_test[ii]['z'], 2))
    if p1d_Mpc_emu[ii][0] != 0:
        ax.plot(k_Mpc, p1d_Mpc_sm[ii]/p1d_Mpc_emu[ii]-1, lw=2, label=lab)
        
        store_data["x"] = k_Mpc
        store_data["y"+str(ii)] = p1d_Mpc_sm[ii]/p1d_Mpc_emu[ii]-1

ax.axhline(ls=":", color="k")
ax.axhline(0.01, ls="--", color="k")
ax.axhline(-0.01, ls="--", color="k")
    
plt.legend(loc="upper left")
ax.set_xscale("log")
ax.set_xlabel(r"$k_\parallel\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
# plt.ylim(-0.03, 0.03)
ax.set_xlim(0.08, 4)
ax.set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{smooth}-1$", fontsize=ftsize)

ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.legend(fontsize=ftsize-5, ncol=3)
plt.tight_layout()
# plt.savefig("figs/nyx_seed.pdf")
# plt.savefig("figs/nyx_seed.png")

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_B2a.npy")
np.save(fname, store_data)
# -



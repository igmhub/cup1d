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

# # Fig 3b
#
# Performance of P1D smooth model in reproducing P1D predictions from MP-Gadget simulations

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
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")
central = archive.get_testing_data("mpg_central")
seed = archive.get_testing_data("mpg_seed")
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data = archive.get_training_data(emu_params=emu_params, average="both")

emulator = GPEmulator(emulator_label="CH24_mpgcen_gpr", train=False)

# +
kmax_Mpc = emulator.kmax_Mpc
kmax_Mpc_use = 4

sim_test = training_data.copy()

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
cols = ["C1", "C0", "C0", "C1"]
percents = np.percentile(p1d_Mpc_sim/p1d_Mpc_sm-1, [5, 16, 84, 95], axis=0)

ax.fill_between(k_Mpc, percents[0], percents[-1], label="5-95th percentiles", color=cols[0], alpha=0.4)
ax.fill_between(k_Mpc, percents[1], percents[2], label="16-84th percentiles", color=cols[1], alpha=0.4)

store_data["x"] = k_Mpc
store_data["y"] = percents

ax.axhline(ls=":", color="k")
ax.axhline(0.01, ls="--", color="k")
ax.axhline(-0.01, ls="--", color="k")
    
plt.legend()
ax.set_xscale("log")
ax.set_xlabel(r"$k_\parallel\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
# plt.ylim(-0.03, 0.03)
# ax.set_xlim(0.08, 4)
ax.set_ylabel(r"$P_\mathrm{1D}^\mathrm{sim}/P_\mathrm{1D}^\mathrm{smooth}-1$", fontsize=ftsize)

ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.legend(fontsize=ftsize-2, ncol=1)
plt.tight_layout()
# plt.savefig("figs/mpg_smooth.pdf")
# plt.savefig("figs/mpg_smooth.png")

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_3b.npy")
np.save(fname, store_data)
# -



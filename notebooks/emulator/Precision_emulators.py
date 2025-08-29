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
# # Precision emulators

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.gp_emulator_multi import GPEmulator

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### Load emulator and data

# %%
archive = gadget_archive.GadgetArchive(postproc="Cabayol23")

# %%
central = archive.get_testing_data("mpg_central")
seed = archive.get_testing_data("mpg_seed")

# %%
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_data = archive.get_training_data(emu_params=emu_params, average="both")

# %%
train = False
# emulator_label = "CH24_mpg_gpr"
emulator_label = "CH24_mpgcen_gpr"
emulator = GPEmulator(emulator_label=emulator_label, archive=archive, train=train, drop_sim=None)

# %%

# %% [markdown]
# ## Fig precision of emulator for central

# %%
kmax_Mpc = emulator.kmax_Mpc
kmax_Mpc_use = 4

sim_test = central
sim_test = seed

_k_Mpc = sim_test[0]['k_Mpc']
ind = (_k_Mpc < kmax_Mpc_use) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/kmax_Mpc

nsam = len(sim_test)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))

for ii in range(nsam):

    if "kF_Mpc" not in sim_test[ii]:
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

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20

for ii in range(1, nsam):
    lab = r"$z=$"+str(np.round(sim_test[ii]['z'], 2))
    ax.plot(k_Mpc, p1d_Mpc_sm[ii]/p1d_Mpc_emu[ii]-1, lw=2, label=lab)

ax.axhline(ls=":", color="k")
ax.axhline(0.01, ls="--", color="k")
ax.axhline(-0.01, ls="--", color="k")
    
plt.legend()
ax.set_xscale("log")
ax.set_xlabel(r"$k\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
# plt.ylim(-0.03, 0.03)
ax.set_xlim(0.08, 4)
ax.set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{smooth}-1$", fontsize=ftsize)

ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-4, ncol=3)
plt.tight_layout()
# plt.savefig("mpg_central.pdf")
# plt.savefig("mpg_central.png")
plt.savefig("mpg_seed.pdf")
plt.savefig("mpg_seed.png")
# plt.savefig("mpgcen_seed.pdf")
# plt.savefig("mpgcen_seed.png")

# %%

# %%
kmax_Mpc = emulator.kmax_Mpc
kmax_Mpc_use = 4

sim_test = training_data

_k_Mpc = sim_test[0]['k_Mpc']
ind = (_k_Mpc < kmax_Mpc_use) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/kmax_Mpc

nsam = len(sim_test)
p1d_Mpc_sim = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_emu = np.zeros((nsam, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nsam, k_Mpc.shape[0]))

for ii in range(nsam):

    if "kF_Mpc" not in sim_test[ii]:
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

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20

for per in [5, 16, 84, 95]:
    lab = str(per) + "th percentile"
    ax.plot(k_Mpc, np.percentile(p1d_Mpc_sm/p1d_Mpc_sim-1, per, axis=0), lw=2, label=lab)

ax.axhline(ls=":", color="k")
ax.axhline(0.01, ls="--", color="k")
ax.axhline(-0.01, ls="--", color="k")
    
plt.legend()
ax.set_xscale("log")
ax.set_xlabel(r"$k\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
# plt.ylim(-0.03, 0.03)
ax.set_xlim(0.08, 4)
ax.set_ylabel(r"$P_\mathrm{1D}^\mathrm{smooth}/P_\mathrm{1D}^\mathrm{sim}-1$", fontsize=ftsize)

ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-4, ncol=1)
plt.tight_layout()
plt.savefig("mpg_smooth.pdf")
plt.savefig("mpg_smooth.png")

# %% [markdown]
# ### L1O

# %%
emulator_label = "CH24_mpgcen_gpr" 

testing_data = archive.get_testing_data("mpg_0")
_k_Mpc = testing_data[0]['k_Mpc']
ind = (_k_Mpc < emulator.kmax_Mpc) & (_k_Mpc > 0)
k_Mpc = _k_Mpc[ind]
k_fit = k_Mpc/emulator.kmax_Mpc

zz = np.array(archive.list_sim_redshifts)
nz = len(zz)
nsim = len(archive.list_sim_cube)
p1d_Mpc_emu = np.zeros((nsim, nz, k_Mpc.shape[0]))
p1d_Mpc_sm = np.zeros((nsim, nz, k_Mpc.shape[0]))

for jj, isim in enumerate(archive.list_sim_cube):
    
    testing_data = archive.get_testing_data(isim)
    emulator = GPEmulator(emulator_label=emulator_label, train=False, drop_sim=isim)

    for ii in range(nz):
        if "kF_Mpc" not in testing_data[ii]:
            print(isim, ii)
            continue

        i2 = np.argwhere(np.abs(zz - testing_data[ii]["z"]) < 0.05)[:, 0]

        p1d_Mpc_emu[jj, i2] = emulator.emulate_p1d_Mpc(
            testing_data[ii], 
            k_Mpc
        )
        norm = np.interp(
            k_Mpc, emulator.input_norm["k_Mpc"], emulator.norm_imF(testing_data[ii]["mF"])
        )
        yfit = np.log(testing_data[ii]["p1d_Mpc"][ind]/norm)
        popt, _ = curve_fit(emulator.func_poly, k_fit, yfit)
        p1d_Mpc_sm[jj, i2] = norm * np.exp(emulator.func_poly(k_fit, *popt))


# %%
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20

for per in [5, 16, 84, 95]:
    lab = str(per) + "th percentile"
    ax.plot(k_Mpc, np.percentile(p1d_Mpc_emu/p1d_Mpc_sm-1, per, axis=(0,1)), lw=2, label=lab)

ax.axhline(ls=":", color="k")
ax.axhline(0.01, ls="--", color="k")
ax.axhline(-0.01, ls="--", color="k")
    
plt.legend()
ax.set_xscale("log")
ax.set_xlabel(r"$k\,\left[\mathrm{Mpc}^{-1}\right]$", fontsize=ftsize)
# plt.ylim(-0.03, 0.03)
ax.set_xlim(0.08, 4)
ax.set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{smooth}-1$", fontsize=ftsize)

ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-4, ncol=1)
plt.tight_layout()
plt.savefig("mpg_l1o.pdf")
plt.savefig("mpg_l1o.png")

# %%

# %% [markdown]
# ### Test consistency for overlaping values of mF, just in the boundary

# %%
emu_params = ['Delta2_p', 'n_p', 'mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
test_params = {}
for key in emu_params:
    test_params[key] = central[5][key]

ind = (central[5]["k_Mpc"] < 4.25) & (central[5]["k_Mpc"] > 0)
k_Mpc = central[5]["k_Mpc"][ind]
P1D_test = central[5]["p1d_Mpc"][ind]

# %%

test_params["mF"] = 0.5403
p1d_Mpc1 = emulator.emulate_p1d_Mpc(model=test_params, k_Mpc=k_Mpc)[0]
test_params["mF"] = 0.5404
p1d_Mpc2 = emulator.emulate_p1d_Mpc(model=test_params, k_Mpc=k_Mpc)[0]

# %%
plt.plot(k_Mpc, (p1d_Mpc1/p1d_Mpc2-1)*100)


# %% [markdown]
# ### OLD

# %%

# %%

# %%
# our modules
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import nn_emulator
from lace.emulator import gp_emulator

# %% [markdown]
# ### Load LaCE emulator

# %% [markdown]
# ### Specify cosmological model
#
# cosmo object will wrap a CAMB results object, and offer useful functionality.

# %%
cosmo=camb_cosmo.get_cosmology(H0=67,ns=0.96)

# %% [markdown]
# ### Compute linear power parameters at the redshift of interest

# %%
z=3.0
test_params=fit_linP.get_linP_Mpc_zs(cosmo,zs=[z],kp_Mpc=emulator.archive.kp_Mpc)[0]
for key,value in test_params.items():
    print(key,'=',value)

# %% [markdown]
# ### Specify IGM parameters at the redshift
#
# We need to choose a value of mean flux (mF), thermal broadening scale (sigT_Mpc), TDR slope gamma and filtering length (kF_Mpc).
#
# We will choose values that are well sampled in the archive.

# %%
dz=0.1
zmask=[ (training_data[i]['z']<z+dz) & (training_data[i]['z']>z-dz) for i in range(Na)]

# %%
test_params

# %%
for param in emu_params:
    if param in ['Delta2_p','n_p']: 
        continue
    test_params[param]=np.mean([ training_data[i][param] for i in range(Na) if zmask[i] ])
    print(param+' = {:.3f}'.format(test_params[param]))

# %% [markdown]
# ### Ask emulator to predict P1D

# %%
# specify wavenumbers to emulate (in velocity units)
k_kms=np.logspace(np.log10(0.002),np.log10(0.02),num=20)
# use test cosmology to translate to comoving units
dkms_dMpc=camb_cosmo.dkms_dMpc(cosmo,z)
print('1 Mpc = {:.2f} km/s at z = {}'.format(dkms_dMpc,z))
k_Mpc=k_kms*dkms_dMpc

# %%
# emulate P1D in comoving units
p1d_Mpc=emulator.emulate_p1d_Mpc(model=test_params,k_Mpc=k_Mpc)
# use test cosmology to translate back to velocity units
p1d_kms=p1d_Mpc*dkms_dMpc

# %%
plt.loglog(k_kms,k_kms*p1d_kms)
plt.xlabel('k [s/km]')
plt.ylabel('k P(k)')

# %%

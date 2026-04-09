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

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt


from cup1d.likelihood.cosmologies import set_cosmo
from lace.cosmo import camb_cosmo

# %%
cosmo_planck = set_cosmo("Planck18")
# cosmo_2 = set_cosmo("DESIDR2_ACT")
# cosmo_2 = set_cosmo("Planck18_h74")
cosmo_2 = set_cosmo("Planck18_high3s_omh2")
# cosmo_2 = set_cosmo("Planck18_mnu03")
# cosmo_3 = set_cosmo("Planck18_mnu03_varh")
cosmo_3 = set_cosmo("Planck18_high_omh2")
# DESIDR2_ACT
# Planck18_h74
# Planck18_mnu03
# Planck18_high_omh2
# Planck18_low_omh2

# %%
omh1 = cosmo_planck.omch2 + cosmo_planck.ombh2 + cosmo_planck.omnuh2
omh2 = cosmo_2.omch2 + cosmo_2.ombh2 + cosmo_2.omnuh2
omh3 = cosmo_3.omch2 + cosmo_3.ombh2 + cosmo_3.omnuh2
print(omh1, omh2, omh3)
print(cosmo_planck.ombh2, cosmo_2.ombh2, cosmo_3.ombh2)
print(cosmo_planck.omch2, cosmo_2.omch2, cosmo_3.omch2)
print(cosmo_planck.omnuh2, cosmo_2.omnuh2, cosmo_3.omnuh2)
print(cosmo_planck.omegam, cosmo_2.omegam, cosmo_3.omegam)
print(cosmo_planck.h, cosmo_2.h, cosmo_3.h)

# %%
zlab = [4.2, 3.0, 2.2]

camb1 = camb_cosmo.get_camb_results(cosmo_planck, zs=zlab)
camb2 = camb_cosmo.get_camb_results(cosmo_2, zs=zlab)
camb3 = camb_cosmo.get_camb_results(cosmo_3, zs=zlab)

# %%
z = 2.2
h1 = camb1.hubble_parameter(z)
h2 = camb2.hubble_parameter(z)
h3 = camb3.hubble_parameter(z)

print(h1/h2-1)
print(h1/h3-1)

# %%
ii = 2
h1 = camb1.get_fsigma8()[ii]/camb1.get_sigma8()[ii]
h2 = camb2.get_fsigma8()[ii]/camb2.get_sigma8()[ii]
h3 = camb3.get_fsigma8()[ii]/camb3.get_sigma8()[ii]

print(h1, camb1.get_sigma8()[ii])
print(h1/h2-1)
print(h1/h3-1)

# %%
jj = 6


for ii in range(3):
    k1_h = camb1.get_matter_transfer_data().transfer_data[0][:, ii] * cosmo_planck.h
    tt1 = camb1.get_matter_transfer_data().transfer_data[jj][:, ii]

    k2_h  = camb2.get_matter_transfer_data().transfer_data[0][:, ii] * cosmo_2.h
    tt2o = camb2.get_matter_transfer_data().transfer_data[jj][:, ii]
    tt2 = np.interp(k1_h, k2_h, tt2o)

    k3_h = camb3.get_matter_transfer_data().transfer_data[0][:, ii] * cosmo_3.h
    tt3o = camb3.get_matter_transfer_data().transfer_data[jj][:, ii]
    tt3 = np.interp(k1_h, k3_h, tt3o)

    _ = np.argmin(np.abs(k1_h - 1))
    print(zlab[ii], np.round(tt2[_]/tt1[_]-1, 4), np.round(tt3[_]/tt1[_]-1, 4))

    plt.plot(k1_h, tt2/tt1, "C"+str(ii), label="z={}".format(zlab[ii]))
    plt.plot(k1_h, tt3/tt1, "C"+str(ii)+"--")
plt.legend()
plt.xlabel(r"$k\, [1/Mpc]$")
# plt.ylabel(r"Transfer matter, ratio Mnu 0.3eV/Planck18")
plt.ylabel(r"Transfer w/o neutrinos, ratio Mnu 0.3eV/Planck18")
plt.xscale("log")

# Transfer_cdm = 2 (cdm)
# Transfer_b = 3 (baryons)
# Transfer_g = 4 (photons)
# Transfer_r = 5 (massless neutrinos)
# Transfer_nu = 6 (massive neutrinos)
# Transfer_tot = 7 (total matter)
# Transfer_nonu = 8 (total matter excluding neutrinos)


# %%

# %%

# %%
z = np.linspace(2.2, 3.2, 10)
plt.plot(z, camb1.hubble_parameter(z), label="Planck18")
plt.plot(z, camb2.hubble_parameter(z), label="Planck18_high_omh2")
plt.plot(z, camb3.hubble_parameter(z), label="Planck18_high_omh2b")
plt.legend()
plt.xlabel(r"$z$")
plt.ylabel(r"$H(z)$")

# %%
z = np.linspace(2.2, 3.2, 10)
plt.plot(z, camb2.hubble_parameter(z)/camb1.hubble_parameter(z), label="Planck18_high_omh2", color="C1")
plt.plot(z, camb3.hubble_parameter(z)/camb1.hubble_parameter(z), label="Planck18_high_omh2b", color="C2")
plt.legend()
plt.xlabel(r"$z$")
plt.ylabel(r"$H(z)/H_\mathrm{Planck18}(z)$")

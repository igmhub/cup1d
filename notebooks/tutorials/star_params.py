# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
from lace.cosmo import camb_cosmo


# %%
def scale_cosmo(fcosmo, As=0, ns=0, nrun=0, z_star=3, kp_kms=0.009):
    """Fast computation of blob when running with fixed background"""
    # differences in primordial power (at CMB pivot point)
    ratio_As = 1.0
    delta_ns = 0.0
    delta_nrun = 0.0
    
    if As == 0:
        ratio_As = 1.0
    else:
        ratio_As = As / fcosmo.cosmo.InitPower.As
        
    if ns == 0:
        delta_ns = 0
    else:
        delta_ns = ns - fcosmo.cosmo.InitPower.ns
    
    
    if ns == 0:
        delta_nrun = 0
    else:
        delta_nrun = nrun - fcosmo.cosmo.InitPower.nrun
    
    # pivot scale of primordial power
    ks_Mpc = fcosmo.cosmo.InitPower.pivot_scalar
    
    # likelihood pivot point, in velocity units
    dkms_dMpc = fcosmo.dkms_dMpc(z_star)
    kp_Mpc = kp_kms * dkms_dMpc
    
    # logarithm of ratio of pivot points
    ln_kp_ks = np.log(kp_Mpc / ks_Mpc)
    
    # get blob for fiducial cosmo
    ### TODO: make this more efficient! Maybe directly storing the params?
    fid_blob = fcosmo.get_linP_params()
    
    # rescale blobs
    delta_alpha_star = delta_nrun
    delta_n_star = delta_ns + delta_nrun * ln_kp_ks
    ln_ratio_A_star = (
        np.log(ratio_As)
        + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
    )

    alpha_star = fid_blob['alpha_star'] + delta_alpha_star
    n_star = fid_blob['n_star'] + delta_n_star
    Delta2_star = fid_blob['Delta2_star'] * np.exp(ln_ratio_A_star)
    
    return Delta2_star, n_star, alpha_star

# %%
cosmo_pl = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1,
)

fcosmo0 = CAMB_model.CAMBModel(
    zs=[3],
    cosmo=cosmo_pl,
)

star0 = fcosmo0.get_linP_params()
print(star0["Delta2_star"], star0["n_star"], star0["alpha_star"])

# %%

# %% [markdown]
# #### Only change As, ns, nrun; it works

# %%
Asnew = 2.105e-09 * 1.5
nsnew=0.9665 + 0.1
nrunnew=0.0 - 0.05
ds_new, ns_new, as_new = scale_cosmo(fcosmo0, As=Asnew, ns=nsnew, nrun=nrunnew)


cosmo_new = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=Asnew,
    ns=nsnew,
    nrun=nrunnew,
    pivot_scalar=0.05,
    w=-1,
)
fcosmo_new = CAMB_model.CAMBModel(
    zs=[3],
    cosmo=cosmo_new,
)
star = fcosmo_new.get_linP_params()
nr = 4
print(np.round(star0["Delta2_star"], nr), 
      np.round(star0["n_star"], nr), 
      np.round(star0["alpha_star"], nr))
print(np.round(star["Delta2_star"], nr), 
      np.round(star["n_star"], nr), 
      np.round(star["alpha_star"], nr))
print(np.round(ds_new, nr), np.round(ns_new, nr), np.round(as_new, nr))
print(ds_new/star["Delta2_star"]-1, ns_new/star["n_star"]-1, as_new/star["alpha_star"]-1)

# %% [markdown]
# #### vary other params

# %%
Asnew = 2.105e-09*1.12
nsnew=0.9665+0.028
nrunnew=-0.006
# omch2new = 0.119
omch2new = 0.13
ds_new, ns_new, as_new = scale_cosmo(fcosmo0, As=Asnew, ns=nsnew, nrun=nrunnew)


cosmo_new = camb_cosmo.get_cosmology(
    H0=67.66,
    mnu=0.0,
    omch2=omch2new,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-09,
    ns=0.9665,
    nrun=0,
    pivot_scalar=0.05,
    w=-1,
)
fcosmo_new = CAMB_model.CAMBModel(
    zs=[3],
    cosmo=cosmo_new,
)
star = fcosmo_new.get_linP_params()
nr = 4
print(np.round(star["Delta2_star"], nr), 
      np.round(star["n_star"], nr), 
      np.round(star["alpha_star"], nr))
print(np.round(ds_new, nr), np.round(ns_new, nr), np.round(as_new, nr))
print(ds_new/star["Delta2_star"]-1, ns_new/star["n_star"]-1, as_new/star["alpha_star"]-1)

# %%
omch2new = np.linspace(0.119 - 0.001 * 5, 0.119 + 0.001 * 5, 100)

# %%
# %%time

res = np.zeros((omch2new.shape[0], 4))
for ii in range(omch2new.shape[0]):
    cosmo_new = camb_cosmo.get_cosmology(
        H0=67.66,
        mnu=0.0,
        omch2=omch2new[ii],
        ombh2=0.0224,
        omk=0.0,
        As=2.105e-09,
        ns=0.9665,
        nrun=0,
        pivot_scalar=0.05,
        w=-1,
    )
    fcosmo_new = CAMB_model.CAMBModel(
        zs=[3],
        cosmo=cosmo_new,
    )
    star = fcosmo_new.get_linP_params()
    res[ii, 0] = star["Delta2_star"]
    res[ii, 1] = star["n_star"]
    res[ii, 2] = star["alpha_star"]
    res[ii, 3] = omch2new[ii]

# %%
from corner import corner


# %%
corner(res, labels=["Delta2_star", "n_star", "alpha_star", "omch2"]);
plt.savefig("star_omch2.png")

# %%

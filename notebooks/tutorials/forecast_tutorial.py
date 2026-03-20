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
# # Tutorial for forescasts with DR1 data

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from cup1d.likelihood.pipeline import Pipeline
from cup1d.likelihood.input_pipeline import Args


# %% [markdown]
# ## Load Mock P1D measurements and set likelihood
#
# The mock measurements are created with the emulator
#
# Sampler configuration:
# - "test": just to check that everything is work
# - "explore": standard setting to get a good chain

# %%
data_label = "mock_DESIY1_QMLE3"
cov_label = "DESIY1_QMLE3"
fit_type = "global_opt"
name_variation = "no_contaminants"

# mcmc_conf = "explore"
mcmc_conf = "test"

args = Args(data_label=data_label, cov_label=cov_label)

args.set_baseline(
    fix_cosmo=False,
    fit_type=fit_type,
    P1D_type=cov_label,
    name_variation=name_variation,
    mcmc_conf=mcmc_conf,
)

pip = Pipeline(args)

# %% [markdown]
# ## Plot Mock P1D data 

# %%
pip.fitter.like.plot_p1d()

# %%
# measurements in z bins (no correlation between z bins)
k_kms = pip.fitter.like.data.k_kms
Pk_kms = pip.fitter.like.data.Pk_kms
cov_Pk_kms = pip.fitter.like.cov_Pk_kms
print(len(k_kms), len(Pk_kms), len(cov_Pk_kms))
print(k_kms[0].shape, Pk_kms[0].shape, cov_Pk_kms[0].shape)

# %%
# measurements in full array (correlation between z bins)
k_kms = pip.fitter.like.data.full_k_kms
Pk_kms = pip.fitter.like.data.full_Pk_kms
cov_Pk_kms = pip.fitter.like.full_cov_Pk_kms
print(k_kms.shape, Pk_kms.shape, cov_Pk_kms.shape)

# %% [markdown]
# ### Components of the covariance matrix

# %%
# access the components of the covariance matrix
## stat + sys + emu
cov_Pk_kms_tot = pip.fitter.like.cov_Pk_kms
## stat
cov_Pk_kms_stat = pip.fitter.like.data.covstat_Pk_kms
## syst
cov_Pk_kms_syst = []
for ii in range(len(cov_Pk_kms_stat)):
    cov_Pk_kms_syst.append(pip.fitter.like.data.cov_Pk_kms[ii] - cov_Pk_kms_stat[ii])
## emu
cov_Pk_kms_emu = pip.fitter.like.cov_emu_Pk_kms

print(
    len(cov_Pk_kms_tot), len(cov_Pk_kms_stat), len(cov_Pk_kms_syst), len(cov_Pk_kms_emu)
)

# %% [markdown]
# #### If you want to modify the covariance matrix by had, you need to change
#
# - pip.fitter.like.icov_Pk_kms
# - pip.fitter.like.full_icov_Pk_kms

# %%
pip.fitter.like.full_icov_Pk_kms.shape

# %% [markdown]
# ### Get predictions from the model

# %%
# list of model parameters

for par in pip.fitter.like.free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %%
# evaluate model for the initial value of the input parameters
zs = pip.fitter.like.data.z
k_kms = pip.fitter.like.data.k_kms
ini_free_params = pip.fitter.like.free_params

ini_model_Pk_kms = pip.fitter.like.theory.get_p1d_kms(
    zs, k_kms, like_params=ini_free_params
)[0]

# %%
# evaluate model for other values of input parameters, only changing As
zs = pip.fitter.like.data.z
k_kms = pip.fitter.like.data.k_kms

new_free_params = []
for par in pip.fitter.like.free_params:
    old_value = par.value
    new_par = par.get_new_parameter(0.5)
    if par.name == "As":
        # increase by 10%
        new_par.value = old_value * 1.1
    else:
        # same value as before
        new_par.value = old_value
    new_free_params.append(new_par)

new_As_model_Pk_kms = pip.fitter.like.theory.get_p1d_kms(
    zs, k_kms, like_params=new_free_params
)[0]

# %% [markdown]
# Show ratio of both predictions

# %%
ii = 0
plt.plot(
    k_kms[ii],
    new_As_model_Pk_kms[ii] / ini_model_Pk_kms[ii],
    label="z = " + str(zs[ii]),
)
plt.ylabel("$P_\mathrm{1D}(k)/P_\mathrm{1D}^\mathrm{ini}(k)$")
plt.xlabel("$k_\parallel$ [km/s]")
plt.legend()
plt.show()

# %% [markdown]
# ### Compressed parameters

# %%
blob = pip.fitter.like.theory.get_blob_fixed_background(ini_free_params)
ini_Delta2_star = blob[0]
ini_n_star = blob[1]
print(ini_Delta2_star, ini_n_star)

# %%
blob = pip.fitter.like.theory.get_blob_fixed_background(new_free_params)
new_Delta2_star = blob[0]
new_n_star = blob[1]
print(new_Delta2_star, new_n_star)

# %%
# as expected, 10% larger value of the new parameter
new_Delta2_star/ini_Delta2_star

# %% [markdown]
# ### Run fitter

# %%
run_fitter = False

if run_fitter:
    p0 = pip.fitter.like.sampling_point_from_parameters()
    pip.run_minimizer(p0, restart=True)
    pip.fitter.like.plot_p1d(pip.fitter.mle_cube)

# %% [markdown]
# ### Run sampler

# %%
run_sampler = True

if run_sampler:
    p0 = pip.fitter.like.sampling_point_from_parameters()
    pip.run_sampler(p0)

# %%
Delta2_star = pip.fitter.blobs["Delta2_star"].reshape(-1)
n_star = pip.fitter.blobs["n_star"].reshape(-1)

# %%
plt.scatter(Delta2_star, n_star)
plt.ylabel("$n_\mathrm{star}$")
plt.xlabel("$\Delta^2_\mathrm{star}$")

# %%

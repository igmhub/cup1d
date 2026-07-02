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
#     display_name: test_lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial for DR1 data

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from cup1d.likelihood.pipeline import Pipeline


# %%
from scipy.stats import qmc

# 1. Define dimensions (features) and number of samples
dimensions = 3
n_samples = 5

# 2. Initialize the Latin Hypercube sampler
sampler = qmc.LatinHypercube(d=dimensions, seed=42)

# 3. Generate samples in the [0, 1) range
sample_matrix = sampler.random(n=n_samples)

# %%
sample_matrix.shape

# %%

# %% [markdown]
# ## Load P1D measurements and set likelihood

# %%
pip = Pipeline()

# %% [markdown]
# ## Plot P1D data 
#
# Get parameters from a point of the parameter space close to the best fit

# %%
p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %% [markdown]
# Plot model for these parameters

# %%
pip.fitter.like.plot_p1d(p0)

# %% [markdown]
# #### If you want to extract the data

# %%
# measurements in z bins (no correlation between z bins)
key = list(pip.fitter.like.data.keys())[0]

k_kms = pip.fitter.like.data[key].k_kms
Pk_kms = pip.fitter.like.data[key].Pk_kms
cov_Pk_kms = pip.fitter.like.cov_Pk_kms[key]
print(len(k_kms), len(Pk_kms), len(cov_Pk_kms))
print(k_kms[0].shape, Pk_kms[0].shape, cov_Pk_kms[0].shape)

# %%
# measurements in full array (correlation between z bins)
k_kms = pip.fitter.like.data[key].full_k_kms
Pk_kms = pip.fitter.like.data[key].full_Pk_kms
cov_Pk_kms = pip.fitter.like.full_cov_Pk_kms[key]
print(k_kms.shape, Pk_kms.shape, cov_Pk_kms.shape)

# %% [markdown]
# #### Components of the covariance matrix

# %%
pip.fitter.like.plot_cov_to_pk()

# %%
# access the components of the covariance matrix
key = list(pip.fitter.like.data.keys())[0]

## stat + sys + emu
cov_Pk_kms_tot = pip.fitter.like.cov_Pk_kms[key]
## stat
cov_Pk_kms_stat = pip.fitter.like.data[key].covstat_Pk_kms
## syst
cov_Pk_kms_syst = []
for ii in range(len(cov_Pk_kms_stat)):
    cov_Pk_kms_syst.append(pip.fitter.like.data[key].cov_Pk_kms[ii] - cov_Pk_kms_stat[ii])
## emu
cov_Pk_kms_emu = pip.fitter.like.cov_emu_Pk_kms[key]

print(
    len(cov_Pk_kms_tot), len(cov_Pk_kms_stat), len(cov_Pk_kms_syst), len(cov_Pk_kms_emu)
)

# %% [markdown]
# ### Get predictions from the model

# %%
# list of model parameters

for par in pip.fitter.like.free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %% [markdown]
# #### Evaluate the model for some input parameters

# %%
# evaluate model for the initial value of the input parameters
zs = pip.fitter.like.data[key].z
k_kms = pip.fitter.like.data[key].k_kms
ini_free_params = pip.fitter.like.free_params

ini_model_Pk_kms = pip.fitter.like.theory.get_p1d_kms(
    zs, k_kms, like_params=ini_free_params
)[0]

# %%
# evaluate model for other values of input parameters, only changing As
zs = pip.fitter.like.data[key].z
k_kms = pip.fitter.like.data[key].k_kms

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
# ## Run minimizer

# %% [markdown]
# Get value of parameters close to best fit again

# %%
p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %% [markdown]
# Run minimizer starting from this point, it should stop the minimization soon

# %%
pip.run_minimizer(p0)

# %% [markdown]
# Evaluate for the new best fit

# %%
p1 = pip.fitter.mle_cube
pip.fitter.like.plot_p1d(p1)

# %%

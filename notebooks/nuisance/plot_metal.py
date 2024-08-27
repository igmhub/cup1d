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
# # Parameterization of the metal contamination
#
# The measured power spectrum presents clear wiggles that are caused by Lya x Silicon correlations. 
# In this notebook we discuss possible parameterizations of this contamination, and choose priors for the parameters.

# %% [markdown]
# For now we describe the amplitude of the contamination with a power law on $(1+z)$ around $z_\star=3$.

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
## Set default plot size, as normally its a bit too small
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.dpi'] = 120
from cup1d.nuisance import metal_model
from cup1d.nuisance import mean_flux_model

# %%
X_model=metal_model.MetalModel(metal_label='SiIII')

# %%
for p in X_model.X_params:
    print(p.info_str())

# %%
X_model.get_amplitude(z=3)

# %%
X_model.get_dv_kms()

# %%
k_kms=np.linspace(0,0.03,1000)
cont=X_model.get_contamination(z=3, k_kms=k_kms, mF=0.7)
plt.plot(k_kms,cont)

# %%
# construct two models for the mean flux, with different number of parameters
X_model_test=metal_model.MetalModel(metal_label='SiIII',ln_X_coeff=[-1,np.log(0.01)])

# %%
mF_model=mean_flux_model.MeanFluxModel()

# %%
mF_model.get_mean_flux(z=3)

# %%
k_kms=np.linspace(0,0.03,1000)
for z in [2,3,4]:
    mF=mF_model.get_mean_flux(z=z)
    plt.figure()
    cont=X_model.get_contamination(z=z, k_kms=k_kms, mF=mF)
    test=X_model_test.get_contamination(z=z, k_kms=k_kms, mF=mF)
    plt.plot(k_kms,cont,label='fiducial')
    plt.plot(k_kms,test,label='test')
    plt.xlabel('k [s/km]')
    plt.ylabel('metal contamination')
    plt.title('z={}'.format(z))
    plt.legend()

# %%

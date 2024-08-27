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
# # Parameterization of the HCD contamination
#
# Contamination from High Column Density (HCD) systems, following McDonald et al. (2005)

# %% [markdown]
# For now we describe the amplitude of the contamination with a power law on $(1+z)$ around $z_0=3$.

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
from cup1d.nuisance import hcd_model_McDonald2005

# %%
hcd_model=hcd_model_McDonald2005.HCD_Model_McDonald2005(ln_A_damp_coeff=[0.0])

# %%
for p in hcd_model.params:
    print(p.info_str())

# %%
hcd_model.get_A_damp(z=10)

# %%
k_kms=np.linspace(0.001,0.02,1000)
cont=hcd_model.get_contamination(z=3, k_kms=k_kms)
plt.plot(k_kms,cont)
plt.ylim(0.9,1.2)

# %%
hcd_model_test=hcd_model_McDonald2005.HCD_Model_McDonald2005(ln_A_damp_coeff=[1,-0.1])

# %%
k_kms=np.linspace(0.001,0.03,1000)
for z in [2,3,4]:
    plt.figure()
    cont=hcd_model.get_contamination(z=z, k_kms=k_kms)
    test=hcd_model_test.get_contamination(z=z, k_kms=k_kms)
    plt.plot(k_kms,cont,label='fiducial')
    plt.plot(k_kms,test,label='test')
    plt.xlabel('k [s/km]')
    plt.ylabel('HCD contamination')
    plt.title('z={}'.format(z))
    plt.legend()

# %%

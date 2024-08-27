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
# # Generate noisy realizations of P1D mocks
#
# This notebook explains how to generate noisy realizations of P1D mocks

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 160
mpl.rcParams['figure.dpi'] = 160
from lace.archive import gadget_archive, nyx_archive
from cup1d.data.data_eBOSS_mock import P1D_eBOSS_mock
from cup1d.data.data_gadget import Gadget_P1D
from cup1d.data.data_nyx import Nyx_P1D

# %% [markdown]
# ## Generate eBOSS P1D mock

# %% [markdown]
# #### Fiducial

# %%
eBOSS_mock =P1D_eBOSS_mock(add_noise=False)
eBOSS_mock.plot_p1d()

# %% [markdown] jupyter={"outputs_hidden": false}
# #### Generate mock by perturbing values accoring to the cov matrix

# %%
eBOSS_mock =P1D_eBOSS_mock(add_noise=True, seed=0)
eBOSS_mock.plot_p1d()

# %% [markdown]
# ### Generate gadget mock

# %%
mpg_archive = gadget_archive.GadgetArchive(postproc='Cabayol23')

# %%
mpg_mock = Gadget_P1D(archive=mpg_archive, input_sim="mpg_central", add_noise=False)
mpg_mock.plot_p1d()

# %% [markdown]
# #### with noise

# %%
mpg_mock =Gadget_P1D(archive=mpg_archive, input_sim="mpg_central", add_noise=True)
mpg_mock.plot_p1d()

# %% [markdown]
# ## Check that we converge to fiducial P1D

# %% [markdown]
# #### Generate random realizations and plot

# %%
nsamples = 50
eBOSS_mock = P1D_eBOSS_mock(add_noise=False)
realization = eBOSS_mock.get_Pk_iz_perturbed(eBOSS_mock.Pk_kms, eBOSS_mock.cov_Pk_kms, nsamples=nsamples)
for iz in range(len(eBOSS_mock.z)):
    k = eBOSS_mock.k_kms
    norm = k/np.pi
    orig = eBOSS_mock.Pk_kms[iz]    
    err = np.sqrt(np.diag(eBOSS_mock.cov_Pk_kms[iz]))
    plt.errorbar(k, norm * orig, norm * err, label=r'$z$='+str(eBOSS_mock.z[iz]))
    for jj in range(nsamples):
        plt.plot(k, norm * realization[iz][jj], 'k', alpha=0.1)
    

plt.xlabel(r'$k$ [s/km]')
plt.ylabel(r'$k P(k)/\pi$')
plt.yscale('log')
plt.legend()

# %% [markdown]
# #### Compare median and error-bars

# %%
nsamples = 1000
eBOSS_mock = P1D_eBOSS_mock(add_noise=False)
realization = eBOSS_mock.get_Pk_iz_perturbed(eBOSS_mock.Pk_kms, eBOSS_mock.cov_Pk_kms, nsamples=nsamples)
for iz in range(len(eBOSS_mock.z)):
    k = eBOSS_mock.k_kms
    norm = k/np.pi
    orig = eBOSS_mock.Pk_kms[iz]    
    err = np.sqrt(np.diag(eBOSS_mock.cov_Pk_kms[iz]))
    plt.errorbar(k, norm * orig, norm * err, label=r'$z$='+str(eBOSS_mock.z[iz]))
    
    yy = np.mean(realization[iz], axis=0)
    err_yy = np.std(realization[iz], axis=0)
    plt.errorbar(k, norm * yy, norm * err_yy, c='k', alpha=0.5)
    

plt.xlabel(r'$k$ [s/km]')
plt.ylabel(r'$k P(k)/\pi$')
plt.yscale('log')
plt.legend()

# %%

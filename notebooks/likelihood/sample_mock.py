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
# ## Sample mock P1D (from model, not sim)
#
# This notebook contains the basic syntax required to run a chain. We set up a mock data object from a given P1D model, construct an emulator and likelihood object, and pass these to a sampler to run for a small number of steps.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 140
mpl.rcParams['figure.dpi'] = 140
import numpy as np
import time
# our own modules
from lace.emulator import gp_emulator
from lace.emulator import nn_emulator
from cup1d.data import mock_data
from cup1d.likelihood import lya_theory
from cup1d.likelihood import likelihood
from cup1d.likelihood import emcee_sampler

# %%
# specify if you want to add high-resolution P1D (only working for Pedersen23)
add_hires=False
if add_hires:
    kmax_Mpc=8
else:
    kmax_Mpc=4

# %% [markdown]
# ### Setup emulator
#
# The emulator will be used both to make a mock, and in the likelihood 

# %%
# emulator=nn_emulator.NNEmulator(training_set='Nyx23',emulator_label='Cabayol23_Nyx')
#emulator=nn_emulator.NNEmulator(training_set='Cabayol23',emulator_label='Cabayol23')
emulator=gp_emulator.GPEmulator(training_set='Pedersen21',kmax_Mpc=kmax_Mpc)

# %% [markdown]
# ### Create mock P1D data
#
# Use Lya theory to mimic mock data

# %%
data=mock_data.Mock_P1D(emulator=emulator,data_label="Chabanier2019")

# %%
# check if we also need mock extra_p1d
if add_hires:
    extra_data=mock_data.Mock_P1D(emulator=emulator,data_label="Karacayli2022")
else:
    extra_data=None

# %% [markdown]
# ### Set free parameters and theory

# %%
# stick to primordial power-law parameters here
free_param_names=["As","ns"]
# specify the number of free parameters per IGM function (default=2)
n_igm=0
for i in range(n_igm):
    for par in ["tau","sigT_kms","gamma","kF"]:
        free_param_names.append('ln_{}_{}'.format(par,i))

# %%
theory=lya_theory.Theory(zs=data.z,emulator=emulator,free_param_names=free_param_names)

# %%
# print parameter values used to create mock data
for p in theory.get_parameters():
    print(p.info_str(all_info=True))

# %% [markdown]
# ### Set up a likelihood
#
# Here we chose which parameters we want to sample, over which range and chose a prior. We pass the data and theory objects to the likelihood.

# %%
# option to include/remove a Gaussian prior (in unit cube)
prior_Gauss_rms=None
# option to include/ignore emulator covariance (it might bias the results)
emu_cov_factor=0
like=likelihood.Likelihood(data=data,theory=theory,
                            free_param_names=free_param_names,
                            prior_Gauss_rms=prior_Gauss_rms,
                            emu_cov_factor=emu_cov_factor,
                            extra_p1d_data=extra_data)

# %% [markdown]
# ### Sampler object
#
# Here we configure our sampler, set the number of walkers, and decide whether or not we want to save the chain to be resumed or plot later on.

# %%
# Set up sampler
n_burn_in=50
n_steps=500 
sampler = emcee_sampler.EmceeSampler(like=like,progress=True, nburnin=n_burn_in, nsteps=n_steps)

# %%
for p in sampler.like.free_params:
    print(p.name,p.value,p.min_value,p.max_value)

# %%

start = time.time()
sampler.run_sampler(n_burn_in,n_steps)
end = time.time()
sampler_time = end - start
print("Sampling took {0:.1f} seconds".format(sampler_time))

# %%
sampler.write_chain_to_file(residuals=True,plot_nersc=True,plot_delta_lnprob_cut=50)

# %%
sampler.plot_corner(plot_params=['$\\Delta^2_\\star$','$n_\\star$'],
                    delta_lnprob_cut=50,usetex=False,serif=False)

# %%
from corner import corner

# %%

# %%
sampler.plot_corner(plot_params=['$\\Delta^2_\\star$','$n_\\star$'],
                    delta_lnprob_cut=50,usetex=False,serif=False)

# %%
params_plot, strings_plot = sampler.get_all_params(
            delta_lnprob_cut=50
        )

# %%
params_plot[:,:2]

# %%
chain, lnprob, blobs = sampler.get_chain(
            cube=False, delta_lnprob_cut=50
        )

# %%
corner(chain, labels=['$\\Delta^2_\\star$','$n_\\star$']);

# %%
blobs_full = np.hstack(
    (
        np.vstack(blobs["Delta2_star"]),
        np.vstack(blobs["n_star"]),
        np.vstack(blobs["f_star"]),
        np.vstack(blobs["g_star"]),
        np.vstack(blobs["alpha_star"]),
        np.vstack(blobs["H0"]),
    )
)
# Array for all parameters
all_params = np.hstack((chain, blobs_full))

# %%
corner(all_params[:,:2], labels=['$\\Delta^2_\\star$','$n_\\star$']);

# %%
all_params.shape

# %%

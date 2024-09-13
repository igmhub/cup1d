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
# # Tutorial: sample data
#
# This notebook shows an illustrative example of how to run cup1d for extracting cosmological constraints from P1D data:
#
# - Set mock data
# - Set emulator
# - Set likelihood
# - Set sampler
# - Run sample for a small number of steps
#
# All these steps are implemented in cup1d/cup1d/likelihood/samplerpipeline.py. If you are interested in running cup1d, please take a look at cup1d/scripts/sam_sim.py. That script is parallelized using MPI and includes a bookkeeper taking care of all relevant options.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
import lace
from lace.archive import gadget_archive, nyx_archive
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.p1ds import (
    data_gadget,
    data_nyx,
    mock_data,
    data_eBOSS_mock,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Karacayli2024,
    data_Ravoux2023,
)
from cup1d.likelihood import lya_theory, likelihood, emcee_sampler
from cup1d.likelihood.sampler_pipeline import set_archive, set_P1D, set_P1D_hires, set_fid_cosmo, set_like
from cup1d.likelihood.input_pipeline import Args

# %% [markdown]
# ### Set up arguments
#
# Info about these and other arguments in cup1d.likelihood.input_pipeline.py

# %%
# set output directory for this test
output_dir = "."

args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")

args.data_label="mock_Karacayli2024"
args.data_label_hires = "mock_Karacayli2022"
args.data_label="nyx_central"
# args.data_label_hires="mpg_central"

args.cosmo_label="mpg_central"
args.igm_label="mpg_central"
args.vary_alphas=False

# args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Oct2023")
# args.data_label="nyx_central"
# args.cosmo_label="nyx_central"
# args.igm_label="nyx_central"
# args.vary_alphas=True

args.n_igm=2
args.n_steps=50
args.n_burn_in=10
args.z_max=4.5
args.parallel=False
args.explore=True
args.add_metals=False
# args.add_hires=True
args.add_hires=False

# %% [markdown]
# ### Set archive

# %%
archive = set_archive(args.training_set)

# %% [markdown]
# ### Set emulator

# %%
emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

# %% [markdown]
# ### Set fiducial cosmology

# %%
cosmo_fid = set_fid_cosmo(cosmo_label=args.cosmo_label)

# %% [markdown]
# ### Set P1D data
#
# We create mock data starting from an mpg simulation, but we can set obs data

# %%
data = {"P1Ds": None, "extra_P1Ds": None}
data["P1Ds"], true_sim_igm = set_P1D(
    archive,
    emulator,
    args.data_label,
    cosmo_fid,
    cov_label=args.cov_label,
    igm_label=args.igm_label,
    apply_smoothing=False,
    z_min=args.z_min,
    z_max=args.z_max,
)

if(args.add_hires):
    data["extra_P1Ds"] = set_P1D_hires(
        archive,
        emulator,
        args.data_label_hires,
        cosmo_fid,
        cov_label_hires=args.cov_label_hires,
        igm_label=args.igm_label,
        apply_smoothing=False,
        z_min=args.z_min,
        z_max=args.z_max,
    )

# %%
data["P1Ds"].plot_p1d()
if(args.add_hires):
    data["extra_P1Ds"].plot_p1d()

# %%
data["P1Ds"].plot_igm()

# %% [markdown]
# ### Set likelihood

# %%
like = set_like(
    emulator,
    data["P1Ds"],
    data["extra_P1Ds"],
    true_sim_igm,
    args.igm_label,
    args.n_igm,
    cosmo_fid,
    vary_alphas=args.vary_alphas,
    add_metals=args.add_metals
)

# %% [markdown]
# Plot residual between P1D data and emulator for fiducial cosmology (should be the same in this case)

# %%
data["P1Ds"].truth

# %%
like.truth

# %%
like.plot_p1d(residuals=False, plot_every_iz=1)
like.plot_p1d(residuals=True, plot_every_iz=2)

# %% [markdown]
# Priors for sampling parameters

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)


# %% [markdown]
# ### Set sampler

# %%
def log_prob(theta):
    return log_prob.sampler.like.get_chi2(theta)

def set_log_prob(sampler):
    log_prob.sampler = sampler
    return log_prob

sampler = emcee_sampler.EmceeSampler(
    like=like,
    rootdir=output_dir,
    save_chain=False,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)
_get_chi2 = set_log_prob(sampler)

# %% [markdown]
# ### Run sampler
# It takes less than 2 min on my laptop without any parallelization

# %%
# %%time
_emcee_sam = sampler.run_sampler(log_func=_log_prob)

# %% [markdown]
# ### Run minimizer

# %%
# %%time
ind = np.argmax(sampler.lnprob.reshape(-1))
nparam = sampler.chain.shape[-1]
p0 = sampler.chain.reshape(-1, nparam)[ind, :]
sampler.run_minimizer(log_func=_log_prob, p0=p0)

# %%
# %%time
p0 = np.zeros(len(like.free_params)) + 0.5
sampler.run_minimizer(log_func_minimize=_get_chi2, p0=p0)

# %%
# like.plot_p1d(residuals=False, plot_every_iz=1, values=sampler.mle_cube)
like.plot_p1d(residuals=True, plot_every_iz=2, values=sampler.mle_cube)
# like.plot_p1d(residuals=True, plot_every_iz=2, values=ptruth)

# %% [markdown]
# ### Get plots
#
# Get interesting plots, these are in the folder created with the output

# %%
sampler.write_chain_to_file()

# %%
# p1 = {'Delta2_p': 0.6424254870204057, 'n_p': -2.284963361317453, 'alpha_p': -0.21536767260941628, 'mF': 0.8333555955907445, 'gamma': 1.5166829814584781, 'sigT_Mpc': 0.10061115435052223, 'kF_Mpc': 10.614589838852988}
# k = np.linspace(0.1, 5, 50)
# z = 2.2
# res = emulator.emulate_p1d_Mpc(p1, k, z=z, return_covar=True)
# p1d, cov = res
# plt.errorbar(k, k*p1d, k*np.sqrt(np.diag(cov)))
# plt.xscale('log')

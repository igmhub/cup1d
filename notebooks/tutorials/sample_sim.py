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
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood import lya_theory, likelihood
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter

from cup1d.likelihood.pipeline import (
    set_archive,
    set_P1D,
    set_cosmo,
    set_free_like_parameters,
    set_like,
)
from cup1d.p1ds.data_DESIY1 import P1D_DESIY1

from cup1d.likelihood.input_pipeline import Args

# %% [markdown]
# ### Set emulator

# %%
# set output directory for this test
output_dir = "."


# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# add to .bashrc export NYX_PATH="/global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/"
args = Args(emulator_label="Nyx_alphap_cov", training_set="Nyx23_Jul2024")

archive = set_archive(args.training_set)

emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

if "Nyx" in emulator.emulator_label:
    emulator.list_sim_cube = archive.list_sim_cube
    if "nyx_14" in emulator.list_sim_cube:
        emulator.list_sim_cube.remove("nyx_14")
else:
    emulator.list_sim_cube = archive.list_sim_cube

# %% [markdown]
# #### Set either mock data or real data
#
# If you want other options, take a look at cup1d.p1ds

# %%
args.apply_smoothing

# %%

# set IGM histories
true_sim_label = "nyx_central"
args.data_label=true_sim_label
args.true_cosmo_label=true_sim_label
args.true_label_mF=true_sim_label
args.true_label_T=true_sim_label
args.true_label_kF=true_sim_label
args.z_max=4

# provide cosmology only to cull the data if needed
true_cosmo = set_cosmo(cosmo_label=true_sim_label)


# you do not need to provide the archive for obs data 
data = {"P1Ds": None, "extra_P1Ds": None}

data["P1Ds"] = set_P1D(
    args,
    archive=archive,
    true_cosmo=true_cosmo,
    emulator=emulator,
    cull_data=False,
)

# %%
print(data["P1Ds"].apply_blinding)
if data["P1Ds"].apply_blinding:
    print(data["P1Ds"].blinding)

# %%
data["P1Ds"].plot_p1d()
if args.data_label_hires is not None:
    data["extra_P1Ds"].plot_p1d()

# %%
try:
    data["P1Ds"].plot_igm()
except:
    print("Real data, no true IGM history")

# %% [markdown]
# #### Set fiducial/initial options for the fit

# %%
## cosmology
# correction due to wrong initial conditions for Lyssa simulations (https://arxiv.org/abs/2412.05372)
args.ic_correction=False

# fiducial cosmology and IGM histories
sim_fid = "nyx_central"
args.fid_cosmo_label=sim_fid
args.fid_label_mF=sim_fid
args.fid_label_T=sim_fid
args.fid_label_kF=sim_fid
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

args.mF_model_type = "chunks"
args.n_tau=len(data["P1Ds"].z)
args.n_sigT=1
args.n_gamma=1
args.n_kF=1

free_parameters = set_free_like_parameters(args, emulator.emulator_label)
free_parameters

# %% [markdown]
# ### Set likelihood

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

# %% [markdown]
# Sampling parameters

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %% [markdown]
# Compare data and fiducial/starting model

# %%
like.plot_p1d(residuals=False)
like.plot_p1d(residuals=True)

# %%
like.plot_igm()

# %% [markdown]
# ### Set fitter

# %%
args.n_steps=5
args.n_burn_in=1
args.parallel=False
args.explore=True

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

# %% [markdown]
# ### Run minimizer

# %%
# %%time
if like.truth is None:
    # p0 = np.zeros(len(like.free_params)) + 0.5
    p0 = np.array(list(like.fid["fit_cube"].values()))
else:
    p0 = np.array(list(like.truth["like_params_cube"].values()))*1.01
p0 = np.array(list(like.fid["fit_cube"].values()))
# p0[:] = 0.5
fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %%
plotter = Plotter(fitter, save_directory=None)
if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()
plotter.plots_minimizer()


# %% [markdown]
# ### Run sampler

# %%
# %%time

def func_for_sampler(p0):
    res = fitter.like.get_log_like(values=p0, return_blob=True)
    return res[0], *res[2]

run_sampler = True
if run_sampler:    
    _emcee_sam = fitter.run_sampler(pini=fitter.mle_cube, log_func=func_for_sampler)

# %%
fitter.write_chain_to_file()

# %%
plotter = Plotter(fitter, save_directory=None)

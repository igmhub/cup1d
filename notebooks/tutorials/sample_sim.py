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

from cup1d.likelihood.pipeline import (
    set_archive,
    set_P1D,
    set_cosmo,
    set_free_like_parameters,
    set_like,
)

from cup1d.likelihood.input_pipeline import Args

# %%
from cup1d.p1ds.data_QMLE_DESIY1 import P1D_QMLE_DESIY1

# %%
p1d = P1D_QMLE_DESIY1()

# %%
p1d.plot_p1d(fname="p1d_desi.png")

# %% [markdown]
# ### Set up arguments
#
# Info about these and other arguments in cup1d.likelihood.input_pipeline.py

# %% [markdown]
# ### Set emulator

# %%
# set output directory for this test
output_dir = "."

# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# the nyx emulator has not properly been validated yet
# args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Oct2023")

archive = set_archive(args.training_set)

emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

if emulator.emulator_label == "Nyx_alphap":
    emulator.list_sim_cube = archive.list_sim_cube
    emulator.list_sim_cube.remove("nyx_14")
else:
    emulator.list_sim_cube = archive.list_sim_cube

# %%

# %%

args = Args(emulator_label="Nyx_v0", training_set="Nyx23_Oct2023")
emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

# %%

if emulator.emulator_label == "Nyx_v0":
    emulator.list_sim_cube = archive.list_sim_cube
    emulator.list_sim_cube.remove("nyx_14")

# %% [markdown]
# #### Set either mock data or real data

# %%
choose_forecast = False
choose_mock = True
choose_data = False

if choose_forecast:
    # for forecast, just start label of observational data with mock
    args.data_label = "mock_Chabanier2019"
    # args.data_label="mock_Karacayli2024"
    args.data_label_hires = "mock_Karacayli2022"

    # you need to provide true cosmology, IGM history, and contaminants
    true_cosmo = set_cosmo(cosmo_label="mpg_central")
    args.true_igm_label="mpg_central"
    # args.true_igm_label="nyx_central"
    # from -11 to -4
    args.true_SiIII=[0, -10]
    args.true_SiII=[0, -10]
    # from -7 to 0
    args.true_HCD=[0, -6]
    # from -5 to 2
    args.true_SN=[0, -4]
elif choose_mock:    
    true_cosmo=None
    # to analyze data from simulations
    # args.data_label = "mpg_central"    
    args.data_label="nyx_central"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None

    # provide cosmology only to cull the data
    # args.true_cosmo_label="mpg_central"
    args.true_cosmo_label="nyx_central"
    
    # you need to provide contaminants
    # from -11 to -4
    args.true_SiIII=[0, -10]
    args.true_SiII=[0, -10]
    # from -7 to 0
    args.true_HCD=[0, -6]
    # from -5 to 2
    args.true_SN=[0, -4]
elif choose_data:    
    true_cosmo=None
    args.data_label = "Chabanier2019"
    # args.data_label="Karacayli2024"
    args.data_label_hires = "Karacayli2022"
    args.z_max = 3.9

# you do not need to provide the archive for obs data 
data = {"P1Ds": None, "extra_P1Ds": None}
data["P1Ds"] = set_P1D(
    args.data_label,
    args,
    archive=archive,
    true_cosmo=true_cosmo,
    emulator=emulator,
    cull_data=False
)
if args.data_label_hires is not None:
    data["extra_P1Ds"] = set_P1D(
        args.data_label_hires,
        args,
        archive=archive,
        true_cosmo=true_cosmo,
        emulator=emulator,
        cull_data=False
    )

# %%
data["P1Ds"].plot_p1d()
if args.data_label_hires is not None:
    data["extra_P1Ds"].plot_p1d()

# %%
if choose_data == False:
    data["P1Ds"].plot_igm()

# %% [markdown]
# #### Set fiducial/initial options for the fit

# %%
# cosmology
# args.fid_cosmo_label="mpg_central"
args.fid_cosmo_label="nyx_central"
# args.fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# IGM
# args.fid_igm_label="mpg_central"
args.fid_igm_label="nyx_central"
if choose_data == False:
    args.igm_priors = "hc"
else:
    args.type_priors = "data"

# contaminants
args.fid_SiIII=[0, -10]
args.fid_SiII=[0, -10]
args.fid_HCD=[0, -6]
args.fid_SN=[0, -4]

# parameters
args.vary_alphas=False
args.fix_cosmo=True
args.n_tau=2
args.n_sigT=2
args.n_gamma=2
args.n_kF=2
args.n_SiIII = 2
args.n_SiII = 0
args.n_dla=0
args.n_sn=0


free_parameters = set_free_like_parameters(args)
free_parameters

# %% [markdown]
# ### Set likelihood

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    fid_cosmo,
    free_parameters,
    args,
    data_hires=data["extra_P1Ds"]
)

# %% [markdown]
# Sampling parameters

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %% [markdown]
# Compare data and fiducial/starting model

# %%
like.plot_p1d(residuals=False, plot_every_iz=1, print_chi2=False)
like.plot_p1d(residuals=True, plot_every_iz=2, print_ratio=False)

# %%
like.plot_igm()

# %% [markdown]
# ### Set fitter

# %%
# for sampler, no real fit, just test
args.n_steps=50
args.n_burn_in=10
args.parallel=False
args.explore=True

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    save_chain=False,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

# %% [markdown]
# ### Run sampler
# It takes less than 2 min on my laptop without any parallelization

# %%
run_sampler = False
if run_sampler:
    _emcee_sam = sampler.run_sampler(log_func=fitter.like.get_chi2)

# %% [markdown]
# ### Run minimizer

# %%
# %%time
if like.truth is None:
    # p0 = np.zeros(len(like.free_params)) + 0.5
    p0 = np.array(list(like.fid["fit_cube"].values()))
else:
    p0 = np.array(list(like.truth["like_params_cube"].values()))*1.05
fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=16)

# %%
fitter.plot_p1d(residuals=False, plot_every_iz=1)

# %%
fitter.plot_p1d(residuals=True, plot_every_iz=2)

# %%
fitter.plot_igm(cloud=True)

# %%

# %%
# sampler.write_chain_to_file()

# %%
# import h5py
# nyx_file = "/home/jchaves/Proyectos/projects/lya/data/nyx/models_Nyx_Oct2023.hdf5"
# ff = h5py.File(nyx_file, "r")
# sim_avail = list(ff.keys())

# zkeys = list(ff["cosmo_grid_0"].keys())

# snap = ff["cosmo_grid_0"][zkeys[0]]
# list_scalings = list(snap.keys())

# z = np.zeros((len(list_scalings), len(zkeys)))
# fbar = np.zeros((len(list_scalings), len(zkeys)))

# for ii in range(len(list_scalings)):
#     for jj in range(len(zkeys)):
#         z[ii, jj] = float(zkeys[jj][-3:])
#         snap = ff["cosmo_grid_0"][zkeys[jj]]
#         if list_scalings[ii] in snap:
#             if "T_0" in snap[list_scalings[ii]].attrs.keys():
#                 fbar[ii, jj] = snap[list_scalings[ii]].attrs["T_0"]            
#             else:
#                 print(list_scalings[ii], zkeys[jj]) 


# for ii in range(len(list_scalings)):
#     if "new" in list_scalings[ii]:
#         col = "red"
#     elif "native" in list_scalings[ii]:
#         col = "k"
#     else:
#         col = "C1"
#     _ = np.argwhere(fbar[ii, :] != 0)[:,0]
#     if(len(_) > 0):
#         plt.plot(z[ii, _], fbar[ii, _], col, label=list_scalings[ii], alpha=0.75)
# # plt.legend()
# plt.xlabel("z")
# plt.ylabel("T_0")
# plt.savefig("nyx_T0.pdf")
# # plt.ylabel("gamma")
# # plt.savefig("nyx_gamma.pdf")

# %%

# %%
# check IGM histories
# zs=data["P1Ds"].z

# plt.plot(zs, like.theory.model_igm.F_model.get_tau_eff(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["tau_eff"], "--")

# plt.plot(zs, like.theory.model_igm.T_model.get_sigT_kms(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["sigT_kms"], "--")

# plt.plot(zs, like.theory.model_igm.T_model.get_gamma(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["gamma"], "--")

# plt.plot(zs, like.theory.model_igm.P_model.get_kF_kms(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["kF_kms"], "--")

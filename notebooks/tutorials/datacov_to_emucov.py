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
from cup1d.p1ds.data_DESIY1 import P1D_DESIY1

from cup1d.likelihood.input_pipeline import Args

# %% [markdown]
# ### Set up arguments
#
# Info about these and other arguments in cup1d.likelihood.input_pipeline.py

# %%
# from astropy.io import fits
# folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
# file = "mock_challenge_0.2_nonoise_CGAN_4096_base.fits"
# res = fits.open(folder+file)
# p1d = P1D_DESIY1(fname=folder+fname)

# folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/"

# # p1d = P1D_DESIY1(fname=folder+fname)
# # p1d.plot_p1d()

# %% [markdown]
# ### Set emulator

# %%
# set output directory for this test
output_dir = "."

# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# the nyx emulator has not properly been validated yet
args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Jul2024")

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

# %% [markdown]
# #### Set either mock data or real data

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
# fname = "mock_challenge_0.2_nonoise_bar_ic_grid_3.fits"
# data["P1Ds"] = P1D_DESIY1(fname = folder + fname, true_sim_label="nyx_3")

# %%
choose_forecast = False
choose_mock = False
choose_data = False
choose_challenge = True
folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
fname = "mock_challenge_0.2_nonoise_fiducial.fits"
# fname = "mock_challenge_0.2_nonoise_CGAN_4096_base.fits"
# fname = "mock_challenge_0.2_nonoise_cosmo_grid_3.fits"
# fname = "mock_challenge_0.2_nonoise_bar_ic_grid_3.fits"
# fname = "mock_challenge_0.2_noise-42-0_fiducial.fits"
true_sim_label="nyx_central"
# true_sim_label="nyx_seed"
# true_sim_label="nyx_3"

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
    args.data_label="nyx_seed"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None

    # provide cosmology only to cull the data
    # args.true_cosmo_label="mpg_central"
    args.true_cosmo_label="nyx_central"
    args.true_cosmo_label="nyx_seed"
    
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

if choose_challenge == True:    
    data["P1Ds"] = P1D_DESIY1(
        fname = folder + fname, 
        true_sim_label=true_sim_label
    )
else:
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

# %%

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
args.ic_correction=False

args.emu_cov_factor = 0.02
# args.fid_cosmo_label="mpg_central"
args.fid_cosmo_label="nyx_central"
# args.fid_cosmo_label="nyx_seed"

# args.fid_cosmo_label="nyx_3"
# args.ic_correction=True
# args.fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# IGM
# args.fid_igm_label="mpg_central"
args.fid_igm_label="nyx_central"
# args.fid_igm_label="nyx_seed"
# args.fid_igm_label="nyx_3"
# args.fid_igm_label="nyx_3_1"
if choose_data == False:
    args.igm_priors = "hc"
else:
    args.type_priors = "data"
args.type_priors = "hc"

# contaminants
args.fid_SiIII=[0, -10]
args.fid_SiII=[0, -10]
args.fid_HCD=[0, -6]
args.fid_SN=[0, -4]

# parameters
args.vary_alphas=True
# args.fix_cosmo=False
args.fix_cosmo=True
# args.n_tau=0
# args.n_sigT=0
# args.n_gamma=0
# args.n_kF=0
args.n_tau=2
args.n_sigT=2
args.n_gamma=2
args.n_kF=2
args.n_SiIII = 0
args.n_SiII = 0
args.n_dla=0
args.n_sn=0


free_parameters = set_free_like_parameters(args)
free_parameters

# %% [markdown]
# #TODO 
# - nyx_central_1 included

# %% [markdown]
# ### Set likelihood

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    fid_cosmo,
    free_parameters,
    args,
    data_hires=data["extra_P1Ds"],
    # P_model=P_model
)

# %%
# Cov matrix from data to weight emulator

camb_model = like.theory.cosmo_model_fid["cosmo"].get_new_model(data["P1Ds"].z, [])
M_of_zs = camb_model.get_M_of_zs()

_ = np.argwhere((archive.data[0]['k_Mpc'] > 0) & (archive.data[0]['k_Mpc'] < 4))[:,0]
k_emu = archive.data[0]['k_Mpc'][_]
yextra = 100

std_out = np.zeros((len(data["P1Ds"].z), k_emu.shape[0]))
for ii in range(len(data["P1Ds"].z)):
    rat = np.sqrt(np.diag(data["P1Ds"].cov_Pk_kms[ii]))/data["P1Ds"].Pk_kms[ii]
    k_Mpc = data["P1Ds"].k_kms[ii]* M_of_zs[ii]
    std_out[ii, :] = np.interp(k_emu, k_Mpc, rat, left=yextra, right=yextra)
    plt.plot(k_Mpc, rat, label=str(data["P1Ds"].z[ii]))
plt.yscale("log")
plt.legend()

dout = {
    "z": np.array(data["P1Ds"].z),
    "k_Mpc": k_Mpc,
    "rel_error_Mpc": std_out,
}
np.save("rel_error_DESIY1.npy", dout)

# %%

# %%

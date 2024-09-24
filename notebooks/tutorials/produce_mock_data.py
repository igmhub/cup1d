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
# # Tutorial: produce mock P1D data

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

# %% [markdown]
# ## Set emulator

# %%
# set output directory for this test
output_dir = "."

# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
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

# %% [markdown]
# ## Set data

# %%
# for forecast, just start label of observational data with mock
args.data_label="mock_Karacayli2024"

# IGM history to be used
args.true_igm_label="mpg_central"
# You can use the one from any simulation in the suite
# args.true_igm_label="mpg_0"

# fiducial cosmology:
# create your own
true_cosmo = camb_cosmo.get_cosmology(
    H0=67,
    mnu=0,
    omch2=0.12,
    ombh2=0.022,
    omk=0,
    As=2.1e-9,
    ns=0.965,
    nrun=0
)
# or use one from one of the simulations
# true_cosmo = set_cosmo(cosmo_label="mpg_central")

data = {"P1Ds": None}
data["P1Ds"] = set_P1D(
    args.data_label,
    args,
    archive=archive,
    emulator=emulator,
    true_cosmo=true_cosmo,
)

# %% [markdown]
# ### Done!

# %%
data["P1Ds"].plot_p1d()

# %% [markdown]
# #### You can directly access the data via

# %%
zs = data["P1Ds"].z
k_kms = data["P1Ds"].k_kms
Pk_kms = data["P1Ds"].Pk_kms

# %% [markdown]
# ### The value of the cosmology, linP parameters, IGM history and cont is stored in truth

# %%
data["P1Ds"].truth

# %%

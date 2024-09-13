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

# our own modules
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood.sampler_pipeline import set_archive, set_P1D, set_P1D_hires, set_like
from cup1d.likelihood.input_pipeline import Args

# %% [markdown]
# ### Set up arguments
#
# Info about these and other arguments in cup1d.likelihood.input_pipeline.py

# %%
# set output directory for this test
output_dir = "."

# provide name of emulator that we will use as input to generate mock data
args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")

# the wavelengths and covariance matrices are extracted from here 
args.data_label="mock_Karacayli2024"
# maximum redshift
args.z_max=4.5

# fiducial cosmology
args.cosmo_label="mpg_central"
#fiducial IGM history
args.igm_label="mpg_central"
# number of free parameters scaling the fiducial IGM history
args.n_igm=2
# add metal contamination
args.add_metals=False

# %% [markdown]
# ### Set archive

# %%
archive = set_archive(args.training_set)

# %% [markdown]
# ### Set emulator
#
# The emulator we use to generate mock data

# %%
emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

# %% [markdown]
# ### <font color='red'>Provide here the cosmology of your choice</font>
#
# We modify IGM parameters later

# %%
cosmo_fid = camb_cosmo.get_cosmology(
    H0=67,
    mnu=0,
    omch2=0.12,
    ombh2=0.022,
    omk=0,
    As=2.1e-9,
    ns=0.965,
    nrun=0
)

# %% [markdown]
# ### Set P1D data
#
# Get P1D data holder

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
# ### Create mock P1D
#
# Once everything is set, we can modify the free parameters of cup1d to create mock P1D data. We can modify all IGM parameters. Of the cosmological parameters, we can only modify As and ns at this point. 

# %% [markdown]
# Print default value of free parameters and prior range:

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %% [markdown]
# ### <font color='red'>Provide here IGM parameters of your choice</font>
#
# In the following example, we modify ln_tau_0, and ln_tau_1

# %%
change_params = {}
change_params["ln_tau_0"] = 0.1
change_params["ln_tau_1"] = 0.2

new_params = np.zeros(len(like.free_params))
for ip, p in enumerate(like.free_params):
    pnew = p.get_new_parameter(0.5)
    if(p.name in change_params):
        pnew.set_without_cube(change_params[p.name])
    else:
        pnew.set_without_cube(p.value)
    print(pnew.name, pnew.value, pnew.min_value, pnew.max_value)
    new_params[ip] = pnew.value_in_cube()

# %% [markdown]
# ### Set new P1Ds

# %%
p1ds, blob, emu_params = like.get_p1d_kms(values=new_params, return_emu_params=True, return_blob=True)

# %%
k_kms_out = data["P1Ds"].k_kms.copy()
p1ds_kms_out = p1ds.copy()

# %% [markdown]
# ### Value of all cup1d parameters used to generate the data
#
# Cup1d put constraints on these parameters

# %%
blob_params = ["Delta2_star", "n_star", "alpha_star"]

for ii in range(len(blob_params)):
    print(blob_params[ii], blob[ii])

for ip, p in enumerate(like.free_params):
    pnew = p.get_new_parameter(0.5)
    pnew.set_from_cube(new_params[ip])
    if(p.name in ["As", "ns"]):
        pass
    else:
        print(p.name, np.round(pnew.value, 5))

# %% [markdown]
# ### Value of emulator parameters
#
# For each redshift, in case it is needed

# %%
emu_params

# %% [markdown]
# ### Plot new P1Ds
#
# The dots show the fiducial P1D, the dashed lines the new ones

# %%
like.plot_p1d(residuals=False, plot_every_iz=1, values=new_params)

# %%

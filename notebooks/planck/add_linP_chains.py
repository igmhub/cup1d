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
# # Add extra column with linear parameters to Planck chains
#
# This notebook shows how to read Planck chain, and compute linear power parameters for each point in the chain.
#
# Actually, the code is fairly slow, so it only does this for a handful of points in the chain. Heavier work is done with a python script in cup1d/scripts/add_linP_chains.py

# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
from getdist import plots
from cup1d.planck import planck_chains
from cup1d.planck import add_linP_params
# because of black magic, getdist needs this strange order of imports
# %matplotlib inline
from cup1d.utils.utils import get_path_repo

# %% [markdown]
# ### Read Planck 2018 chain

# %%
# this should be the original Planck chain, but instead I'm using a lighter version stored in cup1d
# data_type="plikHM_TTTEEE_lowl_lowE"
# cmb = planck_chains.get_planck_2018(model='base_mnu', data=data_type, linP_tag=None)

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")
cmb = planck_chains.get_planck_2018(
    model='base_omegak',
    data='plikHM_TTTEEE_lowl_lowE_BAO',
    root_dir=root_dir,
    linP_tag=None
)

# %%
g = plots.getSinglePlotter()
g.plot_2d(cmb['samples'], ['ns', "omegak"])

# %% jupyter={"outputs_hidden": false}
# dramatically reduce sice of chain, for testing
samples=cmb['samples'].copy()
thinning=1
samples.thin(thinning)
Nsamp,Npar=samples.samples.shape
print('Thinned chains have {} samples and {} parameters'.format(Nsamp,Npar))

# %% [markdown]
# ### For each element in the chain, compute and store linear power parameters

# %% jupyter={"outputs_hidden": false}
# specify linear power parameters that we want to add
# linP_params_names=['Delta2_star','n_star','f_star','g_star','alpha_star']
linP_params_names=['Delta2_star','n_star']
# this will collect a dictionary for each sample in the chain
linP_params_entries=[]
for i in range(Nsamp):
    verbose=(i%10==0)
    if verbose: print('sample point',i)
    # get point from original chain
    params=samples.getParamSampleDict(i)
    # compute linear power parameters (n_star, f_star, etc.)
    linP_params=add_linP_params.get_linP_params(params,verbose=verbose, camb_kmax_Mpc_fast=1.5)
    # add only the relevant ones
    linP_params_entries.append({k: linP_params[k] for k in linP_params_names})
    if verbose: print('linP params',linP_params_entries[-1])

# %% jupyter={"outputs_hidden": false}
# setup numpy arrays with linP parameters
linP_DL2_star=np.array([linP_params_entries[i]['Delta2_star'] for i in range(Nsamp)])
linP_n_star=np.array([linP_params_entries[i]['n_star'] for i in range(Nsamp)])

# %% jupyter={"outputs_hidden": false}
# add new derived linP parameters 
samples.addDerived(linP_DL2_star,'test_linP_DL2_star',label='Ly\\alpha \\, \\Delta_\\ast')
samples.addDerived(linP_n_star,'test_linP_n_star',label='Ly\\alpha \\, n_\\ast')

# %% jupyter={"outputs_hidden": false}
# get basic statistics for the new parameters
param_means=np.mean(samples.samples,axis=0)
param_vars=np.var(samples.samples,axis=0)
print('DL2_star mean = {} +/- {}'.format(param_means[Npar],np.sqrt(param_vars[Npar])))
print('n_star mean = {} +/- {}'.format(param_means[Npar+1],np.sqrt(param_vars[Npar+1])))

# %% [markdown]
# ### Write extended chains to file

# %% jupyter={"outputs_hidden": false}
new_root='./test_linP'
if (thinning > 1.0):
    new_root+='_'+str(thinning)
print('new root',new_root)
samples.saveAsText(root=new_root,make_dirs=True)

# %% jupyter={"outputs_hidden": false}
# Try reading the new file
from getdist import loadMCSamples
key_model = "base"
key_data = "plikHM_TTTEEE_lowl_lowE"
new_root = os.path.join(
        root_dir,
        "COM_CosmoParams_fullGrid_R3.01",
        key_model,
        key_data + "_linP",
    )
new_samples = loadMCSamples(new_root + "/")
# get basic statistics for the new parameters
new_param_means=np.mean(new_samples.samples,axis=0)
new_param_vars=np.var(new_samples.samples,axis=0)
print('old DL2_star mean = {} +/- {}'.format(param_means[Npar],np.sqrt(param_vars[Npar])))
print('new DL2_star mean = {} +/- {}'.format(new_param_means[Npar],np.sqrt(new_param_vars[Npar])))

# %% jupyter={"outputs_hidden": true}

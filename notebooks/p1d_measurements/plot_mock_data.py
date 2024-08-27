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
# # Create fake data from a given model and covariance

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 160
mpl.rcParams['figure.dpi'] = 160
from cup1d.data import data_Chabanier2019
from cup1d.data import data_Karacayli2022
from cup1d.data import data_QMLE_Ohio
from cup1d.data import mock_data
from lace.emulator import nn_emulator

# %%
# setup data to mimic, with a maximum z to avoid redshifts not emulated
#data_label="Chabanier2019"
data_label="QMLE_Ohio"
zmin=2.0
zmax=4.5
if data_label=="Chabanier2019":
    data=data_Chabanier2019.P1D_Chabanier2019(zmin=zmin,zmax=zmax)
elif data_label=="Karacayli2022":
    data=data_Karacayli2022.P1D_Karacayli2022(zmin=zmin,zmax=zmax)
elif data_label=="QMLE_Ohio":
    data=data_QMLE_Ohio.P1D_QMLE_Ohio(zmin=zmin,zmax=zmax)

# %%
emu_params=['Delta2_p', 'n_p','mF', 'sigT_Mpc', 'gamma', 'kF_Mpc']
training_set = 'Cabayol23'
if(training_set):
    emulator_label = "Cabayol23"
    emu_path = "NNmodels/Cabayol23/Cabayol23.pt"
    drop_sim_val = None
    pre_trained = True
    use_GP=False
    
if(pre_trained):
    emulator = nn_emulator.NNEmulator(
        training_set=training_set,
        emulator_label=emulator_label,
        emu_params=emu_params,
        model_path=emu_path,
        drop_sim=drop_sim_val,
        train=False,
    )    
else:
    if use_GP:
        emulator=gp_emulator.GPEmulator(training_set=training_set,emu_params=emu_params)
    else:
        # these might be sub-optimal settings for the Nyx emulator
        emulator=nn_emulator.NNEmulator(training_set=training_set,emu_params=emu_params)

# %%
mock=mock_data.Mock_P1D(data_label=data_label, emulator=emulator, zmin=zmin, zmax=zmax)

# %%
keys = ['label','marker','data'] 
datasets = [dict(zip(keys,['data','*',data])),
            dict(zip(keys,['mock','.',mock]))]


# %%
def combined_plot(datasets,zmin=1.7,zmax=6.0,kmin=0.001,kmax=0.1):
    Ndata=len(datasets)
    for idata in range(Ndata):
        label=datasets[idata]['label']
        marker=datasets[idata]['marker']
        data=datasets[idata]['data']
        k_kms=data.k_kms
        kplot=(k_kms>kmin) & (k_kms<kmax)
        k_kms=k_kms[kplot]
        zs=data.z
        Nz=len(zs)
        for iz in range(Nz):
            z=zs[iz]
            if z < zmin: continue
            if z > zmax: continue
            Pk_kms=data.get_Pk_iz(iz)[kplot]
            err_Pk_kms=np.sqrt(np.diagonal(data.get_cov_iz(iz)))[kplot]    
            fact=k_kms/np.pi
            plt.errorbar(k_kms,fact*Pk_kms,
                         marker=marker,ms=4.5,ls="none",
                         yerr=fact*err_Pk_kms,
                         label=label+' z = {}'.format(z))
    plt.legend()
    plt.yscale('log', nonpositive='clip')
    plt.xscale('log')
    plt.ylabel(r'$k P(k)/ \pi$')


# %% jupyter={"outputs_hidden": false}
combined_plot(datasets,zmin=2.05,zmax=2.25,kmax=0.05)

# %% jupyter={"outputs_hidden": false}
combined_plot(datasets,zmin=2.9,zmax=3.1,kmax=0.05)

# %% jupyter={"outputs_hidden": false}
combined_plot(datasets,zmin=3.9,zmax=4.1,kmax=0.05)

# %%

# %%

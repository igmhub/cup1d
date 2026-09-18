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
# # Create fake data from a given simulation and covariance
#
# Plot mock data from central Gadget sim, and from fiducial Nyx sim

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 160
mpl.rcParams['figure.dpi'] = 160
from lace.archive import gadget_archive
from lace.archive import nyx_archive
from cup1d.data import data_Chabanier2019
from cup1d.data import data_Karacayli2022
from cup1d.data import data_gadget
from cup1d.data import data_nyx

# %%
# setup data to mimic, with a maximum z to avoid redshifts not emulated
data_label="Chabanier2019"
zmin=2.0
zmax=4.5
if data_label=="Chabanier2019":
    data=data_Chabanier2019.P1D_Chabanier2019(zmin=zmin,zmax=zmax)
elif data_label=="Karacayli2022":
    data=data_Karacayli2022.P1D_Karacayli2022(zmin=zmin,zmax=zmax)

# %%
mpg_archive=gadget_archive.GadgetArchive(postproc='Cabayol23')

# %%
nyx_archive=nyx_archive.NyxArchive(verbose=True)

# %%
sim_label="central"
mpg_testing = mpg_archive.get_testing_data(sim_label="mpg_"+str(sim_label))
mpg_mock=data_gadget.Gadget_P1D(mpg_testing,input_sim="mpg_"+str(sim_label),
                                z_max=zmax,data_cov_label=data_label)
nyx_testing = nyx_archive.get_testing_data(sim_label="nyx_"+str(sim_label))
nyx_mock=data_nyx.Nyx_P1D(nyx_testing,input_sim="nyx_"+str(sim_label),
                                z_max=zmax,data_cov_label=data_label)

# %%
keys = ['label','marker','data'] 
datasets = [dict(zip(keys,['mpg_mock','*',mpg_mock])),
            dict(zip(keys,['nyx_mock','.',nyx_mock]))]


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
combined_plot(datasets,zmin=2.15,zmax=2.3,kmax=0.05)

# %% jupyter={"outputs_hidden": false}
combined_plot(datasets,zmin=2.9,zmax=3.1,kmax=0.05)

# %% jupyter={"outputs_hidden": false}
combined_plot(datasets,zmin=3.9,zmax=4.1,kmax=0.05)

# %%
mpg_mock.plot_p1d()

# %%

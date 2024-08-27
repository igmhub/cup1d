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
#     display_name: cup1d
#     language: python
#     name: cup1d
# ---

# %% [markdown]
# # Compare different P1D measurements

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 160
mpl.rcParams['figure.dpi'] = 160
from cup1d.p1ds import (
    data_Irsic2017,
    data_Walther2018,
    data_Chabanier2019,
    data_Karacayli2022,
    data_Ravoux2023,
    data_Karacayli2024,
)

# %% [markdown]
# ### Read all P1D measurements

# %%
keys = ['label','marker', 'color', 'data'] 
datasets = [dict(zip(keys,['Irsic17','o', 'C0',data_Irsic2017.P1D_Irsic2017()])),
            dict(zip(keys,['Walther18','x', 'C1',data_Walther2018.P1D_Walther2018()])),
            dict(zip(keys,['Chabanier19','*', 'C2',data_Chabanier2019.P1D_Chabanier2019()])),
            dict(zip(keys,['Karacayli22','^', 'C3',data_Karacayli2022.P1D_Karacayli2022()])),
            dict(zip(keys,['Karacayli23','v', 'C4',data_Karacayli2024.P1D_Karacayli2024(z_min=0)])),
            dict(zip(keys,['Ravoux23','s', 'C5',data_Ravoux2023.P1D_Ravoux2023()]))]


# %%
def combined_plot(datasets,zmin=1.7,zmax=6.0,kmin=0.001,kmax=0.1):
    Ndata=len(datasets)
    for idata in range(Ndata):
        label=datasets[idata]['label']
        marker=datasets[idata]['marker']
        color=datasets[idata]['color']
        data=datasets[idata]['data']
        zs=data.z
        Nz=len(zs)
        for iz in range(Nz):
            z=zs[iz]
            if z < zmin: continue
            if z > zmax: continue
            k_kms=data.k_kms[iz]
            kplot=(k_kms>kmin) & (k_kms<kmax)
            k_kms=k_kms[kplot]
            Pk_kms=data.get_Pk_iz(iz)[kplot]
            err_Pk_kms=np.sqrt(np.diagonal(data.get_cov_iz(iz)))[kplot]
            fact=k_kms/np.pi
            # to be fixed
            _ = np.isfinite(err_Pk_kms) & (err_Pk_kms < 1e6)
            plt.errorbar(k_kms[_],(fact*Pk_kms)[_],
                         marker=marker,ms=4.5,ls="none",
                         c=color,
                         yerr=(fact*err_Pk_kms)[_],
                         label=label+' z = {}'.format(z), alpha=0.7)
    plt.legend()
    plt.yscale('log', nonpositive='clip')
    plt.xscale('log')
    plt.ylabel(r'$k P(k)/ \pi$')


# %%
kmax=0.08
for z in [2.0, 2.2, 2.6, 3.0, 3.4, 4.0]:
    plt.figure()
    combined_plot(datasets,zmin=z-0.05,zmax=z+0.05,kmax=kmax)
    

# %% [markdown]
# ### Focus on the latest results
#
# Figure 8 of Karaçayli et al. (2022) compares the results of this paper with Chabanier2019, and Karacayli22 at several redshifts. We plot them here.

# %%
zs = data_Karacayli2022.P1D_Karacayli2022().z.copy()
zs = zs[zs <= 3.8]
for z in zs:
    combined_plot(datasets,zmin=z-0.01,zmax=z+0.01,kmax=0.04)
    plt.show()

# %%

# %%

# %%

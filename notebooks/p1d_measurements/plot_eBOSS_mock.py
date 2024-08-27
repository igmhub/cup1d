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
# # Plot P1D fid eBOSS mock from Monte Python

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 160
mpl.rcParams['figure.dpi'] = 160
from cup1d.data.data_eBOSS_mock import P1D_eBOSS_mock
from cup1d.data.data_Chabanier2019 import P1D_Chabanier2019

# %% jupyter={"outputs_hidden": false}
eBOSS_mock = P1D_eBOSS_mock()
eBOSS_mock.plot_p1d()

# %% [markdown]
# #### with noise

# %%
eBOSS_mock_noise = P1D_eBOSS_mock(add_noise=True)
eBOSS_mock_noise.plot_p1d()

# %% [markdown]
# ### Compare to P1D Chabanier et al. (2019)

# %% jupyter={"outputs_hidden": false}
Cha2019=P1D_Chabanier2019(add_syst=True)
Cha2019.plot_p1d()

# %%
for ii in range(len(eBOSS_mock.z)):
    rat_err = np.diag(eBOSS_mock.cov_Pk_kms[ii])/np.diag(Cha2019.cov_Pk_kms[ii])
    plt.plot(eBOSS_mock.k_kms, rat_err, label=r'$z$='+str(eBOSS_mock.z[ii]))

plt.xlabel(r'$k$ [s/km]')
plt.ylabel(r'$\sigma_{P_1}/\sigma_{P_2}$')
plt.legend()

# %%

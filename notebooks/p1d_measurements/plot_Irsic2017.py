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
# # Plot P1D measurement from Irsic et al. (2017)

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.dpi'] = 120
from cup1d.data import data_Irsic2017

# %% [markdown]
# ### Read P1D from Irsic et al. (2017)

# %% jupyter={"outputs_hidden": false}
data=data_Irsic2017.P1D_Irsic2017()
data.plot_p1d()

# %% [markdown]
# ### Using the full covariance matrix
#
# Note that Irsic et al. (2017) actually provide the correlation between different z bins, but our code is not ready to ingest this type of covariance matrix. I have opened a GitHub issue.

# %% jupyter={"outputs_hidden": false}
import os
assert ('CUP1D_PATH' in os.environ),'You need to define CUP1D_PATH'
basedir=os.environ['CUP1D_PATH']+'/data_files/p1d_measurements/Irsic2017/'
cov_file=basedir+'/cov_pk_xs_final.txt'

# %%
inA,inB,inCov=np.loadtxt(cov_file,unpack=True)

# %% jupyter={"outputs_hidden": false}
plt.imshow(inCov.reshape(133,133),vmin=-10,vmax=10)
plt.colorbar()

# %%

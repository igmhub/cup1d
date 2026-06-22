# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Irsic et al. (2017)

# %% jupyter={"outputs_hidden": false}
from cup1d.p1ds.observations import data_Irsic2017

data = data_Irsic2017.P1D_Irsic2017()
data.plot_p1d()

# %% [markdown]
# ### Using the full covariance matrix
#
# Note that Irsic et al. (2017) actually provide the correlation between different z bins, we have not implemented to read it yet

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

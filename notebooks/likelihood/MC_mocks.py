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
# # MC mocks

# %%
import numpy as np
import matplotlib.pyplot as plt 

# %%
local_computer = "/home/jchaves/Proyectos/projects/lya/data/cup1d/sampler/"
base_folder = "v3/emu_Pedersen23_ext/cov_Chabanier2019/mock_mpg_central_igm_mpg_central_cosmo_mpg_central_nigm_2_smooth_noise_"

# %%
nsam = 15
best = np.zeros((nsam, 2))
err = np.zeros((nsam, 2))
for ii in range(nsam):
    path = local_computer + base_folder + str(ii)+"/chain_1/results.npy"
    dat = np.load(path, allow_pickle=True).item()
    best[ii] = dat['param_percen'][-2:, 1]
    err[ii] = 0.5*(dat['param_percen'][-2:, 2] - dat['param_percen'][-2:, 0])

# %%
mc_err = np.std(best, axis=0)
pred_err = np.median(err, axis=0)
print(mc_err)
print(pred_err)
print(mc_err/pred_err-1)

# %%
fig, ax = plt.subplots(1,2, sharey=True)
for ii in range(2):
    ax[ii].hist(err[:,ii]);
    ax[ii].axvline(mc_err[ii], c="C1")
ax[0].set_ylabel("Histogram")
ax[0].set_xlabel(r"Error on $\Delta^2_*$")
ax[1].set_xlabel(r"Error on $n_*$")
folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/sampler/v3/figs/png/"
plt.tight_layout()
plt.savefig(folder+"mc_mocks.png")

# %% [markdown]
# ### cov matrix well estimated?

# %% [markdown]
# problem:
# - drop path ghost
# - return mle in real space, not in data space
# - more sample to have better errors

# %% [markdown]
#

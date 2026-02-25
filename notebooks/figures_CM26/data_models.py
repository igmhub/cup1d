# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Extract best-fitting models
#
# In this notebook, we extract the different terms of the best-fitting model to DESI DR1

# +
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import cup1d
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib import colormaps

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# -

args = Args(pre_defined="CM2026", system="local")
pip = Pipeline(args, out_folder=None)

# +
# my local machine
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"

fname = os.path.join(folder, "chain.npy")
chain = np.array(np.load(fname))
chain = chain.reshape(-1, 53)

data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]

# +

free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)
# -

# #### Extract measurements

out_data = {}
out_data["zs"] = pip.fitter.like.data.z
out_data["k_kms"] = pip.fitter.like.data.k_kms
out_data["data_Pk_kms"] = pip.fitter.like.data.Pk_kms
out_data["cov_Pk_kms"] = pip.fitter.like.cov_Pk_kms
out_data["full_cov_Pk_kms"] = pip.fitter.like.full_cov_Pk_kms

nn = 10000
ind = np.random.permutation(np.arange(chain.shape[0]))[:nn]

chain_res = {
    "p1d_kms": np.zeros((nn, len(out_data["k_kms"]), len(out_data["k_kms"][-1]))),
    "p1d_emu_kms": np.zeros((nn, len(out_data["k_kms"]), len(out_data["k_kms"][-1]))),
    "C_mul_metals": np.zeros((nn, len(out_data["k_kms"]), len(out_data["k_kms"][-1]))),
    "C_add_metals": np.zeros((nn, len(out_data["k_kms"]), len(out_data["k_kms"][-1]))),
    "C_HCD": np.zeros((nn, len(out_data["k_kms"]), len(out_data["k_kms"][-1]))),
    "C_res": np.zeros((nn, len(out_data["k_kms"]), len(out_data["k_kms"][-1]))),
}

for ii in range(ind.shape[0]):
# for ii in range(100):
    if ii % 100 == 0:
        print(ii)

    like_params = pip.fitter.like.parameters_from_sampling_point(chain[ind[ii], :])
    results = pip.fitter.like.theory.get_p1d_kms(
        pip.fitter.like.data.z,
        pip.fitter.like.data.k_kms,
        like_params=like_params,
        return_blob=False,
        return_contaminants=True
    )

    p1d_tot = results[0]
    p1d_conts = results[1]
    
    for jj in range(len(p1d_tot)):
        nelem = len(p1d_tot[jj])
        chain_res["p1d_kms"][ii, jj, :nelem] = p1d_tot[jj]
        for par in p1d_conts[0].keys():
            if par in ["z", "p1d_tot_kms", "k_kms"]:
                continue
            chain_res[par][ii, jj, :nelem] = p1d_conts[jj][par]

# +
out_data['model_p1d_kms'] = []
out_data['lya_p1d_kms'] = []
out_data['C_mul_metals'] = []
out_data['C_add_metals'] = []
out_data['C_HCD'] = []
out_data['C_res'] = []

out_data['err_model_p1d_kms'] = []
out_data['err_lya_p1d_kms'] = []
out_data['err_C_mul_metals'] = []
out_data['err_C_add_metals'] = []
out_data['err_C_HCD'] = []
out_data['err_C_res'] = []

for jj in range(chain_res["C_HCD"].shape[1]):
    _ = chain_res["p1d_kms"][0, jj, :] != 0
    out_data["model_p1d_kms"].append(np.mean(chain_res["p1d_kms"][:, jj, _], axis=0))
    out_data["err_model_p1d_kms"].append(np.std(chain_res["p1d_kms"][:, jj, _], axis=0))
    
    out_data["lya_p1d_kms"].append(np.mean(chain_res["p1d_emu_kms"][:, jj, _], axis=0))
    out_data["err_lya_p1d_kms"].append(np.std(chain_res["p1d_emu_kms"][:, jj, _], axis=0))
    
    out_data["C_mul_metals"].append(np.mean(chain_res["C_mul_metals"][:, jj, _], axis=0))
    out_data["err_C_mul_metals"].append(np.std(chain_res["C_mul_metals"][:, jj, _], axis=0))
    
    out_data["C_add_metals"].append(np.mean(chain_res["C_add_metals"][:, jj, _], axis=0))
    out_data["err_C_add_metals"].append(np.std(chain_res["C_add_metals"][:, jj, _], axis=0))
    
    out_data["C_HCD"].append(np.mean(chain_res["C_HCD"][:, jj, _], axis=0))
    out_data["err_C_HCD"].append(np.std(chain_res["C_HCD"][:, jj, _], axis=0))
    
    out_data["C_res"].append(np.mean(chain_res["C_res"][:, jj, _], axis=0))
    out_data["err_C_res"].append(np.std(chain_res["C_res"][:, jj, _], axis=0))

# +
out_data["README"] = "model_p1d_kms = [(C_mul_metals * C_HCD * lya_p1d_kms + C_add_metals) * C_res]"

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "data_model_contaminants_systematics.npy")
np.save(fname, out_data)
# -

out_data = np.load(fname, allow_pickle=True).item()
out_data.keys()

out_data["README"]

# ## Figures

Nz = len(out_data["zs"])
fig, ax = plt.subplots(Nz, figsize=(12, Nz*2), sharex=True, sharey=True)
for ii in range(Nz):
    # par = "C_res"
    # par = "C_HCD"
    par = "C_add_metals"
    # par = "C_mul_metals"
    # par = "lya_p1d_kms"
    # par = "model_p1d_kms"
    if par == "C_add_metals":
        y0 = 0
    else:
        y0 = 1
    ax[ii].axhline(y0, color="k", ls=":")
    ax[ii].errorbar(out_data["k_kms"][ii], out_data[par][ii], out_data["err_" + par][ii])
    ax[ii].set_ylabel(out_data["zs"][ii])

Nz = len(out_data["zs"])
fig, ax = plt.subplots(Nz, figsize=(12, Nz*2), sharex=True)
for ii in range(Nz):
    k = out_data["k_kms"][ii]
    data_p1d = out_data["data_Pk_kms"][ii]
    err_p1d = np.sqrt(np.diag(out_data["cov_Pk_kms"][ii]))
    mod_p1d = out_data["model_p1d_kms"][ii]
    lya_p1d = out_data["lya_p1d_kms"][ii]
    
    ax[ii].axhline(1, color="k", ls=":")
    ax[ii].errorbar(k, data_p1d/lya_p1d, err_p1d/lya_p1d)
    ax[ii].fill_between(
        k, 
        (-0.5 * out_data["err_model_p1d_kms"][ii] + mod_p1d)/lya_p1d, 
        (0.5 * out_data["err_model_p1d_kms"][ii] + mod_p1d)/lya_p1d,
        color="C1",
        alpha=0.5
    )
    ax[ii].set_ylabel(out_data["zs"][ii])

# +
fig, ax = plt.subplots(figsize=(12, 10))
fontsize=20

for ii in range(len(out_data["k_kms"])):
    k = out_data["k_kms"][ii]
    data_p1d = out_data["data_Pk_kms"][ii]
    err_p1d = np.sqrt(np.diag(out_data["cov_Pk_kms"][ii]))
    fact = k/np.pi
    ax.errorbar(k, fact*data_p1d, fact*err_p1d, linestyle=":", alpha=0.8, color=colormaps["tab20"].colors[ii])

    _ = chain_res["p1d_kms"][0,ii,:] != 0
    p1d_model = fact*np.percentile(chain_res["p1d_kms"][:, ii, _], [16, 84], axis=0)
    ax.fill_between(k, p1d_model[0], p1d_model[1], color=colormaps["tab20"].colors[ii], alpha=0.5, label=r"$z=$"+str(out_data["zs"][ii]))
    

plt.legend(ncol=3, loc="lower right", fontsize=fontsize)
ax.tick_params(axis="both", which="major", labelsize=fontsize)
ax.set_yscale("log")
ax.set_xlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{\pi}^{-1}k_\parallel\,P(k)$", fontsize=fontsize)
# plt.xscale("log")
plt.tight_layout()
plt.savefig("figs/fig_with_model.pdf")
plt.savefig("figs/fig_with_model.png")
# -







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
# # Tutorial: sample data
#
# This notebook shows an illustrative example of how to run cup1d for extracting cosmological constraints from P1D data:
#
# - Set mock data
# - Set emulator
# - Set likelihood
# - Set sampler
# - Run sample for a small number of steps
#
# All these steps are implemented in cup1d/cup1d/likelihood/samplerpipeline.py. If you are interested in running cup1d, please take a look at cup1d/scripts/sam_sim.py. That script is parallelized using MPI and includes a bookkeeper taking care of all relevant options.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood import lya_theory, likelihood
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter

from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline

from cup1d.p1ds.data_DESIY1 import P1D_DESIY1
from astropy.io import fits


from corner import corner

from cup1d.utils.utils import get_path_repo

from scipy.stats import chi2 as chi2_scipy
from cup1d.pipeline.set_archive import set_archive


# %%
# from cup1d.planck import planck_chains
# from getdist import plots

# %%
# root_dir = "/home/jchaves/Proyectos/projects/lya/cup1d/data/cmbspa_linP_chains/"
# # root_dir = "/pscratch/sd/j/jjchaves/data/cmbspa_linP_chains/"
# key_model = "base_mnu"
# key_data = "DESI_CMB-SPA"
# key_models = ["base_mnu", "base_nnu", "base_nrun", "base_nrunrun"]

# new_samples = []
# for key_model in key_models:
#     cmb_lite = planck_chains.get_cobaya(
#         root_dir=root_dir,
#         model=key_model,
#         data=key_data,
#         lite=True,
#         linP_tag=None,
#     )
#     new_samples.append(cmb_lite["samples"])

# arr_plot = ['As','ns', "mnu"]

# g = plots.getSinglePlotter(width_inch=10)
# g.settings.num_plot_contours = 2

# for ii in range(len(new_samples)):
#     print(key_models[ii])
#     g.plot_2d(
#         new_samples[ii], 
#         ['H0', 'ns'], 
#         colors=["C"+str(ii)], 
#         lws=2, 
#         alphas=[0.8, 0.5],
#         filled=False,
#         label=key_models[ii]
#     )
# plt.legend()

# %%
# zzs = np.array([2, 2.44, 3.01, 3.49, 4.43])
# alphas = np.array([0.106, 0.149, 0.218, 0.27, 0.366])

# ((1 + zzs)/3)**-3.55 * alphas

# %%

# base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
# folder = "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"

# from cup1d.plots_and_tables.table_nuisance import table_nuisance
# table_nuisance(base + folder)

# from cup1d.plots_and_tables.table_variations import table_variations
# base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# table_variations(base)


# from cup1d.plots_and_tables.plot_table_igm import plot_table_igm
# base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# # save_fig = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/figs/"
# # store_data = plot_table_igm(base, name_variation=None, save_fig=save_fig, chain="7", store_data=True)
# save_fig = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/figs/test/"
# store_data = plot_table_igm(base, name_variation="nyx", save_fig=save_fig, chain="3", store_data=True)
# plot_table_igm(base, name_variation="more_igm", save_fig=save_fig, chain="2")


# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_19.npy")
np.save(fname, store_data)

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_C1.npy")
np.save(fname, store_data)

# %%

# %%
fig 17, 20d, A1, A2, A3

# %%
4400/1216 -1

# %%
5800/1216 -1

# %%

# %%
from cup1d.plots_and_tables.plots_corner import plots_chain

# %%
base_notebook = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/"
real_blinding = np.load(base_notebook + "blinding.npy", allow_pickle=True).item()


# %%
real_blinding

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"

variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "zmax": ["Data: $z \\leq 3.4$", "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"],
    "zmin": ["Data: $z \\geq 2.6$", "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE_mpg": ["Data: w/ low SNR", "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "DESIY1_FFT3_dir_mpg": ["Data: FFT", "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    # "DESIY1_FFT_dir_mpg": ["Data: FFT w/ low SNR", "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    
    "no_emu_cov": ["Cov: w/o emu err", "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate": ["Cov: w/o 5\% err", "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate_no_emu_cov": ["Cov: w/o emu, 5\% err", "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE3_nyx": ["Model: lace-lyssa", "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/"],
    
    "cosmo": ["Model: $\omega_0\omega_a$CDM", "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_high": ["Model: high $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_low": ["Model: low $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"],
    
    "more_igm": ["Model: IGM $n_z=8$", "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"],
    "less_igm": ["Model: IGM $n_z=4$", "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"],
    "Turner24": ["Model: $\\bar{F}\\, n_z=1$", "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"],
    
    "hcd_z": ["Model: HCD $n_z=2$", "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"],
    "dlas": ["Model: only DLAs", "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"],
    
    "metals_z": ["Model: metals $n_z=2$", "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"],
    "metal_trad": ["Model: trad metal", "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"],
    "metal_thin": ["Model: metal thin", "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"],
    "metal_deco": ["Model: no metal decorr", "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"],
    "metal_si2": ["Model: no SiII-SiII", "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"],
    
    "no_res": ["Model: no resolution", "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"],
    
    # "sim_mpg_central": ["Val: mpg-central simulation", "DESIY1_QMLE3/sim_mpg_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm": ["Val: mpg-central simulation only IGM", "DESIY1_QMLE3/sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm0": ["Val: mpg-central simulation only cosmo", "DESIY1_QMLE3/sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"],
    # "sim_nyx_central": ["Val: nyx-central simulation", "DESIY1_QMLE3/sim_nyx_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_sherwood": ["Val: sherwood simulation", "DESIY1_QMLE3/sim_sherwood/CH24_mpgcen_gpr/chain_1/"],
}

variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"],
}

for ii, var in enumerate(variations):
    folder = os.path.join(base, variations[var][1])
    store_data = plots_chain(folder, store_data=True, truth=real_blinding)

# %%
store_data

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20d.npy")
np.save(fname, store_data)

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_17.npy")
np.save(fname, store_data)

# %%

# %%

# %%

# %%
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "figs11_to_16":"check out the first two columns of Table 5",
}
fname = os.path.join(path_out, "fig_11_to_16.npy")
np.save(fname, store_data)

# %%

# %%
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "figs23_24":"Table 6",
}
fname = os.path.join(path_out, "fig_23_24.npy")
np.save(fname, store_data)

# %%
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "blue":{
        "mu":-0.026,
        "sigma":0.203
    },
    "orange":{
        "A":-0.047,
        "B":0.198
    },
}
fname = os.path.join(path_out, "fig_25.npy")
np.save(fname, store_data)

# %%
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "fig_D1":"Table D1",
}
fname = os.path.join(path_out, "fig_D1.npy")
np.save(fname, store_data)

# %%

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "other_figures.npy")
store_data = {
    "figs11_to_16":"first two columns of Table 5",
    "fig18":"likelihoods in https://github.com/igmhub/cobaya_lya_p1d or check out corresponding papers",
    "figs23_24":"Table 6",
    "figD1": "Table D1",
}
np.save(fname, store_data)

# %%

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
plots_chain(base + folder)
# chain = np.load(base + folder + "chain.npy")
# chain = 0
# blobs = np.load(base + folder + "blobs.npy")
# lnprob = np.load(base + folder + "lnprob.npy")

# mask, other = purge_chains(lnprob)

# for ii in range(other.shape[0]):
# # for ii in range(5):
#     # print(np.median(lnprob[:, other[ii]]))
#     plt.plot(lnprob[:, other[ii]])

# %%
# plt.hist2d(blobs["Delta2_star"].reshape(-1), blobs["n_star"].reshape(-1), bins=50);

# %% [markdown]
# ### Fisher

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"


# data_label = "mock_DESIY1_QMLE3"
# data_label = "nyx_central"
# data_label = "nyx_seed"
# data_label = "nyx_cgan_base"
# data_label = "accel2"
data_label = "sherwood"

# data_label = "mpg_central"
# data_label = "mpg_seed"
# data_label = "nyx_seed"

if data_label == "mpg_central":
    zmin=2.2
    zmax=4.2
elif data_label == "nyx_central":
    zmin=2.2
    zmax=4.2
else:
    zmin=2.2
    zmax=4.2

cov_label="DESIY1_QMLE3"
true_cosmo_label = data_label
fid_cosmo_label = data_label
name_variation= "sim_" + data_label
# name_variation= "sim_" + data_label + "_igm"
# name_variation= "sim_" + data_label + "_igm0"
fit_type = "global_opt"
# name_variation = None

args = Args(
    data_label=data_label,
    cov_label=cov_label,
    emulator_label=emulator_label,
    true_cosmo_label=data_label,
    apply_smoothing=True,
    add_noise=False,
    seed_noise=0,
    emu_cov_type="full",
)

args.set_baseline(
    fit_type=fit_type,
    fix_cosmo=False,
    fid_cosmo_label=data_label,
    P1D_type=cov_label,
    name_variation=name_variation,
    z_min=zmin,
    z_max=zmax,
    mcmc_conf="explore",
)

# %%
name_variation

# %%

# %%

# archive_mock = set_archive(training_set="Pedersen21")
# dat = archive_mock.get_testing_data("mpg_central")

# %%

# %%

# %% [markdown]
# ### Mocks

# %%
# nyx_training_set = "models_Nyx_Sept2025_include_Nyx_fid_rseed"
# archive_mock = set_archive(training_set=nyx_training_set)
# pip = Pipeline(args, archive=archive_mock)
pip = Pipeline(args)

# %%
pip.fitter.like.plot_p1d()

# %%
dict_out = {
    "k_kms": pip.fitter.like.data.k_kms,
    "Pk_kms": pip.fitter.like.data.Pk_kms,
    "cov_Pk_kms": pip.fitter.like.data.cov_Pk_kms,
    "z": pip.fitter.like.data.z,
}
np.save("smooth_" + data_label + ".npy", dict_out)

# %%
data = np.load("smooth_" + data_label + ".npy", allow_pickle=True).item()
data.keys()

# %%
# cov0 = pip.fitter.like.full_cov_Pk_kms.copy()
# kms0 = pip.fitter.like.data.full_k_kms.copy()
# zz0 = pip.fitter.like.data.full_zs.copy()

# cov1 = pip.fitter.like.full_cov_Pk_kms.copy()
# kms1 = pip.fitter.like.data.full_k_kms.copy()
# zz1 = pip.fitter.like.data.full_zs.copy()

# print(cov0.shape)
# print(cov1.shape)
# zz = np.unique(zz0)
# for iz in range(len(zz)):
#     col = "C"+str(iz)
#     ind0 = np.argwhere(zz0 == zz[iz])[:,0]
#     plt.plot(kms0[ind0], np.diag(cov0)[ind0], col, label=np.round(zz[iz], 2), alpha=0.5)

# zz = np.unique(zz1)
# for iz in range(len(zz)):  
#     col = "C"+str(iz)  
#     ind1 = np.argwhere(zz1 == zz[iz])[:,0]
#     plt.plot(kms1[ind1], np.diag(cov1)[ind1], col+"--", label=np.round(zz[iz], 2))
# plt.legend()
# plt.yscale("log")

# %%

# %%

# %%

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
# p0[:] = 0.5
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
for par in free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %%
p0[18:] = 0
p0[-11:] = 0.5

# %%

pip.fitter.like.get_chi2(p0)

# %%
# for par in pip.fitter.like.free_params:
#     print(par.name, par.value, par.min_value, par.max_value)

# %%
pip.fitter.like.plot_p1d(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %% [markdown]
# ### Data

# %%

# args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_nyxcen_gpr")
# args = Args(data_label="DESIY1_FFT_dir", emulator_label="CH24_nyxcen_gpr")
# args.set_baseline(fit_type="andreu2", fix_cosmo=True)

# args.set_baseline(fit_type="global_all", fix_cosmo=True)

# fid 0.18697989940945323
# HCD depend z, 0.13580926057347983
# same num IGM, new fid, 0.1755366851767883

# name_variation = None
# name_variation = "no_res"
# name_variation = "Turner24"
variations = [
    "fid",
    "no_inflate",  # no increase errors for 3, 3.6, and 4
    "all_inflate",
    "cosmo",  # different fiducial cosmo
    "metal_trad",  # 2 params for metals like eBOSS
    "metal_si2",  # no SiII-SiII cont
    "metal_deco",  # no decorrelation metals
    # "metal_thin",  # no desviation from optically-thin limit
    # "no_res",  # no resolution correction
    "Turner24",  # mF from Turner24 with 1 free param to scale
    "more_igm",  # 8 params for IGM evolution
    "less_igm",  # 4 params for IGM evolution
    # "metals_z",  # 2 params for z ev metals
    # "hcd_z",  # 2 params for z ev hcd
]

# metals_z 3.89548175069871

# name_variation = variations[12]
# name_variation = "metals_z"
# name_variation = "all_inflate"
# name_variation = "Turner24"

data_label = "DESIY1_QMLE3"
# data_label = "DESIY1_QMLE"
# data_label = "DESIY1_FFT3_dir"
# name_variation = None
# name_variation = "no_inflate"
# name_variation = "no_emu_cov"
# name_variation = "no_inflate_no_emu_cov"

# name_variation = "metal_deco"
# name_variation = "metal_si2"
# name_variation = "no_res"
# name_variation = "HCD0"
# name_variation = "kF_kms"
# name_variation = "Gaikwad21"
# name_variation = "Gaikwad21T"
# name_variation = "Turner24"

# name_variation = "data_syst_diag"

# emu_cov_type = "block"
# emu_cov_type = "diagonal"
# name_variation = "Metals_Ma2025"
# name_variation = "HCD_BOSS"

# name_variation = "more_igm"
# name_variation = "LLS_nz4"
# name_variation = "IGM_priors"
# name_variation = "bias_eBOSS"

###
name_variation = None
###

emu_cov_type = "full"
# emu_cov_type = "block"
# emu_cov_type = "diagonal"


emulator_label="CH24_mpgcen_gpr"
# emulator_label="CH24_nyxcen_gpr"
# name_variation = "cosmo_h74"
# name_variation = "cosmo"

args = Args(data_label=data_label, emulator_label=emulator_label, emu_cov_type=emu_cov_type)
args.set_baseline(
    fit_type="global_opt", 
    fix_cosmo=False, 
    P1D_type=data_label, 
    name_variation=name_variation, 
)

pip = Pipeline(args, out_folder=args.out_folder)


# %%

# %%
# Fig. 1
store_data = pip.fitter.like.data.plot_p1d(store_data=True)


# %%

import cup1d
path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_1.npy")
np.save(fname, store_data)

# %%
# Fig. 2
store_data = pip.fitter.like.plot_cov_to_pk(use_pk_smooth=False, store_data=True)

# %%

import cup1d
path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_2.npy")
np.save(fname, store_data)

# %%




# %%
# pip.fitter.like.plot_correlation_matrix()

# %%
# np.save("blinding.npy", pip.fitter.like.blind)

# %%
# Pk_kms0 = pip.fitter.like.data.Pk_kms.copy()
Pk_kms1 = pip.fitter.like.data.Pk_kms.copy()

# %%
err_Pk_kms = pip.fitter.like.cov_Pk_kms.copy()

# %%
kk = pip.fitter.like.data.k_kms.copy()

# %%
res = np.zeros((11, 2))
fig, ax = plt.subplots(11, 1, sharex=True, figsize=(8, 15))
for ii in range(len(kk)):
    ax[ii].errorbar(kk[ii], Pk_kms0[ii]/Pk_kms1[ii]-1, np.sqrt(np.diag(err_Pk_kms[ii]))/Pk_kms1[ii])
    ax[ii].plot(kk[ii], kk[ii][:]*0, ":")
    res[ii, 0] = np.median(Pk_kms0[ii]/Pk_kms1[ii]-1)
    res[ii, 1] = np.mean(np.sqrt(np.diag(err_Pk_kms[ii]))/Pk_kms1[ii])
fig.supylabel("Residual")
fig.supxlabel(r"$k_\parallel$")
plt.tight_layout()
plt.savefig("qmle_res_p1d.png")

# %%
plt.errorbar(pip.fitter.like.data.z, res[:,0], res[:,1])
plt.plot(pip.fitter.like.data.z, pip.fitter.like.data.z[:] * 0)

# %%

# %%

# pip.fitter.like.plot_cov_to_pk(fname="figs/err2p1d_qmle3")
# pip.fitter.like.plot_cov_to_pk(use_pk_smooth=False, fname="figs/err2p1d_qmle3")

# %%
for ii, par in enumerate(pip.fitter.like.free_params):
    print(ii, par.name, par.value, par.min_value, par.max_value)

# %%

# %%

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%

pip.fitter.like.theory.model_cont.metal_models["Si_mult"].coeffs

# %%
# cov1
p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
data_lab = "DESIY1_QMLE3"
# data_lab = "DESIY1_FFT3_dir"
fit_type = "global_opt"
# fit_type = "emu_diag"
# fit_type = "emu_block"
emu = "mpg"
# folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_3/"
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_7/"
# folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_2/"
data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
ylims=np.array([
    [0.85, 1.15],
    [0.85, 1.15],
    [0.7, 1.3],
    [0., 2.0],
])

out_data = pip.fitter.like.plot_p1d(
    p0, 
    residuals=True, 
    plot_panels=True, 
    print_chi2=False, 
    fix_cosmo=False,
    ylims=ylims, 
    # plot_fname=None,
    plot_fname="figs/residual_fid_opt_global",
    store_data=True
)

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_9.npy")
np.save(fname, out_data)

# %%

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%
XXX

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
chain = np.load(base + folder + "chain.npy")


# %%
# Fig. 20a
out_data = pip.fitter.like.plot_metal_cont_mult(chain=chain, save_directory="figs", store_data=True)

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20a.npy")
np.save(fname, out_data)

# %%
# Fig. 20b
out_data = pip.fitter.like.plot_metal_cont_add(free_params=free_params, chain=chain, save_directory="figs", store_data=True)

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20b.npy")
np.save(fname, out_data)

# %%
# Fig. 20c
out_data = pip.fitter.like.plot_hcd_cont(p0=p0, chain=chain, save_directory="figs", store_data=True)

# %%
# out_data

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_20c.npy")
np.save(fname, out_data)

# %%

# %%

# %%

# %%
# std_mpg = np.sqrt(np.diag(pip.fitter.like.emu_full_cov_Pk_kms)).copy()
# std_nyx = np.sqrt(np.diag(pip.fitter.like.emu_full_cov_Pk_kms)).copy()
# np.mean(std_nyx/std_mpg)

# %%
# pip.fitter.like.data.plot_p1d()

# pip.fitter.like.data.plot_p1d(cov_ext=pip.fitter.like.cov_Pk_kms, fname="figs/p1d_qmle3")
# pip.fitter.like.plot_p1d()
# pip.fitter.like.plot_cov_to_pk(fname="figs/err2p1d_qmle3")
# pip.fitter.like.plot_cov_to_pk()

# %%

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %%
fname = os.path.join(
    os.path.dirname(get_path_repo("cup1d")), "data", "ics", "mpg_ic_global_red.npy"
)
pip.save_global_ic(fname)

# %%
p0 = pip.fitter.mle_cube
# p0 = data_best["mle_cube"].copy()
pip.fitter.like.get_chi2(p0)

# %%

new baseline, 
657.150798756201
Delta2_star 0.41149
n_star -2.27902


# %%
# pip.fitter.mle

# %%
# pip.fitter.like.theory.model_cont.metal_models["Si_add"].coeffs

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)
# pip.fitter.like.plot_cov_to_pk(fname="figs/nyx_err2p1d_qmle3")

# %%
# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)
pip.fitter.like.plot_p1d(
    p0, 
    residuals=True, 
    plot_panels=True, 
    print_chi2=False, 
    fix_cosmo=False, 
    plot_fname="figs/residual_fid_opt_global"
)

# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)

# %%
len(pip.fitter.like.free_params)

# %%
# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)

# %%
npoints = 0
for ii in range(len(pip.fitter.like.data.z)):
    npoints += len(pip.fitter.like.data.k_kms[ii])
npoints - len(pip.fitter.like.free_param_names)

# %%
# data_lab = "DESIY1_QMLE3"
# fit_type = "global_opt"
# emu = "mpg"
# folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
# data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()


# ii = 556
# type_prof = "prof_2d_deep"
# data_best = np.load(folder + type_prof + "/profile_"+str(ii)+ ".npy", allow_pickle=True).item()

# data_best.keys()

# %%

tar_cosmo = pip.fitter.like.apply_unblinding(data_best["blind_cosmo"])
pip.fitter.like.theory.rescale_fid_cosmo(tar_cosmo)

# %%
p0 = data_cen["mle_cube"].copy()
# p0 = data_best["mle_cube"].copy()
pip.fitter.like.get_chi2(p0)

# %%

# %%
# from cup1d.likelihood.cosmologies import set_cosmo

# cosmos = set_cosmo("mpg_central", return_all=True)

# p1 = np.zeros(len(cosmos))
# p2 = np.zeros(len(cosmos))
# for ii, key in enumerate(cosmos):
#     p1[ii] = cosmos[key]["star_params"]['Delta2_star']
#     p2[ii] = cosmos[key]["star_params"]['n_star']

# plt.scatter(p1, p2)

# print(p1.min(), p1.max())
# print(p2.min(), p2.max())

# %%

# %% [markdown]
# # OLD

# %% [markdown]
# ### Set emulator

# %% [markdown]
# #### Set either mock data or real data

# %%
# for forecast, just start label of observational data with mock
# choose_forecast = True 
choose_forecast = False
# to analyze data from simulations
choose_mock = False
# choose_mock = True
# to analyze data from observations
choose_data = False
# to analyze data from mock challenge
choose_challenge = False
# to analyze data from desiy1
choose_desiy1 = True

if choose_forecast:
    args.data_label_hires = None
    # for forecast, just start label of observational data with mock
    # args.data_label = "mock_Chabanier2019"
    # args.data_label="mock_Karacayli2024"
    # args.data_label_hires = "mock_Karacayli2022"
    args.data_label="mock_challenge_DESIY1"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    args.p1d_fname=folder + fname

    # # you need to provide true cosmology, IGM history, and contaminants
    # true_cosmo = set_cosmo(cosmo_label="nyx_central")
    # args.true_igm_label="nyx_central"
    # true_sim = "nyx_seed"
    true_sim = "nyx_seed_val"
    # true_sim = "sherwood"
    # true_sim = "accel2"
    # true_sim = "nyx_central"
    # true_sim = "mpg_central"
    # true_sim_cosmo = "mpg_central"
    args.mF_model_type="chunks"
    true_cosmo = set_cosmo(cosmo_label=true_sim)
    args.true_label_mF=true_sim
    args.true_label_T=true_sim
    
    # true_sim = "nyx_central"
    args.true_label_kF=true_sim
    
    # true_sim_cosmo = "Planck18_low"
    # args.true_label_kF="kF_both"
    
    # true_cosmo = set_cosmo(cosmo_label="mpg_22")
    # true_cosmo = set_cosmo(cosmo_label="Planck18")
    # args.true_igm_label="nyx_central"
    # args.true_igm_label="mpg_22"
    # from -11 to -4
    # args.true_SiIII=[[0, 0], [-10, -10]]
    # args.true_SiII=[[0, 0], [-10, -10]]
    # # from -7 to 0
    # args.true_HCD=[0, -6]
    # # from -5 to 2
    # args.true_SN=[0, -4]
    # # from -5 to 1.5
    # args.true_AGN=[0, -5]
    args.z_min = 2.1
    args.z_max = 4.3
elif choose_mock:    
    # true_cosmo=None
    # to analyze data from simulations
    true_sim = "nyx_central"
    # true_sim = "nyx_seed"
    # true_sim = "nyx_central"
    # args.data_label = "mock_DESIY1"
    args.true_cosmo_label=true_sim
    args.true_label_mF=true_sim
    # args.mF_model_type="pivot"
    args.mF_model_type="chunks"
    args.true_label_T=true_sim
    args.true_label_kF=true_sim
    # args.data_label="nyx_seed"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None
    # args.apply_smoothing=True
    args.apply_smoothing=False

    # args.cov_label = "DESIY1"
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_Sherwood_2048_40.fits.gz"
    args.cov_fname = folder + fname
    # args.p1d_fname = folder + fname
    # args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"

    # provide cosmology only to cull the data
    # args.true_cosmo_label="mpg_central"
    # args.true_cosmo_label="nyx_central"
    true_cosmo = set_cosmo(cosmo_label=true_sim)
    # args.true_cosmo_label="nyx_seed"

    # you may provide contaminants
    # from 1 to 6, -11 to -4
    # args.true_SiIII=[[0, 0], [2, -10]]
    # args.true_SiII=[[0, 0], [2, -10]]
    # # from -5 to 0
    # args.true_HCD=[0, -4]
    # # from -5 to 2
    # args.true_SN=[0, -4]
    # args.true_AGN=[0, -5]
    args.z_max=4.3
elif choose_data:    
    true_cosmo=None
    args.data_label = "Chabanier2019"
    # args.data_label="Karacayli2024"
    args.data_label_hires = "Karacayli2022"
    args.z_max = 3.9
elif choose_challenge:
    args.data_label = "challenge_DESIY1"
    # version = "1.1qh"
    # version = "1.9fsh"
    version = "1.10qsh"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-"+version+"/"
    fname = "mockchallenge-"+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-"+version+"_fsiiii1.3e-03_fsiii1.1e-03_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-"+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_bar_ic_grid_3.fits.gz"
    # fname = "mockchallenge-0."+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_Sherwood_2048_40.fits.gz"
    args.p1d_fname = folder + fname
    if "fiducial" in args.p1d_fname:
        true_sim_label = "nyx_central"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "CGAN" in args.p1d_fname:
        true_sim_label = "nyx_seed"        
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "grid_3" in args.p1d_fname:
        true_sim_label = "nyx_3"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "Sherwood_2048_40" in args.p1d_fname:
        true_sim_label = "Sherwood_2048_40"
        args.true_label_mF=true_sim_label
        args.true_label_T="nyx_central"
        args.true_label_kF="nyx_central"
    elif "ACCEL2_6144_160" in args.p1d_fname:
        true_sim_label = "ACCEL2_6144_160"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF="nyx_central"
    else:
        true_sim_label = None
    # true_sim_label = "nyx_central"

    true_cosmo = set_cosmo(cosmo_label=true_sim_label)
    
    args.z_min = 2.1
    # args.z_max = 4.3
    args.z_max = 2.7
    # args.z_min = 2.8
    # args.z_max = 3.2
elif choose_desiy1:
    true_cosmo = None
    args.true_igm_label= None
    args.data_label = "DESIY1"
    # args.cov_syst_type = "xred"
    # args.cov_syst_type = "fid"
    args.cov_syst_type = "red"
    folder = "/home/jchaves/Proyectos/projects/lya/data/DESI-DR1/"
    # in NERSC
    # /global/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-baseline/qmle_measurement/DataProducts/
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    
    # args.p1d_fname=folder + "/qmle_measurement/DataProducts/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    args.p1d_fname= folder + "/qmle_measurement/DataProducts/v3/desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    
    # args.p1d_fname = folder + "/fft_measurement/p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"
    
    args.z_min = 2.1
    # args.z_min = 3.5
    args.z_max = 4.3
    # args.z_max = 2.7
    # args.z_max = 3.3
    # args.z_max = 2.9

# you do not need to provide the archive for obs data 
data = {"P1Ds": None, "extra_P1Ds": None}

data["P1Ds"] = set_P1D(
    args,
    archive=archive,
    true_cosmo=true_cosmo,
    emulator=emulator,
    cull_data=False
)
if args.data_label_hires is not None:
    data["extra_P1Ds"] = set_P1D(
        args,
        archive=archive,
        true_cosmo=true_cosmo,
        emulator=emulator,
        cull_data=False
    )

# %%

# %% [markdown]
# ## Read results from chains

# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/fid/chain_5/2.2/"
folder_priors = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/priors/chain_3/2.2/"
file = "fitter_results.npy"
# sampler_data = np.load(folder + file, allow_pickle=True).item()

# %%
plotter = Plotter(fname_chain=folder_data + file, fname_priors=folder_priors + file)
# plotter.fitter.like.plot_p1d(residuals=True)

# %% [markdown]
# ### Normal corner

# %%
# plotter.fitter.like.plot_p1d()
# plotter.save_directory = folder_data
# plotter.save_directory = None
# plotter.plot_corner()

# %% [markdown]
# ### Corner natural units

# %%
plotter.save_directory = folder_data

only_plot = [
    '$\\mathrm{ln}\\,\\tau_0$',
    '$\\mathrm{ln}\\,\\sigma^T_0$',
    '$\\mathrm{ln}\\,\\gamma_0$',
    '$\\mathrm{ln}\\,k^F_0$',
    '$\\mathrm{R}_0$'
]
plotter.plot_corner_1z_natural(2.2, only_plot=only_plot)

# %%

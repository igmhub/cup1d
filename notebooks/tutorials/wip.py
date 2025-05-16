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

from cup1d.likelihood.pipeline import (
    set_archive,
    set_P1D,
    set_cosmo,
    set_free_like_parameters,
    set_like,
    Pipeline,
)
from cup1d.p1ds.data_DESIY1 import P1D_DESIY1
from astropy.io import fits

from cup1d.likelihood.input_pipeline import Args

from corner import corner
from cup1d.likelihood import CAMB_model

from cup1d.utils.utils import get_path_repo

from lace.archive.nyx_archive import NyxArchive

# %%
# from cup1d.likelihood.model_systematics import Systematics
# k_kms = np.logspace(-3, -1, 100)
# syst_model = Systematics(fid_R_coeff=[0,0.01])
# syst_model.get_contamination(z=2, k_kms=k_kms)

# %%

# %%

# %%


# fname_chain = "/home/jchaves/Proyectos/projects/lya/data/obs/CH24_nyx_gpr_emu_err_full/chain_2/fitter_results.npy"
# plotter = Plotter(fname_chain=fname_chain)

# %%
# plotter.plot_corner(only_cosmo=True)

# %%

# %%
nyx_file = "models_Nyx_Mar2025_with_CGAN_val_3axes"
archive = NyxArchive(nyx_version=nyx_file)
# archive = NyxArchive(nyx_version=nyx_file)

# %%
seed_1 = archive.get_testing_data("nyx_seed")
seed_2 = archive.get_testing_data("nyx_seed_val")
cen = archive.get_testing_data("nyx_central")

# %%
# par = "T0"
par = "sigT_Mpc"
# par = "gamma"
# par = "mF"

for ii in range(len(seed_1)):
    plt.scatter(seed_1[ii]["z"], seed_1[ii][par], color="C0")
for ii in range(len(seed_2)):
    plt.scatter(seed_2[ii]["z"], seed_2[ii][par], color="C1")
for ii in range(len(cen)):
    plt.scatter(cen[ii]["z"], cen[ii][par], color="C2")

# %%
mask = (seed[ii]["z"]
p1d_all = np.zeros((nn, 3))
for ii in range(len(test2)):
    print(test1[ii]["z"], test2[ii]["z"])

# %%
test1[ii].keys()

# %%



# %%
p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v2.fits"
# p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v6.fits"
hdu = fits.open(p1d_fname)

# %%
for ii in range(len(hdu)):
    if "EXTNAME" in hdu[ii].header:
        print(ii, hdu[ii].header["EXTNAME"])

# %%
zz = np.unique(hdu[2].data["Z"])
for ii in range(len(zz)):
    ind = hdu[1].data["Z"] == zz[ii]
    plt.plot(hdu[2].data["K"][ind], hdu[2].data["E_RESOLUTION"][ind], label=str(zz[ii]))
plt.legend()

# %%
ind = hdu[1].data["Z"] == 3.6
plt.plot(hdu[1].data["K"][ind], hdu[1].data["PLYA"][ind])
plt.plot(hdu[1].data["K"][ind], hdu[1].data["PSMOOTH"][ind])
plt.plot(hdu[1].data["K"][ind], hdu[1].data["PSMOOTH"][ind]*0.95)
plt.yscale("log")
plt.ylim(1, 1e2)

# %%
# # FFT

# sys_labels = [
#     "E_SYST",
#     "E_PSF",
#     "E_RESOLUTION",
#     "E_SIDE_BAND",
#     "E_LINES",
#     "E_DLA",
#     "E_BAL",
#     "E_CONTINUUM",
#     "E_DLA_COMPLETENESS",
#     "E_BAL_COMPLETENESS",
# ]

# sys_include = [
#     # "E_SYST",
#     "E_PSF",
#     "E_RESOLUTION",
#     "E_SIDE_BAND",
#     "E_LINES",
#     "E_DLA",
#     "E_BAL",
#     "E_CONTINUUM",
#     "E_DLA_COMPLETENESS",
#     "E_BAL_COMPLETENESS",
# ]


# %%
# QMLE

sys_labels = [
    "E_DLA_COMPLETENESS",
    "E_BAL_COMPLETENESS",
    "E_RESOLUTION",
    "E_CONTINUUM",
    "E_CONTINUUM_ADD",
    "E_NOISE_SCALE",
    "E_NOISE_ADD",
    "E_SYST",
]
sys_corr = [
    "E_DLA_COMPLETENESS",
    "E_BAL_COMPLETENESS",
    "E_RESOLUTION",
    "E_CONTINUUM_ADD",
    "E_CONTINUUM",
    "E_NOISE_ADD",
    "E_NOISE_SCALE",
]

sys_diag = [
    # "E_DLA_COMPLETENESS",
    # "E_BAL_COMPLETENESS",
    # "E_RESOLUTION",
    # "E_CONTINUUM",
    # "E_CONTINUUM_ADD",
    # "E_NOISE_ADD",
    # "E_NOISE_SCALE", # goes to diagonal
]

sys_corr_red = [
    # "E_DLA_COMPLETENESS",
    # "E_BAL_COMPLETENESS",
    "E_RESOLUTION",
    "E_CONTINUUM_ADD",
    "E_CONTINUUM",
    "E_NOISE_ADD",
    "E_NOISE_SCALE",
]


sys_diag_red = [
    # "E_DLA_COMPLETENESS",
    # "E_BAL_COMPLETENESS",
    # "E_RESOLUTION",
    # "E_CONTINUUM_ADD",
    # "E_CONTINUUM",
    # "E_NOISE_ADD",
    # "E_NOISE_SCALE",
]
sys_corr_xred = [
    # "E_DLA_COMPLETENESS",
    # "E_BAL_COMPLETENESS",
    "E_RESOLUTION",
    "E_CONTINUUM_ADD",
    "E_CONTINUUM",
    # "E_NOISE_ADD",
    # "E_NOISE_SCALE",
]


sys_diag_xred = [
    # "E_DLA_COMPLETENESS",
    # "E_BAL_COMPLETENESS",
    # "E_RESOLUTION",
    # "E_CONTINUUM_ADD",
    # "E_CONTINUUM",
    # "E_NOISE_ADD",
    # "E_NOISE_SCALE",
]

# %%
diag_emu = np.sqrt(np.diag(dict_save["cov"]))
emu_cov_zz = dict_save["zz"]
emu_cov_k_Mpc = dict_save["k_Mpc"]
emu_cov_unique_zz = np.unique(emu_cov_zz)
emu_cov_k_kms = np.zeros_like(emu_cov_k_Mpc)

for jj in range(len(emu_cov_unique_zz)):
    ind = emu_cov_zz == emu_cov_unique_zz[jj]
    dkms_dMpc = like.theory.fid_cosmo["cosmo"].dkms_dMpc(emu_cov_unique_zz[jj])
    emu_cov_k_kms[ind] = emu_cov_k_Mpc[ind] / dkms_dMpc



diag_emu2 = np.sqrt(np.diag(dict_save2["cov"]))
emu_cov_zz2 = dict_save2["zz"]
emu_cov_k_Mpc2 = dict_save2["k_Mpc"]
emu_cov_unique_zz2 = np.unique(emu_cov_zz2)
emu_cov_k_kms2 = np.zeros_like(emu_cov_k_Mpc2)

for jj in range(len(emu_cov_unique_zz2)):
    ind = emu_cov_zz2 == emu_cov_unique_zz2[jj]
    dkms_dMpc = like.theory.fid_cosmo["cosmo"].dkms_dMpc(emu_cov_unique_zz2[jj])
    emu_cov_k_kms2[ind] = emu_cov_k_Mpc2[ind] / dkms_dMpc


# %%

# %%
cov = compute_cov(hdu[2].data, type_analysis="xred")

# %%

# %%

# %%
zz = np.unique(hdu[2].data["Z"])

fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12, 10))
ax = ax.reshape(-1)

diag_stat = np.sqrt(np.diag(hdu[4].data))

for jj in range(len(zz) - 1):

    ind = (hdu[2].data["Z"] == zz[jj]) & (hdu[2].data["K"] > 1e-3)
    
    diag = np.zeros(np.sum(ind))
    diag_red = np.zeros(np.sum(ind))
    diag_xred = np.zeros(np.sum(ind))
    for lab in sys_labels:
        ax[jj].plot(hdu[2].data["K"][ind], hdu[2].data[lab][ind], label=lab)
        if lab in sys_corr:
            diag += hdu[2].data[lab][ind]**2
        if lab in sys_corr_red:
            diag_red += hdu[2].data[lab][ind]**2
        if lab in sys_corr_xred:
            diag_xred += hdu[2].data[lab][ind]**2
    
    cov = np.outer(np.sqrt(diag), np.sqrt(diag))
    cov_red = np.outer(np.sqrt(diag_red), np.sqrt(diag_red))
    cov_xred = np.outer(np.sqrt(diag_xred), np.sqrt(diag_xred))
    
    for lab in sys_diag:
        cov[np.diag_indices_from(cov)] += hdu[2].data[lab][ind]**2
    for lab in sys_diag_red:
        cov_red[np.diag_indices_from(cov_red)] += hdu[2].data[lab][ind]**2
    for lab in sys_diag_xred:
        cov_xred[np.diag_indices_from(cov_xred)] += hdu[2].data[lab][ind]**2
        
    ax[jj].plot(hdu[2].data["K"][ind], diag_stat[ind], ".-", label="Stat")   

    ax[jj].text(1.2e-3, 10, "z="+str(zz[jj]))

    indz = np.argmin(np.abs(emu_cov_zz - zz[jj]))
    ind2 = (emu_cov_zz == emu_cov_zz[indz]) & (emu_cov_k_kms > 0.5e-3)
    interp_y = np.interp(hdu[2].data["K"][ind], emu_cov_k_kms[ind2], diag_emu[ind2])
    # ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PLYA"][ind], label="Emu")
    ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PSMOOTH"][ind], "C0-.", label="Emu nyx")

    fid = np.sqrt(
        np.sqrt(np.diag(cov))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    # red = np.sqrt(
    #     np.sqrt(np.diag(cov_red))**2 
    #   + diag_stat[ind]**2 
    #   + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    # )
    xred1 = np.sqrt(
        np.sqrt(np.diag(cov_xred))**2 
      + diag_stat[ind]**2 
      # + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    xred2 = np.sqrt(
        np.sqrt(np.diag(cov_xred))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )

    
    indz = np.argmin(np.abs(emu_cov_zz2 - zz[jj]))
    ind2 = (emu_cov_zz2 == emu_cov_zz2[indz]) & (emu_cov_k_kms2 > 0.5e-3)
    interp_y = np.interp(hdu[2].data["K"][ind], emu_cov_k_kms2[ind2], diag_emu2[ind2])
    # ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PLYA"][ind], label="Emu")
    ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PSMOOTH"][ind], "C1-.", label="Emu mpg")
    
    ax[jj].plot(hdu[2].data["K"][ind], fid, "k:", label="fid")
    # ax[jj].plot(hdu[2].data["K"][ind], red, "k--", label="red")
    ax[jj].plot(hdu[2].data["K"][ind], xred1, "k--", label="stat+xred")
    ax[jj].plot(hdu[2].data["K"][ind], xred2, "k-", label="stat+xred+emu")
    
    
    # if(jj == 0):
ax[0].legend(ncols=2, loc="upper right", fontsize=5)
ax[-1].axis('off')

fig.supxlabel(r'$k[\mathrm{km}^{-1}\mathrm{s}]$')
fig.supylabel(r'$\sigma$')


plt.xscale("log")
plt.yscale("log")
plt.ylim(7e-3, 20)
plt.tight_layout()
plt.savefig("nyx_qmle.pdf")

# %%

# %%

# %%

# %%
zz = np.unique(hdu[2].data["Z"])

fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12, 10))
ax = ax.reshape(-1)

diag_stat = np.sqrt(np.diag(hdu[4].data))

for jj in range(len(zz) - 1):

    ind = (hdu[2].data["Z"] == zz[jj]) & (hdu[2].data["K"] > 1e-3)
    psmooth = hdu[1].data["PSMOOTH"][ind]
    
    diag = np.zeros(np.sum(ind))
    diag_red = np.zeros(np.sum(ind))
    diag_xred = np.zeros(np.sum(ind))
    for lab in sys_labels:
        ax[jj].plot(hdu[2].data["K"][ind], hdu[2].data[lab][ind]/psmooth, label=lab)
        if lab in sys_corr:
            diag += hdu[2].data[lab][ind]**2
        if lab in sys_corr_red:
            diag_red += hdu[2].data[lab][ind]**2
        if lab in sys_corr_xred:
            diag_xred += hdu[2].data[lab][ind]**2
    
    cov = np.outer(np.sqrt(diag), np.sqrt(diag))
    cov_red = np.outer(np.sqrt(diag_red), np.sqrt(diag_red))
    cov_xred = np.outer(np.sqrt(diag_xred), np.sqrt(diag_xred))
    
    for lab in sys_diag:
        cov[np.diag_indices_from(cov)] += hdu[2].data[lab][ind]**2
    for lab in sys_diag_red:
        cov_red[np.diag_indices_from(cov_red)] += hdu[2].data[lab][ind]**2
    for lab in sys_diag_xred:
        cov_xred[np.diag_indices_from(cov_xred)] += hdu[2].data[lab][ind]**2
        
    ax[jj].plot(hdu[2].data["K"][ind], diag_stat[ind]/psmooth, ".-", label="Stat")

    ax[jj].text(1.2e-3, 0.07, "z="+str(zz[jj]))

    indz = np.argmin(np.abs(emu_cov_zz - zz[jj]))
    ind2 = (emu_cov_zz == emu_cov_zz[indz]) & (emu_cov_k_kms > 0.5e-3)
    interp_y = np.interp(hdu[2].data["K"][ind], emu_cov_k_kms[ind2], diag_emu[ind2])
    # ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PLYA"][ind], label="Emu")
    ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PSMOOTH"][ind]/psmooth, "C0-.", label="Emu nyx")

    fid_nyx = np.sqrt(
        np.sqrt(np.diag(cov))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    red_nyx = np.sqrt(
        np.sqrt(np.diag(cov_red))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    xred_nyx = np.sqrt(
        np.sqrt(np.diag(cov_xred))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )

    
    indz = np.argmin(np.abs(emu_cov_zz2 - zz[jj]))
    ind2 = (emu_cov_zz2 == emu_cov_zz2[indz]) & (emu_cov_k_kms2 > 0.5e-3)
    interp_y = np.interp(hdu[2].data["K"][ind], emu_cov_k_kms2[ind2], diag_emu2[ind2])

    fid_mpg = np.sqrt(
        np.sqrt(np.diag(cov))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    red_mpg = np.sqrt(
        np.sqrt(np.diag(cov_red))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    xred_mpg = np.sqrt(
        np.sqrt(np.diag(cov_xred))**2 
      + diag_stat[ind]**2 
      + (interp_y * hdu[1].data["PSMOOTH"][ind])**2
    )
    
    # ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PLYA"][ind], label="Emu")
    ax[jj].plot(hdu[2].data["K"][ind], interp_y * hdu[1].data["PSMOOTH"][ind]/psmooth, "C1-.", label="Emu mpg")
    
    ax[jj].plot(hdu[2].data["K"][ind], fid_nyx/psmooth, "k:", label="fid nyx")
    ax[jj].plot(hdu[2].data["K"][ind], fid_mpg/psmooth, "r:", label="fid mpg")
    ax[jj].plot(hdu[2].data["K"][ind], red_nyx/psmooth, "k-.", label="red nyx")
    ax[jj].plot(hdu[2].data["K"][ind], red_mpg/psmooth, "r-.", label="red mpg")
    ax[jj].plot(hdu[2].data["K"][ind], xred_nyx/psmooth, "k--", label="xred nyx")
    ax[jj].plot(hdu[2].data["K"][ind], xred_mpg/psmooth, "r--", label="xred mpg")
    
    
    # if(jj == 0):
ax[0].legend(ncols=2, loc="upper right", fontsize=5)
ax[-1].axis('off')

fig.supxlabel(r'$k[\mathrm{km}^{-1}\mathrm{s}]$')
fig.supylabel(r'$\sigma/P1D$')


ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylim(5e-3, 0.1)
plt.tight_layout()
plt.savefig("qmle_error_ratio.pdf")

# %%

# %%
diag = np.zeros(len(hdu[2].data["Z"]))
for lab in sys_include:
    diag += hdu[2].data[lab]**2

cov = np.outer(np.sqrt(diag), np.sqrt(diag))

for lab in sys_diag:
    cov[np.diag_indices_from(cov)] += hdu[2].data[lab]**2

# %%
hdu[5].data[:10,0]/cov[:10,0]

# %%
hdu[5].data[:10,0]/cov[:10,0]

# %% [markdown]
# ### Set archive (old)

# %%
# args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="Nyx_alphap_cov", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")

# path nyx files in NERSC /global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/
archive = set_archive(args.training_set)

# set output directory for this test
output_dir = "."

emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

if "Nyx" in emulator.emulator_label:
    emulator.list_sim_cube = archive.list_sim_cube
    if "nyx_14" in emulator.list_sim_cube:
        emulator.list_sim_cube.remove("nyx_14")
else:
    emulator.list_sim_cube = archive.list_sim_cube

# %% [markdown]
# ### New

# %%

# args = Args(emulator_label="CH24_nyxcen_gpr", training_set="models_Nyx_Mar2025_with_CGAN_val_3axes")
args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")


# args = Args(emulator_label="CH24_nyx_gp", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="CH24_nyx_gp", training_set="models_Nyx_Mar2025_with_CGAN_val_3axes")
# args = Args(emulator_label="CH24_nyx_gpr", training_set="models_Nyx_Mar2025_with_CGAN_val_3axes")
# args = Args(emulator_label="CH24", training_set="Cabayol23")
# args = Args(emulator_label="CH24_NYX", training_set="Nyx23_Jul2024")
output_dir = "."

emulator = set_emulator(
    emulator_label=args.emulator_label,
)
archive = None

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
    args.z_max = 4.3
    # args.z_min = 2.8
    # args.z_max = 3.2
elif choose_desiy1:
    true_cosmo = None
    args.true_igm_label= None
    args.data_label = "DESIY1"
    # args.cov_syst_type = "xred"
    # args.cov_syst_type = "fid"
    args.cov_syst_type = "red"
    # in NERSC
    # /global/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-baseline/qmle_measurement/DataProducts/
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_resocorr_v2.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v2.fits"
    
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_resocorr_v3.fits"
    args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/desi_y1_xe_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v6.fits"
    
    args.z_min = 2.1
    args.z_max = 4.3

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
# hdu = fits.open(args.p1d_fname)
# hdu[1].header
# plt.imshow(data["P1Ds"].full_cov_kms)

# rat = np.diag(hdu[5].data)/np.diag(hdu[4].data)
# zu = np.unique(hdu[1].data["Z"])
# for zz in zu:
#     _ = (hdu[1].data["Z"] == zz)
#     plt.plot(hdu[1].data["K"][_], rat[_], label=str(zz))
# plt.legend(ncol=3)
# plt.xscale("log")
# plt.yscale("log")

# %%
# ntos = 100 * np.sqrt(np.diag(data["P1Ds"].cov_Pk_kms[0]))/data["P1Ds"].Pk_kms[0]
# plt.plot(data["P1Ds"].k_kms[0], ntos)

# %%
print(data["P1Ds"].apply_blinding)
if data["P1Ds"].apply_blinding:
    print(data["P1Ds"].blinding)

# %%
data["P1Ds"].apply_blinding = False
data["P1Ds"].blinding = False

# %%
data["P1Ds"].plot_p1d()
if args.data_label_hires is not None:
    data["extra_P1Ds"].plot_p1d()

# %%
try:
    data["P1Ds"].plot_igm()
except:
    print("Real data, no true IGM history")

# %%

# %% [markdown]
# #### Set fiducial/initial options for the fit

# %%
# # std
# p2 = np.array([-0.00109798,  0.00691753])
# # bias + std
# p2 = np.array([-0.0058123,  0.0237336])

# %%

# args.fid_cosmo_label="Planck18_low"
# fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params()
# blob


# %%

# %%
# cosmology

# args.emu_cov_factor = None
args.emu_cov_factor = 1
# args.emu_cov_type = "diagonal"
args.emu_cov_type = "block"
# args.emu_cov_type = "full"



# args.fix_cosmo=True
args.fix_cosmo=False
args.vary_alphas=False
# args.vary_alphas=True
if "nyx" in args.emulator_label:
    sim_fid = "nyx_central"
    args.ic_correction=True
    # args.ic_correction=False
    args.fid_cosmo_label="Planck18_nyx"
else:
    sim_fid = "mpg_central"
    args.ic_correction=False
    args.fid_cosmo_label="Planck18_mpg"
# args.fid_cosmo_label="Planck18"

args.fid_label_mF=sim_fid
args.fid_label_T=sim_fid
args.fid_label_kF=sim_fid


fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# args.use_star_priors = None
# args.use_star_priors = {}
# Planck18 0.354 -2.300 -0.2155
# 5 sigma 0.056 0.011 0.0028
# blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params
# amin = blob["alpha_star"] - 0.0028
# amax = blob["alpha_star"] + 0.0028
# args.use_star_priors["alpha_star"] = [amin, amax]

# IGM
if choose_data == False:
    args.igm_priors = "hc"
else:
    args.igm_priors = "data"

# args.hcd_model_type = "Rogers2017"
# all z

# full
args.hcd_model_type = "new"
# args.mF_model_type = "chunks"    
# args.n_tau=len(data["P1Ds"].z)
# args.n_sigT=2
# args.n_gamma=2
# args.n_kF=2

# args.n_x_SiIII=1
# args.n_d_SiIII=1
# args.n_a_SiIII=1
# args.n_d_dla = 2
# args.n_s_dla = 1

# one z at a time
# args.mF_model_type = "chunks"
# args.n_tau=len(data["P1Ds"].z)

args.mF_model_type = "pivot"
args.n_tau=3
args.n_gamma=2
args.n_sigT=2
args.n_kF=2

# z at a time
# args.n_tau=1
# args.n_gamma=1
# args.n_sigT=1
# args.n_kF=1

args.resolution_model_type = "chunks"
args.n_res = len(data["P1Ds"].z)

# z at a time
# args.resolution_model_type = "pivot"
# args.n_res = 1


args.n_x_SiII=1
args.n_d_SiII=1
args.n_a_SiII=1

args.n_x_SiIII=1
args.n_d_SiIII=1
args.n_a_SiIII=1

args.n_x_CIV=1
args.n_d_CIV=1
args.n_a_CIV=1

args.n_x_MgII=1
args.n_d_MgII=1
args.n_a_MgII=1

args.hcd_model_type = "new"
args.n_d_dla = 1
args.n_s_dla = 1

args.n_agn = 0



# args.fid_SiIII_X=[0, -10] # fine
# args.fid_SiIII_D=[0, 5]
# args.fid_SiIII_A=[0, 1]
# args.fid_A_damp = [0, -9]
# args.fid_A_scale = [0, 5]
if "nyx" in args.emulator_label:
    args.fid_val_mF = [2.48, -6.0e-1, 7.46e-2]
    args.fid_val_gamma = [-0.425, 0.13]
    args.fid_val_sigT = [0, 5.82e-2]

    args.fid_SiIII_X=[0, -4.2]
    args.fid_SiIII_D=[0, 5.1]
    args.fid_SiIII_A=[0, 1.0]
    
    args.fid_SiII_X=[0, -5.4]
    args.fid_SiII_D=[0, 6.0]
    args.fid_SiII_A=[0, 1.25]
    
    args.fid_CIV_X=[0, -8.3]
    args.fid_CIV_D=[0, 4.7]
    args.fid_CIV_A=[0, 5]
    
    args.fid_A_damp = [0, -0.78]
    args.fid_A_scale = [0, 7.2]
else:
    args.fid_val_mF = [1.50, -5.97e-1, -7.51e-2]
    args.fid_val_gamma = [-6.16e-1, 4.39e-2]
    args.fid_val_sigT = [0, 1.08e-2]

    args.fid_SiIII_X=[0, -4.7]
    args.fid_SiIII_D=[0, 4.8]
    args.fid_SiIII_A=[0, 1.4]
    
    args.fid_SiII_X=[0, -5.8]
    args.fid_SiII_D=[0, 6.0]
    args.fid_SiII_A=[0, 1.7]
    
    args.fid_CIV_X=[0, -8.5]
    args.fid_CIV_D=[0, 4.8]
    args.fid_CIV_A=[0, 5.8]
    
    args.fid_A_damp = [0, -1.43]
    args.fid_A_scale = [0, 5.4]


args.fid_SiII_X=[0, -5]
args.fid_SiII_D=[0, 1]
args.fid_SiII_A=[0, 1]
args.fid_SiIII_X=[0, -5]
args.fid_SiIII_D=[0, 1]
args.fid_SiIII_A=[0, 1]
args.fid_CIV_X=[0, -5]
args.fid_CIV_D=[0, 1]
args.fid_CIV_A=[0, 1]
args.fid_MgII_X=[0, -5]
args.fid_MgII_D=[0, 1]
args.fid_MgII_A=[0, 1]
    
# args.fid_val_mF = [0,0,0]
# args.fid_val_gamma = [0,0]
# args.fid_val_sigT = [0,0]


args.fid_AGN = [0, -5.5]
# args.fid_AGN = [0, -1.5]

args.fid_R_coeff = [0,  0]

free_parameters = set_free_like_parameters(args, emulator.emulator_label)
free_parameters

# %%

# %% [markdown]
# ### Set likelihood

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

# %%

# %%
# from scipy.interpolate import interp1d

# %%
# # priors_tau = np.array([1.42945563, 1.29749214, 1.18380211, 1.08838556, 1.01124247,
# #        0.95237286, 0.91177671, 0.88945402, 0.88540481, 0.89962906,
# #        0.93212679])

# priors_tau = np.exp(np.array([ 0.30973341,  0.23064586,  0.14973022,  0.06698649, 0.01817552,
#         0.00985069,  0.00152586, -0.00679897, -0.0151238 , -0.02344864,
#        -0.03177347]))

# plt.plot(like.theory.model_igm.F_model.fid_z[:11], like.theory.model_igm.F_model.fid_tau_interp(like.theory.model_igm.F_model.fid_z[:11]) * priors_tau)
# plt.plot(like.theory.model_igm.F_model.fid_z[:11], like.theory.model_igm.F_model.fid_tau_interp(like.theory.model_igm.F_model.fid_z[:11]))


# like.theory.model_igm.F_model.fid_tau_interp = interp1d(like.theory.model_igm.F_model.fid_z[:11], like.theory.model_igm.F_model.fid_tau_interp(like.theory.model_igm.F_model.fid_z[:11]) * priors_tau, kind="cubic")

# %%
# priors_gamma = np.exp(np.array([ 0.21143884,  0.2070271 ,  0.19430682,  0.173278  ,  0.14394062,
#         0.10629471,  0.06034025,  0.00607725, -0.0564943 , -0.12737439,
#        -0.20656303]))
# plt.plot(like.theory.model_igm.T_model.fid_z[:11], like.theory.model_igm.T_model.fid_gamma_interp(like.theory.model_igm.T_model.fid_z[:11]) * priors_gamma)
# plt.plot(like.theory.model_igm.T_model.fid_z[:11], like.theory.model_igm.T_model.fid_gamma_interp(like.theory.model_igm.T_model.fid_z[:11]))
# like.theory.model_igm.T_model.fid_gamma_interp = interp1d(like.theory.model_igm.T_model.fid_z[:11], like.theory.model_igm.T_model.fid_gamma_interp(like.theory.model_igm.T_model.fid_z[:11]) * priors_gamma, kind="cubic")

# %%
# priors_sigT_kms = np.exp(0.15)
# plt.plot(like.theory.model_igm.T_model.fid_z[:11], like.theory.model_igm.T_model.fid_sigT_kms_interp(like.theory.model_igm.T_model.fid_z[:11]) * priors_sigT_kms)
# plt.plot(like.theory.model_igm.T_model.fid_z[:11], like.theory.model_igm.T_model.fid_sigT_kms_interp(like.theory.model_igm.T_model.fid_z[:11]))
# like.theory.model_igm.T_model.fid_sigT_kms_interp = interp1d(like.theory.model_igm.T_model.fid_z[:11], like.theory.model_igm.T_model.fid_sigT_kms_interp(like.theory.model_igm.T_model.fid_z[:11]) * priors_sigT_kms, kind="cubic")

# %%
# like.plot_cov_to_pk()
# like.plot_correlation_matrix()

# %%
# like.plot_hull_fid()

# %%
# like.plot_igm(cloud=True)

# %%
# z = like.data.z
# k_kms = like.data.k_kms
# like.theory.model_cont.agn_model.plot_contamination(z, k_kms)

# z = like.data.z
# k_kms = like.data.k_kms
# like.theory.model_cont.hcd_model.plot_contamination(z, k_kms);

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %% [markdown]
# Compare data and fiducial/starting model

# %%

like.plot_p1d(residuals=False)
like.plot_p1d(residuals=True)

# %%

# %%
# baseline 1354
# snr3 1088
# xe 2373
# fft 3358

# %% [markdown]
# ### Set fitter

# %%
# for sampler, no real fit, just test
# args.n_steps=1000
# args.n_burn_in=1500
# args.parallel=False
# args.explore=False
# args.n_steps=500
# args.n_burn_in=500
# args.parallel=False
# args.explore=True

args.n_steps=5
args.n_burn_in=1
args.parallel=False
args.explore=True

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

# %%

# %%
# plotter.plot_hull(p0=p0)

# %% [markdown]
# ### Run minimizer

# %% [markdown]
# 4 min 30 s

# %%
# %%time
if like.truth is None:
    # p0 = np.zeros(len(like.free_params)) + 0.5
    p0 = np.array(list(like.fid["fit_cube"].values()))
else:
    p0 = np.array(list(like.truth["like_params_cube"].values()))*1.01
p0 = np.array(list(like.fid["fit_cube"].values()))
# p0[:] = 0.5
fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)
# zmask = np.array([2.4])
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %%
955

# %% [markdown]
# stop 3.8 fid
#
# nyx 811, 807 with z-evolve SiIII (1.85108976e-03?!)
#
# mpg

# %%

# %%
## mpg (cosmo estable between tau3 and tau11)
# cosmo, tau3 1346
# [ 2.86243181e-09  9.22639007e-01]
# Delta2_star 0.42953
# n_star -2.34356

# cosmo, tau11, 1289
# [ 2.88678243e-09  9.17074644e-01]
# Delta2_star 0.4269
# n_star -2.34912

## nyx full_emu_cov (Delta2_star not estable between tau3 and tau11)
# cosmo, tau11, 1108, varies a lot!
# [ 2.67372327e-09  9.45267892e-01]
# Delta2_star 0.42575
# n_star -2.32093

# cosmo, tau3, 1144
# [ 2.28964445e-09  9.39897597e-01]
# Delta2_star 0.35949
# n_star -2.3263

# cosmo, tau3, 1144 (no IC correction, cosmo estable)
# [ 2.31703526e-09  9.31103034e-01]
# Delta2_star 0.35549
# n_star -2.3351

# cosmo, tau3, 1040 (CIV)
# 2.33627483e-09  9.35807556e-01
# Delta2_star 0.36289
# n_star -2.33039

# full baseline nyx 1021
# 2.38475206e-09  9.36796097e-01
# Delta2_star 0.37139
# n_star -2.3294

# %%
# plotter.plot_p1d(residuals=True, plot_panels=True)

# %%
plotter = Plotter(fitter, save_directory="mpg_baseline_chunk")
plotter.plots_minimizer()
# plotter.plot_metal_cont(plot_data=True)

# %%
# plotter.plot_igm()

# %%
# plotter.plot_mle_cosmo()

# %%

# %%
# plotter = Plotter(fitter, save_directory=None)
# if args.fix_cosmo == False:
    # plotter.plot_mle_cosmo()
# plotter.plots_minimizer()

# %%
# help(plotter.plot_hcd_cont)

# %%

# plotter.plot_metal_cont(plot_data=True)

# %%

# plotter.plot_hcd_cont(plot_data=True)

# %%
# QMLE v3
# baseline 1309, 986.21
# w/ rescorr 1346, 983
# snr 1111, 845 (no z dependence!)
# xe (up to z=3.8) 2299, 1972
# fft 1399, 1179

# baseline
# Fit params no cube: [ 2.57392369e-09  9.38081163e-01 -2.32907115e-02 -5.63143248e-02
#   1.25715055e-01  1.50849502e-02 -6.43016274e-03 -4.25355260e+00
#   5.07771208e+00  1.07178963e+00 -5.17128039e+00  6.36987725e+00
#   1.17152887e+00 -8.55561903e-01  7.11276860e+00 -2.03653868e+00]
# $\mathrm{ln}\,\tau_0$ -0.05327614106312262
# $\mathrm{ln}\,\sigma^T_0$ -0.21475119400021098
# $\mathrm{ln}\,\gamma_0$ 0.007708478413611053
# $\mathrm{ln}\,k^F_0$ 0.0082702143141003
# $\mathrm{ln}\,f^{SiIII}_0$ -4.184385480417157
# $\mathrm{ln}\,d^{SiIII}_0$ 5.22986622701272
# $a^{SiIII}_0$ 0.9835026303734834
# $\mathrm{ln}\,f^{SiII}_0$ -5.77632130037337
# $\mathrm{ln}\,d^{SiII}_0$ 6.004611334746445
# $a^{SiII}_0$ 1.4272487413191837
# $\mathrm{ln}\,f^\mathrm{HCD}_0$ -0.20194201272522783
# $\mathrm{ln}\,s^\mathrm{HCD}_0$ 7.233054495813131
# Delta2_star 0.6144759530287037
# n_star -2.261580861937121
# $A_s$ 3.301904862136354e-09
# $n_s$ 1.0046175332038638
# $\mathrm{ln}\,\mathrm{AGN}_0$ -1.7825685481689622

# fft
# Fit params no cube: [ 2.70795062e-09  9.59887430e-01 
# -5.89238167e-02 -6.57261002e-02 2.03845630e-01  5.88594448e-02  2.46042037e-01 -1.72288393e-02
#  -4.75898427e+00  5.04208770e+00  1.06470466e+00 
# -5.57316079e+00 6.45162450e+00  7.45457315e-01 
# -1.95259166e+00  6.83429638e+00
#  -1.82567876e+00]

# %% [markdown]
# ### Run one z at a time

# %%
args.fid_MgII_X = [0, -5]
args.fid_MgII_D = [0, 0.5]
args.fid_MgII_A = [0, 0]

args.fid_SiIII_X = [0, -5]
args.fid_SiIII_D = [0, 1]
args.fid_SiIII_A = [0, 1]

args.fid_SiII_X = [0, -5]
args.fid_SiII_D = [0, 1]
args.fid_SiII_A = [0, 6]


args.fid_CIV_X=[0, -5]
args.fid_CIV_D=[0, 1]
args.fid_CIV_A=[0, 0]

# args.fid_AGN = [-2]

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_chi2 = []
# for ii in range(len(like.data.z)): 
for ii in range(2,3): 
    print(ii)
    zmask = np.array([like.data.z[ii]])
    fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%
66

# %%
-4.47170249e+00  7.91460529e-01  1.52984703e+00 
-5.51025276e+00  5.66789498e-01  2.69205069e+00
-5.95337137e+00  9.93091827e-01 -1.12098517e-01

0.81603719 0.15829211 0.7549745  
0.68621841 0.1133579  0.94867511
 0.63082858 0.19861837 0.48131691

# %%
-4.93667451e+00  5.08733031e+00 9.00758381e-02

# %%
# plotter = Plotter(fitter, save_directory=None, zmask=zmask)
# plotter = Plotter(fitter, save_directory=None)
# plotter.plots_minimizer()

# %%
# plotter.plot_res_cont(zmask=zmask)

# %%
plotter = Plotter(fitter, save_directory=None, zmask=zmask)
plotter.plot_metal_cont(plot_data=True, plot_panels=False)

# %%
dv = 2250
dv = 720


plt.plot(k_kms, np.cos(k_kms * dv))
a1 = 0.09 * 1e2
a1 = 100 # alpha
a2 = 161.95 * 1e-3
a2 = 1e-2 # damp

a2 = 0.5044365312410204
a1 = -2.0643981142654413


a2 = 0.5
a1 = -5



damp = 1/(1+np.exp(a1 * 1e2 * (k_kms - a2 * 1e-2)))
plt.plot(k_kms, np.cos(k_kms * dv) * damp/np.max(damp))
plt.plot(k_kms, damp/np.max(damp))
plt.xscale("log")

# %%
np.exp(-5)

# %%

# %%
k_kms = data["P1Ds"].k_kms[0].copy()
damp_coeff = 178.9100418222273 
alphas = np.linspace(0, 2, 10)
adim_damp = k_kms * damp_coeff

for alpha in alphas:
    damping = (adim_damp) ** alpha * np.exp(-1 * adim_damp**alpha)
    # damping = np.exp(-1 * adim_damp**alpha)
    plt.plot(k_kms, damping/np.max(damping), label=str(alpha))
plt.legend()
plt.xscale("log")

# %%

# %%
plotter.plot_agn_cont(plot_data=True)

# %%
# # fitter.mle_cube
# # plotter.fitter.mle_cube[-6] = 0.47838117
# # plotter.fitter.mle_cube[-6] = 0.57
# # plotter.fitter.mle_cube[-4] = 0.14
# # plotter.fitter.mle_cube[-3] = 0.9
# mle_results = plotter.fitter.like.plot_p1d(
#     values=plotter.fitter.mle_cube,
#     return_all=True,
#     show=False,
#     zmask=zmask,
# )
# plotter.plot_metal_cont(plot_data=True, mle_results=mle_results, plot_panels=False)

# %%

(1215.67 - 1206.5)/1215.67 * c_kms
        

# %%
SiIII 2261.384125511053
SiII 5713.878973619503
CIV 757.0827988352504

# %%
c_kms = 299792.458
lambda_lya = 1215.67
lambda_rest = [1206.52, 1193.28]
for ii in range(len(lambda_rest)):
    dv = (lambda_lya - lambda_rest[ii]) / lambda_lya * c_kms
    print(dv)

# %%
c_kms = 299792.458
# http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
lambda_rest = [1548.187, 1550.772]
lambda_rest = [2795.528, 2802.705]
lambda_rest = [1190.42, 1193.28]
dv = (lambda_rest[0] - lambda_rest[1]) / lambda_rest[0] * c_kms
dv

# %%

plotter.plot_metal_cont(plot_data=True, mle_results=mle_results, plot_panels=False)

# %%

# %%
print(np.sum(np.array(out_chi2)))
plt.plot(like.data.z, np.array(out_chi2))

# %%
keys_plot = [
    '$\\mathrm{ln}\\,\\tau_0$', 
    '$\\mathrm{ln}\\,\\sigma^T_0$', 
    '$\\mathrm{ln}\\,\\gamma_0$',
    '$\\mathrm{ln}\\,k^F_0$',
    '$\\mathrm{ln}\\,f^{SiIII}_0$',
    '$\\mathrm{ln}\\,d^{SiIII}_0$',
    '$a^{SiIII}_0$',
    '$\\mathrm{ln}\\,f^{SiII}_0$',
    '$\\mathrm{ln}\\,d^{SiII}_0$',
    '$a^{SiII}_0$',
    '$\\mathrm{ln}\\,f^\\mathrm{HCD}_0$',
    '$\\mathrm{ln}\\,s^\\mathrm{HCD}_0$',
    'Delta2_star',
    'n_star',
    "alpha_star",
    # '$A_s$',
    # '$n_s$',
    "$\\mathrm{ln}\\,\\mathrm{AGN}_0$",
    '$\\mathrm{R}_0$',
    # '$\\Delta^2_\\star$',
    # '$n_\\star$',
]

# %%

# %%

# fig, ax = plt.subplots(4, 3, figsize=(10, 8))
fig, ax = plt.subplots(5, 3, figsize=(10, 8), sharex=True)
ax = ax.reshape(-1)
dict_out = {}
jj = 0
for ii, key in enumerate(keys_plot):
    if key not in out_mle[0]:
        continue
    dict_out[key] = np.zeros(len(like.data.z))
    for iz in range(len(like.data.z)):
        ax[jj].scatter(like.data.z[iz], out_mle[iz][key])
        dict_out[key][iz] = out_mle[iz][key]
    ax[jj].set_ylabel(key)
    ax[jj].set_xlabel(r"$z$")
    jj += 1

jj = 0
for ii, key in enumerate(keys_plot):
    if key not in dict_out:
        continue
    print(key, np.median(dict_out[key]))
    ax[jj].plot(like.data.z, like.data.z[:]*0 + np.median(dict_out[key]))
    jj += 1
    
plt.tight_layout()

# %%
for key in dict_out:
    print(key, np.median(dict_out[key]))

# %%
plt.plot(like.data.z, dict_out['$\\mathrm{ln}\\,\\tau_0$'])
x = like.data.z
y = dict_out['$\\mathrm{ln}\\,\\tau_0$']
fit = np.polyfit(x[6:], y[6:], 1)
plt.plot(like.data.z, np.poly1d(fit)(like.data.z))

# %%
np.poly1d(fit)(like.data.z)

# %%
np.poly1d(fit)(like.data.z)

# %%

# %%
plt.plot(like.data.z, dict_out['$\\mathrm{ln}\\,\\gamma_0$'])
x = like.data.z
y = dict_out['$\\mathrm{ln}\\,\\gamma_0$']
fit = np.polyfit(x, y, 2)
plt.plot(like.data.z, np.poly1d(fit)(like.data.z))

# %%
np.poly1d(fit)(like.data.z) 

# %%
plt.plot(like.data.z, dict_out['$\\mathrm{ln}\\,\\sigma^T_0$'])
x = like.data.z
y = dict_out['$\\mathrm{ln}\\,\\sigma^T_0$']
fit = np.polyfit(x[:-3], y[:-3], 1)
plt.plot(like.data.z, np.poly1d(fit)(like.data.z))

# %% [markdown]
# return IGM parameters!!

# %%
np.poly1d(fit)(like.data.z) 

# %%
# I now have a version 3: /global/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-baseline/qmle_measurement/DataProducts/v3 . These have resolution systematics updated and propagated to higher redshifts. I suggest you try what happens with resolution correction. This is still not the baseline, but made this variation for testing purposes.
# desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits
# and
# desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_resocorr_v3.fits
# There is also SNR>3 variation which should have minimal to zero noise systematics

# %% [markdown]
# For Nyx fiducial
#
# ['As', 'ns', 'nrun', 'ln_tau_0', 'ln_tau_1', 'ln_tau_2', 'ln_tau_3', 'ln_tau_4', 'ln_tau_5', 'ln_tau_6', 'ln_tau_7', 'ln_tau_8', 'ln_tau_9', 'ln_tau_10', 'ln_sigT_kms_0', 'ln_gamma_0', 'ln_kF_0']
#
# ['As', 'ns', 'nrun', 'ln_tau_0', 'ln_tau_1', 'ln_tau_2', 'ln_tau_3', 'ln_tau_4', 'ln_tau_5', 'ln_tau_6', 'ln_tau_7', 'ln_tau_8', 'ln_tau_9', 'ln_tau_10', 'ln_sigT_kms_0', 'ln_gamma_0', 'ln_kF_0', 'ln_x_SiIII_0', 'ln_d_SiIII_0', 'a_SiIII_0', 'ln_A_damp_0', 'ln_A_scale_0']

# %%
# plotter.plot_p1d(zmask=zmask, residuals=True)
# plotter.plot_igm(zmask=zmask)
# plotter.plot_hcd_cont(plot_data=True)
# plotter.plot_metal_cont(plot_data=True)

# %%
plotter = Plotter(fitter, save_directory=None, zmask=zmask)
if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()
plotter.plots_minimizer()

# %%

# %%
plotter = Plotter(fitter, save_directory=None, zmask=zmask)
if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()
plotter.plots_minimizer()

# %%

# %%
plotter = Plotter(fitter, save_directory=None)
if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()
plotter.plots_minimizer()

# %%

# %%

# %%

# %%

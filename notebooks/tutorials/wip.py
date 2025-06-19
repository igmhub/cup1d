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

# %% [markdown]
# ### Set arguments

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
    args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/DESI-DR1/qmle_measurement/DataProducts/v3/desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/desi_y1_xe_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v6.fits"
    # folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/fft_measurement/"
    # args.p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7_no_metal_corr.fits"
    # args.p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7.fits"
    # args.p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"
    
    args.z_min = 2.1
    args.z_max = 4.3
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
# Plot covariance


# from cup1d.likelihood.plotter import plot_cov

# folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/fft_measurement/"
# p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7.fits"

# folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/"
# p1d_fname = folder + "desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
# p1d_fname = folder + "desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"


# plot_cov(p1d_fname, save_directory='.', lab = "fid")



# %%
# different contributions to QMLE P1D
# p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
# hdu = fits.open(p1d_fname)
# _ = (hdu[1].data["Z"] == 2.2) & (hdu[1].data["K"] < 0.04)
# plt.plot(hdu[1].data["K"][_], hdu[1].data["PLYA"][_])
# plt.plot(hdu[1].data["K"][_], hdu[1].data["PRAW"][_])
# plt.plot(hdu[1].data["K"][_], hdu[1].data["PNOISE"][_])
# plt.plot(hdu[1].data["K"][_], hdu[1].data["ThetaP"][_])
# # plt.plot(hdu[1].data["K"][_], hdu[1].data["PRAW"][_] - hdu[1].data["PNOISE"][_])
# plt.plot(hdu[1].data["K"][_], hdu[1].data["PFID"][_])

# %% [markdown]
# Metal subtraction

# %%
# # different contributions to FFT P1D

# folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/fft_measurement/"
# # p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v6.fits"
# # hdu[1].header


# p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7.fits"
# hdu = fits.open(p1d_fname)

# p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7_no_metal_corr.fits"
# hdu_nometal = fits.open(p1d_fname)

# p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"
# hdu_direct = fits.open(p1d_fname)
# zuse = np.unique(hdu[1].data["Z"])

# # for iz in range(len(zuse)):
# for iz in range(1,2):

#     _ = (hdu[1].data["Z"] == zuse[iz]) & (hdu[1].data["K"] < 0.03)
#     p1d_no = hdu_nometal[1].data["PLYA"][_]
#     p1d_fid = hdu[1].data["PLYA"][_]
#     p1d_dir = hdu_direct[1].data["PLYA"][_]
    
#     plt.plot(hdu[1].data["K"][_], p1d_fid/p1d_dir-1)
#     # plt.plot(hdu[1].data["K"][_], p1d_no, label="no_metal_corr")
#     # plt.plot(hdu[1].data["K"][_], p1d_dir/p1d_no-1, label="direct_metal_subtraction")
#     # plt.plot(hdu[1].data["K"][_], (p1d_no - p1d_dir)/p1d_fid)
#     # plt.plot(hdu[1].data["K"][_], p1d_no/p1d_fid)
#     # plt.plot(hdu[1].data["K"][_], p1d_dir/p1d_fid)
#     # y = p1d_no - p1d_yes
#     # y = hdu[1].data["K"][_] * (p1d_no - p1d_yes)/np.pi
#     # plt.plot(hdu[1].data["K"][_], y, label=str(z))
# # plt.plot(hdu[1].data["K"][_], hdu[1].data["PRAW"][_])
# # plt.plot(hdu[1].data["K"][_], hdu[1].data["PNOISE"][_])
# plt.legend()
# # plt.xscale("log")
# # plt.yscale("log")
# # plt.ylim(-0.05, 0.05)

# %%

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
# data["P1Ds"].apply_blinding = False
# data["P1Ds"].blinding = False

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

args.emu_cov_factor = 1
args.emu_cov_type = "block"
args.rebin_k = 6
args.cov_factor = 1
args.fix_cosmo=True
args.vary_alphas=False
args.fid_cosmo_label="Planck18"
sim_fid = "mpg_central"
args.fid_label_mF=sim_fid
args.fid_label_T=sim_fid
args.fid_label_kF=sim_fid

# args.emu_cov_factor = None
# args.emu_cov_type = "diagonal"
# args.emu_cov_type = "full"



# args.fix_cosmo=True
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
args.fid_cosmo_label="Planck18"

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
args.n_tau=1

args.n_gamma=1
args.n_sigT=1
args.n_kF=1

args.resolution_model_type = "pivot"
args.n_res = 2
# args.resolution_model_type = "chunks"
# args.n_res = len(data["P1Ds"].z)

# z at a time
# args.n_tau=1
# args.n_gamma=1
# args.n_sigT=1
# args.n_kF=1
# args.resolution_model_type = "pivot"
# args.n_res = 1

lines_use = [
    # "Lya_SiIII",
    # "Lya_SiIIa",
    # "Lya_SiIIb",
    # "SiIIa_SiIIb",
    # "SiIIa_SiIII",
    # "SiIIb_SiIII",
]

lines_not_use = [
    "Lya_SiIII",
    "Lya_SiIIa",
    "Lya_SiIIb",
    "SiIIa_SiIIb",
    "SiIIa_SiIII",
    "SiIIb_SiIII",
]

# at a time
f_prior = {
    "Lya_SiIII": -4.2,
    "Lya_SiII": -4.6,
    "SiIIb_SiIII": -6.6,
    "SiII_SiII": -5.5,
    "SiIIa_SiIII": -6.2,
    "CIV_CIV": -10.5,
}


# all z
# d_prior = {
#     "Lya_SiIII": 0.6,
#     "Lya_SiII": -0.3,
#     "SiIIb_SiIII": 3,
#     "SiII_SiII": 1,
#     "SiIIa_SiIII": 1.4,
#     "CIV_CIV": 0,
# }

# at a time
d_prior = {
    "Lya_SiIII": 0.4,
    "Lya_SiII": -0.9,
    "SiIIb_SiIII": 2.7,
    "SiII_SiII": 0.8,
    "SiIIa_SiIII": 1.6,
    "CIV_CIV": 0,
}

# all z
# a_prior = {
#     "Lya_SiIII": 1.5,
#     "Lya_SiII": 3,
#     "SiIIb_SiIII": 3.5,
#     "SiII_SiII": 3.5,
#     "SiIIa_SiIII": 0.5,
#     "CIV_CIV": 0,
# }

# at a time
a_prior = {
    "Lya_SiIII": 1.5,
    "Lya_SiII": 4.0,
    "SiIIb_SiIII": 2.5,
    "SiII_SiII": 4.0,
    "SiIIa_SiIII": 0.5,
    "CIV_CIV": 0,
}
    

for metal_line in lines_use:
    args.fid_metals[metal_line + "_X"] = [0, 0, f_prior[metal_line]]
    args.fid_metals[metal_line + "_D"] = [0, d_prior[metal_line]]
    args.fid_metals[metal_line + "_L"] = [0, 0]
    args.fid_metals[metal_line + "_A"] = [0, a_prior[metal_line]]
    args.n_metals["n_x_" + metal_line] = 1
    args.n_metals["n_d_" + metal_line] = 1
    args.n_metals["n_l_" + metal_line] = 0
    args.n_metals["n_a_" + metal_line] = 1


for metal_line in lines_not_use:
    args.fid_metals[metal_line + "_X"] = [0, -10.5]
    args.n_metals["n_x_" + metal_line] = 0
    args.n_metals["n_d_" + metal_line] = 0
    args.n_metals["n_l_" + metal_line] = 0
    args.n_metals["n_a_" + metal_line] = 0


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

    # args.fid_SiIII_X=[0, -4.2]
    # args.fid_SiIII_D=[0, 5.1]
    # args.fid_SiIII_A=[0, 1.0]
    
    # args.fid_SiII_X=[0, -5.4]
    # args.fid_SiII_D=[0, 6.0]
    # args.fid_SiII_A=[0, 1.25]
    
    # args.fid_CIV_X=[0, -8.3]
    # args.fid_CIV_D=[0, 4.7]
    # args.fid_CIV_A=[0, 5]
    
    args.fid_A_damp = [0, -0.78]
    args.fid_A_scale = [0, 7.2]
else:
    # args.fid_val_mF = [0.727, -0.558, -0.057]
    # args.fid_val_gamma = [-0.61, 0.037]
    # args.fid_val_sigT = [0, -0.004]
    # args.fid_val_kF = [0, -0.049]
    
    # args.fid_SiIII_X=[0, -4.7]
    # args.fid_SiIII_D=[0, 4.8]
    # args.fid_SiIII_A=[0, 1.4]
    
    # args.fid_SiII_X=[0, -5.8]
    # args.fid_SiII_D=[0, 6.0]
    # args.fid_SiII_A=[0, 1.7]
    
    # args.fid_CIV_X=[0, -8.5]
    # args.fid_CIV_D=[0, 4.8]
    # args.fid_CIV_A=[0, 5.8]
    
    args.fid_A_damp = [0, -1.43]
    args.fid_A_scale = [0, 5.4]

    
# args.fid_val_mF = [0,0,0]
# args.fid_val_gamma = [0,0]
# args.fid_val_sigT = [0,0]


args.fid_AGN = [0, -5.5]
# args.fid_AGN = [0, -1.5]

args.fid_R_coeff = [0,  0]


free_parameters = set_free_like_parameters(args, emulator.emulator_label)
free_parameters

# %%

# %%

# Impact of rebinning
# 11064 100
# 11042 10
# 11030 8
# 11004 6
# 10978 5
# 10932 4
# 10834 3
# 10578 2
# 14083 no

# %% [markdown]
# ### Set likelihood

# %%
args.set_baseline(fit_type="all")

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

# %%
len(like.free_param_names)

# %%
for p in like.free_params:
    print(p.name, '\t', p.value, '\t', np.round(p.min_value, 3), '\t', np.round(p.max_value, 3), '\t', p.Gauss_priors_width)

# %%
# Gaussian_priors

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

# %% [markdown]
# Compare data and fiducial/starting model

# %%
# # %%time
like.plot_p1d(residuals=False)
# like.plot_p1d(residuals=True)

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

# %% [markdown]
# ### Run minimizer

# %% [markdown]
# 2 min 44 s

# %%
# %%time
if like.truth is None:
    # p0 = np.zeros(len(like.free_params)) + 0.5
    p0 = np.array(list(like.fid["fit_cube"].values()))
else:
    p0 = np.array(list(like.truth["like_params_cube"].values()))*1.01
p0 = np.array(list(like.fid["fit_cube"].values()))
# p0[:] = 0.5
# fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)
# zmask = np.array([like.data.z[0]])
# fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)
fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, restart=True, nsamples=4)
# zmask = np.array([2.4])
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %%
fitter.mle

# %%

# %%
# tau chuncks from fit

res = np.array([1.04476564e+00,  1.02108923e+00,  1.01683558e+00,  9.87310936e-01,
  9.77622019e-01,  9.75382284e-01,  9.69040461e-01,  9.73455144e-01,
  9.73710999e-01,  9.28686659e-01,  9.16028012e-01])
tau_hist = like.theory.model_igm.F_model.fid_tau_interp(like.data.z)
# plt.plot(like.data.z, like.theory.model_igm.F_model.fid_tau_interp(like.data.z))
# plt.plot(like.data.z, like.theory.model_igm.F_model.fid_tau_interp(like.data.z)*res)
pfit = np.polyfit(like.data.z, res, 3)
plt.plot(like.data.z, res)
plt.plot(like.data.z, np.poly1d(pfit)(like.data.z))

z0 = 3
xz = np.log((1 + like.data.z) / (1 + z0))
pfit = np.polyfit(xz, np.log(res), 4)
print(pfit)
plt.plot(like.data.z, np.exp(np.poly1d(pfit)(xz)))

# %% [markdown]
# #### Latest

# %% [markdown]
# snr3 mpg fixcosmo chunk_tau 825
#
# snr3 mpg fixcosmo 3 tau 855
#
# snr3 mpg fixcosmo 3222 854
#
# snr3 mpg 3211 859?! No clear improvement when leaving cosmo free
#
# NEW baseline
#
# snr3 mpg fixcosmo 2222 872
# snr3 mpg fixcosmo 3022 851
# snr3 mpg fixcosmo 11,0,2,2 823 (delta chi2 of 30 between 3 and 11 params)
# snr3 mpg fixcosmo 5,0,3,0 tau 831 (Dchi2 8 with chunks)
# snr3 mpg fixcosmo 5,0,2,0 tau 833 (Dchi2 8 with chunks)

# %%

# diru = "test_snr3_11_1_1_1"
# diru = "test_snr3_3_3_1_1"
# diru = "test_snr3_3_2_2_2"
# diru = "test_snr3_3_2_1_1_cosmo"
# diru = "allz_snr3_nocosmo_2tau"
# diru = None
plotter = Plotter(fitter, save_directory=diru)
# plotter.plots_minimizer()

# %%
plotter.plot_igm()

# %%
plotter.plot_p1d(plot_panels=True, residuals=True)

# %%
plotter.plot_p1d_errors()

# %%
plotter.plot_p1d(residuals=True)

# %%

# %%
res

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
# plotter = Plotter(fitter, save_directory="mpg_baseline_chunk")
# plotter = Plotter(fitter, save_directory="mpg_baseline_pivot")
# plotter.plots_minimizer()
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

# %%
list_props = like.free_param_names.copy()


# %%

# %%
def chi2_param_at_time(args, list_props):
    
    args.emu_cov_factor = 1
    args.emu_cov_type = "block"
    args.rebin_k = 6
    args.cov_factor = 1
    args.fix_cosmo=True
    args.vary_alphas=False
    args.fid_cosmo_label="Planck18"
    sim_fid = "mpg_central"
    args.fid_label_mF=sim_fid
    args.fid_label_T=sim_fid
    args.fid_label_kF=sim_fid

    lines_use = [
        "Lya_SiIII",
        "Lya_SiIIa",
        "Lya_SiIIb",
        "SiIIa_SiIIb",
        "SiIIa_SiIII",
        "SiIIb_SiIII",
    ]
    args.hcd_model_type = "new"
    args.resolution_model_type = "pivot"
    args.fid_A_scale = [0, 5]

    lines_not_use = [
        # "CIV_CIV",
    ]

    for metal_line in lines_not_use:
        args.fid_metals[metal_line + "_X"] = [0, -10.5]
        args.n_metals["n_x_" + metal_line] = 0
        args.n_metals["n_d_" + metal_line] = 0
        args.n_metals["n_l_" + metal_line] = 0
        args.n_metals["n_a_" + metal_line] = 0

    # at a time
    f_prior = {
        "Lya_SiIII": -4.2,
        "Lya_SiIIa": -4.6,
        "Lya_SiIIb": -4.6,
        "SiIIa_SiIIb": -5.5,
        "SiIIa_SiIII": -6.2,
        "SiIIb_SiIII": -6.6,
    }

    # at a time
    d_prior = {
        "Lya_SiIII": 0.4,
        "Lya_SiIIa": -0.9,
        "Lya_SiIIb": -0.9,
        "SiIIa_SiIIb": 0.8,
        "SiIIa_SiIII": 1.6,
        "SiIIb_SiIII": 2.7,
    }
    
    # at a time
    a_prior = {
        "Lya_SiIII": 1.5,
        "Lya_SiIIa": 4.0,
        "Lya_SiIIb": 4.0,
        "SiIIa_SiIIb": 4.0,
        "SiIIa_SiIII": 0.5,
        "SiIIb_SiIII": 2.5,
    }

    for prop in list_props:
        if "ln_tau" in prop:
            args.n_tau = 0
        else:
            args.n_tau = 1

        if "ln_sigT" in prop:
            args.n_sigT = 0
        else:
            args.n_sigT = 1

        if "ln_gamma" in prop:
            args.n_gamma = 0
        else:
            args.n_gamma = 1

        if "ln_kF" in prop:
            args.n_kF = 0
        else:
            args.n_kF = 1

        for metal_line in lines_use:
            args.fid_metals[metal_line + "_L"] = [0, 0]
            args.n_metals["n_l_" + metal_line] = 0

            if "ln_x_" + metal_line in prop:
                args.n_metals["n_x_" + metal_line] = 0
                args.fid_metals[metal_line + "_X"] = [0, -10.5]
            else:
                args.n_metals["n_x_" + metal_line] = 1
                args.fid_metals[metal_line + "_X"] = [0, f_prior[metal_line]]

            if "d_" + metal_line in prop:
                args.n_metals["n_d_" + metal_line] = 0
                args.fid_metals[metal_line + "_D"] = [0, 0]
            else:
                args.n_metals["n_d_" + metal_line] = 1
                args.fid_metals[metal_line + "_D"] = [0, d_prior[metal_line]]

            if "a_" + metal_line in prop:
                args.n_metals["n_a_" + metal_line] = 0
                args.fid_metals[metal_line + "_A"] = [0, 2]
            else:
                args.n_metals["n_a_" + metal_line] = 1
                args.fid_metals[metal_line + "_A"] = [0, a_prior[metal_line]]

        if "R_coeff" in prop:
            args.n_res = 0
        else:
            args.n_res = 1

        if "ln_A_damp" in prop:
            args.n_d_dla = 0
            args.fid_A_damp = [0, -9.5]
        else:
            args.n_d_dla = 1
            args.fid_A_damp = [0, -1.5]

        if "ln_A_scale" in prop:
            args.n_s_dla = 0
            args.fid_A_scale = [0, 5]
        else:
            args.n_s_dla = 1
            args.fid_A_scale = [0, 5.6]

        args.fid_AGN = [0, -5.5]

        like = set_like(
            data["P1Ds"],
            emulator,
            args,
            data_hires=data["extra_P1Ds"],
        )

        # print()
        # f_space_len = 14
        # s_space_len = 5
        # for p in like.free_params:
        #     print(
        #         p.name,
        #         (f_space_len - len(p.name)) * " ",
        #         "\t",
        #         np.round(p.value, 3),
        #         (s_space_len - len(str(np.round(p.value, 3)))) * " ",
        #         "\t",
        #         np.round(p.min_value, 3),
        #         (s_space_len - len(str(np.round(p.min_value, 3)))) * " ",
        #         "\t",
        #         np.round(p.max_value, 3),
        #         (s_space_len - len(str(np.round(p.max_value, 3)))) * " ",
        #         "\t",
        #         p.Gauss_priors_width,
        #     )
        # print()

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
        out_mle_cube = []
        out_chi2 = []
        for ii in range(len(like.data.z)):
        # for ii in range(2,3): 
            print(prop, like.data.z[ii])
            zmask = np.array([like.data.z[ii]])
            # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
            # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask, restart=True)
            fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=8)
            # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
            out_mle.append(fitter.mle)
            out_mle_cube.append(fitter.mle_cube)
            out_chi2.append(fitter.mle_chi2)

        out = {}
        out["param_names"] = list_props
        out["zs"] = like.data.z
        out["mle"] = out_mle
        out["mle_cube"] = out_mle_cube
        out["chi2"] = out_chi2

        np.save("qmle3_lpo/"+prop+".npy", out)



# %%

from scipy.stats.distributions import chi2 as chi2_scipy

def chi2_grow_model_atz(args, iz, fix_props, basic_props, label_fit="basic"):

    fid_vals_metals = {
        "f_Lya_SiIII": -4.0,
        "f_Lya_SiIIb": -4.0,
        "f_SiIIa_SiIIb": -2.0,
        "f_SiII_SiIII": -6.0,
    }

    igm_params = [
        "n_tau",
        "n_gamma",
        "n_sigT",
        "n_kF",
    ]

    add_lines = [
        "SiIIa_SiIIb",
        "CIVa_CIVb",
        "MgIIa_MgIIb",
    ]
    
    list_all_props = [
        "n_tau",
        "n_gamma",
        "n_sigT",
        "n_kF",
        "n_f_Lya_SiIII",
        "n_s_Lya_SiIII",
        "n_f_Lya_SiIIb",
        "n_s_Lya_SiIIb",
        "n_p_Lya_SiIIb",
        "n_f_SiII_SiIII",
        "n_f_SiIIa_SiIIb",
        "n_s_SiIIa_SiIIb",
        "n_d_dla1",
        "n_d_dla2",
        "n_d_dla3",
        "n_d_dla4",
        "n_c_dla",
    ]
        
    
    args.set_baseline()
    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )

    out = {}
    out["zs"] = like.data.z[iz]
    out["param_names"] = []
    out["mle"] = []
    out["mle_cube"] = []
    out["chi2"] = []
    out["ndeg"] = []
    
    for prop in basic_props:
        list_props = []
        
        # setting to zero all props
        for prop2 in list_all_props:
            if prop2 in igm_params:
                args.fid_igm[prop2] = 0
            else:
                args.fid_cont[prop2] = 0

        # setting to one fix props
        for prop2 in fix_props:
            if prop2 in igm_params:
                args.fid_igm[prop2] = 1
            else:
                args.fid_cont[prop2] = 1
            list_props.append(prop2)

        # setting to one basic_props
        if prop is not None:
            if prop in igm_params:
                args.fid_igm[prop] = 1
            else:
                args.fid_cont[prop] = 1
            list_props.append(prop)

        for metal_label in args.metal_lines:
            if args.fid_cont["n_f_" + metal_label] == 0:
                args.fid_cont["f_" + metal_label] = [0, -10.5]
            else:
                args.fid_cont["f_" + metal_label] = [
                    0,
                    fid_vals_metals["f_" + metal_label],
                ]
                if metal_label in ["Lya_SiIII", "Lya_SiIIb", "SiIIa_SiIIb"]:
                    args.fid_cont["n_s_" + metal_label] = 1

            if args.fid_cont["n_s_" + metal_label] == 0:
                args.fid_cont["s_" + metal_label] = [0, -10.5]
            else:
                args.fid_cont["s_" + metal_label] = [0, 4.5]

            if metal_label in add_lines:
                args.fid_cont["p_" + metal_label] = [0, 1]

        for ii in range(4):
            if args.fid_cont["n_d_dla" + str(ii + 1)] == 0:
                args.fid_cont["HCD_damp" + str(ii + 1)] = [0, -10.5]
            else:
                args.fid_cont["HCD_damp" + str(ii + 1)] = [0, -2]

        like = set_like(
            data["P1Ds"],
            emulator,
            args,
            data_hires=data["extra_P1Ds"],
        )

        print()
        f_space_len = 14
        s_space_len = 5
        for p in like.free_params:
            print(
                p.name,
                (f_space_len - len(p.name)) * " ",
                "\t",
                np.round(p.value, 3),
                (s_space_len - len(str(np.round(p.value, 3)))) * " ",
                "\t",
                np.round(p.min_value, 3),
                (s_space_len - len(str(np.round(p.min_value, 3)))) * " ",
                "\t",
                np.round(p.max_value, 3),
                (s_space_len - len(str(np.round(p.max_value, 3)))) * " ",
                "\t",
                p.Gauss_priors_width,
            )
        print()

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
    
        zmask = np.array([like.data.z[iz]])
        fitter.run_minimizer(
            log_func_minimize=fitter.like.minus_log_prob, 
            p0=p0, 
            zmask=zmask, 
            restart=True, 
            nsamples=6
        )
        out_mle.append(fitter.mle)
        out_mle_cube.append(fitter.mle_cube)
        out_chi2.append(fitter.mle_chi2)
        
        print(list_props, fitter.mle_chi2)

        out["param_names"].append(list_props)
        out["mle"].append(fitter.mle)
        out["mle_cube"].append(fitter.mle_cube)
        out["chi2"].append(fitter.mle_chi2)
        out["ndeg"].append(len(like.data.k_kms[iz]) - len(list_props))

    np.save("qmle3_lpo/grow_"+label_fit+".npy", out)



# %%
# res = np.load("qmle3_lpo/grow_basic.npy", allow_pickle=True).item()

# %%
basic_props = [
    None,
    "n_gamma",
    "n_sigT",
    "n_kF",
    "n_f_Lya_SiIII",
    "n_f_Lya_SiIIb",
    "n_f_SiII_SiIII",
    "n_f_SiIIa_SiIIb",
    "n_d_dla1",
    "n_d_dla2",
    "n_d_dla3",
    "n_d_dla4",
]

len(basic_props)


# %%

for iz in range(2, len(like.data.z)):
# for iz in range(2):
    keep = True
    chi2_im = []
    prob_im = []

    fix_props = [
        "n_tau",
    ]
    
    basic_props = [
        None,
        "n_gamma",
        "n_sigT",
        "n_kF",
        "n_f_Lya_SiIII",
        "n_f_Lya_SiIIb",
        "n_f_SiII_SiIII",
        "n_f_SiIIa_SiIIb",
        "n_d_dla1",
        "n_d_dla2",
        "n_d_dla3",
        "n_d_dla4",
    ]
    it = 0
    while keep:
        label_fit = "iz_" + str(iz) + "_it_" + str(it)

        if it < 12:
            pass
        else:
            chi2_grow_model_atz(args, iz, fix_props, basic_props, label_fit=label_fit)
            
        res = np.load("qmle3_lpo/grow_" + label_fit + ".npy", allow_pickle=True).item()
        Dchi2 = np.zeros(len(res["chi2"]))
        
        for ii in range(1, len(res["chi2"])):
            Dchi2[ii] = res["chi2"][0] - res["chi2"][ii]
            # print(Dchi2[ii])
        ind = np.argmax(Dchi2)
        prob1 = chi2_scipy.sf(res["chi2"][0], res["ndeg"][0])
        prob2 = chi2_scipy.sf(res["chi2"][ind], res["ndeg"][ind])
        best_prop = res["param_names"][ind][-1]

        if Dchi2[ind] == 0:
            break
        
        if it == 0:
            prob_im.append(prob1)
        else:
            chi2_im.append(Dchi2[ind])
            prob_im.append(prob2)

        
        if prob2 > prob1:
            print(best_prop, "\t", np.round(Dchi2[ind], 2), prob1, prob2)
            fix_props.append(best_prop)
            basic_props.remove(best_prop)
            it += 1
        else:
            keep = False
            
    print(iz, fix_props)

# %%
len(like.data.z)

# %%
chi2_im

# %%
# plt.plot(prob_im)
plt.plot(chi2_im)
plt.yscale("log")

# %%
basic_props

# %%
basic_props.remove('n_kF')

# %%
basic_props

# %%
res["param_names"][ind][-1]

# %%
like.data.k_kms

# %%

# %%
fix_props = [
    "n_tau",
]

basic_props = [
    None,
    "n_gamma",
    "n_sigT",
    "n_kF",
    "n_f_Lya_SiIII",
    "n_f_Lya_SiIIb",
    "n_f_SiII_SiIII",
    "n_f_SiIIa_SiIIb",
    "n_d_dla1",
]

refine_props = [
    "n_s_Lya_SiIII",
    "n_s_Lya_SiIIb",
    "n_p_Lya_SiIIb",
    "n_s_SiIIa_SiIIb",
    "n_d_dla2",
    "n_d_dla3",
    "n_d_dla4",
]

chi2_grow_model_atz(args, fix_props, basic_props, label_fit="basic")


# %%

# %%

# %%
list_props = [
    "ln_tau_0",
    "ln_sigT_kms_0",
    "ln_gamma_0",
    "ln_kF_0",
    "ln_x_Lya_SiIII_0",
    "d_Lya_SiIII_0",
    "a_Lya_SiIII_0",
    "ln_x_Lya_SiIIa_0",
    "d_Lya_SiIIa_0",
    "a_Lya_SiIIa_0",
    "ln_x_Lya_SiIIb_0",
    "d_Lya_SiIIb_0",
    "a_Lya_SiIIb_0",
    "ln_x_SiIIa_SiIIb_0",
    "d_SiIIa_SiIIb_0",
    "a_SiIIa_SiIIb_0",
    "ln_x_SiIIa_SiIII_0",
    "d_SiIIa_SiIII_0",
    "a_SiIIa_SiIII_0",
    "ln_x_SiIIb_SiIII_0",
    "d_SiIIb_SiIII_0",
    "a_SiIIb_SiIII_0",
    "ln_A_damp_0",
    "ln_A_scale_0",
    "R_coeff_0",
]


list_props = ["all"]

chi2_param_at_time(args, list_props)

# %%

prop = "all"
res_all = np.load("qmle3_lpo/"+prop+".npy", allow_pickle=True).item()
chi2_total = np.array(res_all["chi2"])
zs = np.array(res_all["zs"])
print(np.sum(chi2_total), chi2_total)
for key in res_all["mle"][0]:
    print(key, res_all["mle"][0][key])


# %%
plt.plot(zs, chi2_total, "o-", label="FFT SB1 direct")

# %%
list_props = [
    "ln_tau_0",
    "ln_sigT_kms_0",
    "ln_gamma_0",
    "ln_kF_0",
    "ln_x_Lya_SiIII_0",
    "d_Lya_SiIII_0",
    "a_Lya_SiIII_0",
    "ln_x_Lya_SiIIa_0",
    "d_Lya_SiIIa_0",
    "a_Lya_SiIIa_0",
    "ln_x_Lya_SiIIb_0",
    "d_Lya_SiIIb_0",
    "a_Lya_SiIIb_0",
    "ln_x_SiIIa_SiIIb_0",
    "d_SiIIa_SiIIb_0",
    "a_SiIIa_SiIIb_0",
    "ln_x_SiIIa_SiIII_0",
    "d_SiIIa_SiIII_0",
    "a_SiIIa_SiIII_0",
    "ln_x_SiIIb_SiIII_0",
    "d_SiIIb_SiIII_0",
    "a_SiIIb_SiIII_0",
    "ln_A_damp_0",
    "ln_A_scale_0",
    "R_coeff_0",
]

chi2_prop = {}
for prop in list_props:
    chi2_prop[prop] = np.array(np.load("qmle3_lpo/"+prop+".npy", allow_pickle=True).item()["chi2"])

# %%
delta_chi2 = np.zeros((len(chi2_prop), len(chi2_total)))
for iprop, prop in enumerate(list_props):
    delta_chi2[iprop] = chi2_prop[prop] - chi2_total

# %%

# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.reshape(-1)
nn = delta_chi2.shape[0]
for jj in range(nn):
    if jj / nn < 0.25:
        ii = 0
    elif (jj / nn >= 0.25) & (jj / nn < 0.5):
        ii = 1
    elif (jj / nn >= 0.5) & (jj / nn < 0.75):
        ii = 2
    if jj / nn >= 0.75:
        ii = 3
    ax[ii].plot(zs, delta_chi2[jj], ".-", label=list_props[jj])
    ax[ii].axhline(2, color="k")

for ii in range(4):
    ax[ii].legend()
    ax[ii].set_yscale("log")
# plt.savefig("lpO_chi2z.png")

# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.reshape(-1)
nn = delta_chi2.shape[0]

for jj in range(nn):
    if jj / nn < 0.25:
        ii = 0
    elif (jj / nn >= 0.25) & (jj / nn < 0.5):
        ii = 1
    elif (jj / nn >= 0.5) & (jj / nn < 0.75):
        ii = 2
    if jj / nn >= 0.75:
        ii = 3
    ax[ii].scatter(jj, np.sum(delta_chi2[jj]), label=list_props[jj])
    ax[ii].axhline(2, color="k")

for ii in range(4):
    ax[ii].legend()
    ax[ii].set_yscale("log")
# plt.savefig("lpO_chi2.png")

# %%
# for z in fitter.like.data.z:
#     y = like.theory.model_cont.metal_models['Lya_SiIII'].get_amplitude(z)
#     print(np.log(y))

# %%

# %%

# %%

args.emu_cov_factor = 1
args.emu_cov_type = "block"
args.rebin_k = 6
args.cov_factor = 1
args.fix_cosmo=True
args.vary_alphas=False
args.fid_cosmo_label="Planck18"
sim_fid = "mpg_central"
args.fid_label_mF=sim_fid
args.fid_label_T=sim_fid
args.fid_label_kF=sim_fid

# z at a time
args.n_tau=1
args.n_gamma=0
args.n_sigT=0
args.n_kF=0
args.resolution_model_type = "pivot"
args.n_res = 1



lines_not_use = [
#     "CIV_CIV",
]

lines_use = [
    "Lya_SiIII",
    "Lya_SiIIa",
    "Lya_SiIIb",
    "SiIIa_SiIIb",
    "SiIIa_SiIII",
    "SiIIb_SiIII",
]


f_prior = {
    "Lya_SiIII": -4.2,
    "Lya_SiIIa": -4.6,
    "Lya_SiIIb": -4.6,
    "SiIIa_SiIIb": -5.5,
    "SiIIa_SiIII": -6.2,
    "SiIIb_SiIII": -6.6,
}

# at a time
d_prior = {
    "Lya_SiIII": 0.4,
    "Lya_SiIIa": -0.9,
    "Lya_SiIIb": -0.9,
    "SiIIa_SiIIb": 0.8,
    "SiIIa_SiIII": 1.6,
    "SiIIb_SiIII": 2.7,
}

# at a time
a_prior = {
    "Lya_SiIII": 1.5,
    "Lya_SiIIa": 4.0,
    "Lya_SiIIb": 4.0,
    "SiIIa_SiIIb": 4.0,
    "SiIIa_SiIII": 0.5,
    "SiIIb_SiIII": 2.5,
}


n_metals = {
    "Lya_SiIII": [1, 1, 1],
    "Lya_SiIIa": [0, 0, 0],
    "Lya_SiIIb": [1, 0, 1],
    "SiIIa_SiIII": [1, 0, 0],
    "SiIIb_SiIII": [1, 1, 0],
    "SiIIa_SiIIb": [1, 1, 1],
}
# n_metals = {
#     "Lya_SiIII": [1, 1, 1],
#     "Lya_SiIIa": [1, 1, 1],
#     "Lya_SiIIb": [1, 1, 1],
#     "SiIIa_SiIII": [1, 1, 1],
#     "SiIIb_SiIII": [1, 1, 1],
#     "SiIIa_SiIIb": [1, 1, 1],
# }
n_metals = {
    "Lya_SiIII": [1, 0, 1],
    "Lya_SiIIa": [1, 0, 1],
    "Lya_SiIIb": [1, 0, 1],
    "SiIIa_SiIII": [1, 0, 1],
    "SiIIb_SiIII": [1, 0, 1],
    "SiIIa_SiIIb": [1, 0, 1],
}


for metal_line in lines_use:
    args.fid_metals[metal_line + "_X"] = [0, f_prior[metal_line]]
    args.fid_metals[metal_line + "_D"] = [0, d_prior[metal_line]]
    args.fid_metals[metal_line + "_L"] = [0, 0]
    args.fid_metals[metal_line + "_A"] = [0, a_prior[metal_line]]
    args.n_metals["n_x_" + metal_line] = n_metals[metal_line][0]
    args.n_metals["n_d_" + metal_line] = n_metals[metal_line][1]
    if args.n_metals["n_d_" + metal_line] == 0:
        args.fid_metals[metal_line + "_D"] = [0, 0]
    args.n_metals["n_l_" + metal_line] = 0
    args.n_metals["n_a_" + metal_line] = n_metals[metal_line][2]
    if args.n_metals["n_a_" + metal_line] == 0:
        args.fid_metals[metal_line + "_A"] = [0, 2]

# metal_line = "Lya_SiIII"
# args.n_metals["n_x_" + metal_line] = 1

for metal_line in lines_not_use:
    args.fid_metals[metal_line + "_X"] = [0, -10.5]
    args.n_metals["n_x_" + metal_line] = 0
    args.n_metals["n_d_" + metal_line] = 0
    args.n_metals["n_l_" + metal_line] = 0
    args.n_metals["n_a_" + metal_line] = 0

args.hcd_model_type = "new"
args.n_d_dla = 1
args.n_s_dla = 1
args.fid_A_damp = [0, -1.4]
args.fid_A_scale = [0, 5.2]


# args.fid_val_kF = [0, -0.05]
args.fid_val_kF = [0, 0]
args.fid_AGN = [0, -5.5]

free_parameters = set_free_like_parameters(args, emulator.emulator_label)

# args.fid_AGN = [-2]

# args.prior_Gauss_rms = 0.1
args.prior_Gauss_rms = None

args.Gauss_priors = {}
args.Gauss_priors["ln_tau_0"] = [10]
# args.Gauss_priors["ln_sigT_kms_0"] = [0.02]
# args.Gauss_priors["ln_gamma_0"] = [0.08]
# args.Gauss_priors["ln_kF_0"] = [0.003]

f_Gprior = {
    "Lya_SiIII": 1,
    "Lya_SiIIa": 1,
    "Lya_SiIIb": 1,
    "SiIIa_SiIIb": 3,
    "SiIIa_SiIII": 4,
    "SiIIb_SiIII": 3,
}

d_Gprior = {
    "Lya_SiIII": 1.5,
    "Lya_SiIIa": 0.05,
    "Lya_SiIIb": 0.05,
    "SiIIa_SiIIb": 1,
    "SiIIa_SiIII": 0.03,
    "SiIIb_SiIII": 1,
}

a_Gprior = {
    "Lya_SiIII": 10,
    "Lya_SiIIa": 10,
    "Lya_SiIIb": 10,
    "SiIIa_SiIIb": 2,
    "SiIIa_SiIII": 0.05,
    "SiIIb_SiIII": 0.01,
}

for metal_line in lines_use:
    args.Gauss_priors["ln_x_"+metal_line+"_0"] = [f_Gprior[metal_line]]
    args.Gauss_priors["d_"+metal_line+"_0"] = [d_Gprior[metal_line]]
    args.Gauss_priors["a_"+metal_line+"_0"] = [a_Gprior[metal_line]]
args.Gauss_priors["ln_A_damp_0"] = [0.3]
args.Gauss_priors["ln_A_scale_0"] = [1]
args.Gauss_priors["R_coeff_0"] = [2]

args.Gauss_priors = {}





# %% [markdown]
# # Start here z_at_time

# %%
# args.set_baseline(fit_type="full")
args.set_baseline()

# %%
# like = set_like(
#     data["P1Ds"],
#     emulator,
#     args,
#     data_hires=data["extra_P1Ds"],
# )

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

print()

f_space_len = 14
s_space_len = 5
for p in like.free_params:
    print(
        p.name, (f_space_len-len(p.name)) * " ", "\t", 
        np.round(p.value, 3), (s_space_len-len(str(np.round(p.value, 3)))) * " ", '\t', 
        np.round(p.min_value, 3), (s_space_len-len(str(np.round(p.min_value, 3)))) * " ", '\t', 
        np.round(p.max_value, 3), (s_space_len-len(str(np.round(p.max_value, 3)))) * " ", '\t', 
        p.Gauss_priors_width
    )

print()

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

# %% [markdown]
# ### results (tets z at a time first 4 bins)
#
# #### metals (a, d, f fixed to free 198 to 245, 47):
#
# - 1 a for all z (delta chi2 6 fixed, 198 to 205)
# - 1 d for all z (delta chi2 22 fixed, 205 to 228) NEED MANY, 1 param ev?
# - 1 f for all z (delta chi2 17 fixed, 228 to 245) NEED MANY, 1 param ev?
# - no need of l
#
# #### check now if all metals are the same. How much do we lose for "all z the same" for each
#
# - 2-3 chi2 difference for f metals, but 12 for Lya-SiIII. At least for this one, redshift evolution. 

# %%
# p0_arr = out_mle_cube.copy()

# %%
p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_mle_cube = []
out_chi2 = []
# for ii in range(len(like.data.z)): 
for ii in range(1): 
# for ii in range(4, 5): 
# for ii in range(2,3): 
# for ii in range(9, 10): 
# for ii in range(2, 3): 
    zmask = np.array([like.data.z[ii]])
    print(ii, like.data.z[ii])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=4)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%
# args.rebin_k = 8
# like = set_like(
#     data["P1Ds"],
#     emulator,
#     args,
#     data_hires=data["extra_P1Ds"],
# )
# like.plot_p1d(fitter.mle_cube, zmask=[2.2])

# %%
fitter.mle

# %%
# fitter.like.theory.model_cont.metal_models["SiIIa_SiIIb"].plot_contamination(2.4, fitter.like.data.k_kms[1])

# %%

print(np.sum(out_chi2))
out_chi2

# %%
# new baseline
print(np.sum(out_chi2))
out_chi2

# %% [markdown]
# QMLE snr3, baseline 650

# %%

plt.plot(fitter.like.data.z, out_chi2, "o-", label="FFT SB1 direct")

# %%
# # out_chi2_dir = out_chi2.copy()
# # out_chi2_ind = out_chi2.copy()
# # out_chi2_qmle_snr3 = out_chi2.copy()
# # out_chi2_qmle_fid = out_chi2.copy()
# plt.plot(out_chi2_dir, "o-", label="FFT SB1 direct")
# plt.plot(out_chi2_ind, "o-", label="FFT SB1 model")
# plt.plot(out_chi2_qmle_snr3, "o-", label="QMLE snr3 SB1 model")
# plt.plot(out_chi2_qmle_fid, "o-", label="QMLE fid SB1 model")
# plt.legend()
# plt.xlabel(r"$z$")
# plt.ylabel(r"$\chi^2$")
# # plt.savefig("chi2_all.png")

# %%
nk = 0
for ii in range(len(like.data.k_kms)):
    nk += len(like.data.k_kms[ii])
nk

# %%
nk - 11 * len(like.data.k_kms)

# %%
# plotter.plot_p1d(zmask=zmask)

# %%
# diru = "fft_dirmetal_mpg_z_at_time_fulllines"
# diru = "fft_fid_mpg_z_at_time_fulllines"

# diru = "qmle_snr3_mpg_z_at_time_baseline"
# diru = "qmle_snr3_mpg_z_at_time_baseline5_p"
# diru = "qmle_snr3_mpg_z_at_time_fid_weakp"
# diru = "qmle_fid_mpg_z_at_time_fulllines"
diru = None

plotter = Plotter(fitter, save_directory=diru, zmask=zmask)

# %%
plotter.plot_metal_cont(smooth_k=False, plot_data=True, zrange=[2.3, 2.5], plot_panels=False)
# plotter.plot_metal_cont(smooth_k=False, plot_data=True, zrange=[2.9, 3.1], plot_panels=False)

# %%
c_kms = 299792.458
metals = ["SiIIa_SiIIb", "Lya_SiIII", "SiIIb_SiIII",  
          "SiIIa_SiIII", "Lya_SiIIb", "Lya_SiIIa", 
          "CIVa_CIVb", "MgIIa-MgIIb", "Lya_SiIIc", "SiIIc_SiIII"
         ]

for metal_label in metals:

    if metal_label == "Lya_SiIII":
        lambda_rest = [1206.52, 1215.67]
    elif metal_label == "Lya_SiIIb":
        lambda_rest = [1193.28, 1215.67]
    elif metal_label == "Lya_SiIIa":
        lambda_rest = [1190.42, 1215.67]
    elif metal_label == "SiIIa_SiIIb":
        lambda_rest = [1190.42, 1193.28]  # SiIIa-SiIIb
    elif metal_label == "SiIIa_SiIII":
        lambda_rest = [1190.42, 1206.52]  # SiIIa-SiIII
    elif metal_label == "SiIIb_SiIII":
        lambda_rest = [1193.28, 1206.52]  # SiIIb-SiIII
    elif metal_label == "CIVa_CIVb":
        lambda_rest = [1548.20, 1550.78]  # CIV-CIV
    elif metal_label == "MgIIa-MgIIb":
        lambda_rest = [2795.53, 2802.70]  # MgII-MgII
    elif metal_label == "SiIIc_SiIII":
        lambda_rest = [1206.51, 1260.42]
    elif metal_label == "Lya_SiIIc":
        lambda_rest = [1215.67, 1260.42]
    else:
        print("NO", metal_label)

    dv = np.log(lambda_rest[1]/lambda_rest[0]) * c_kms
    # z=3
    # dk = 1 / dv * c_kms / np.mean(lambda_rest) / (1+z)
    # dk = 1/lambda_rest[0] / (np.exp(dv/c_kms)-1)
    print(metal_label, np.round(dv, 2), np.round(2 * np.pi/dv, 4))

# %%

# %%

# %%
plotter.plot_p1d(out_mle_cube, z_at_time=True, plot_panels=True, residuals=True)

# %%
lines_use = [
    "Lya_SiIII",
    "Lya_SiIIb",
    "SiII_SiIII",
    "SiIIa_SiIIb",
]

for ii in range(len(fitter.like.data.z)):
    plotter.plot_illustrate_contaminants(out_mle_cube[ii].copy(), [fitter.like.data.z[ii]], lines_use=lines_use)

# %%
like.free_param_names

# %%
lines_use = [
    "Lya_SiIII",
    # "Lya_SiIIa",
    "Lya_SiIIb",
    # "Lya_SiIIc",
    # "SiIIa_SiIII",
    # "SiIIb_SiIII",
    "SiII_SiIII",
    # "SiIIc_SiIII",
    "SiIIa_SiIIb",
    # "CIVa_CIVb",
    # "MgIIa_MgIIb",
]
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.2], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.4], lines_use=lines_use)
plotter.plot_illustrate_contaminants(out_mle_cube[2].copy(), [2.6], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.8], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [3.6], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [3.8], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [4.0], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [3.6], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [3.0], lines_use=lines_use)

# %%
# plotter.plot_illustrate_contaminants(out_mle_cube[-4].copy(), [3.6], lines_use=lines_use)

# %%

# %%

# %%
# from cup1d.nuisance.metal_correction import SB1_power

# folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/fft_measurement/"
# file_metal = folder + "param_fit_side_band_1_kms.pickle"
# Pk_cont = SB1_power(data["P1Ds"].z, data["P1Ds"].k_kms, file_metal)
# for iz in range(0, 10):
#     # fun = a * data["P1Ds"].k_kms[iz] ** (-b)
#     # fun2 = B1 * np.exp(-b1 * data["P1Ds"].k_kms[iz])
#     # fun3 = C1 * np.exp(-c1 * data["P1Ds"].k_kms[iz])
#     plt.plot(data["P1Ds"].k_kms[iz], Pk_cont[iz])
#     # plt.plot(data["P1Ds"].k_kms[iz], fun+fun2+fun3)
#     y = v1 * data["P1Ds"].k_kms[iz] ** -v2
#     plt.plot(data["P1Ds"].k_kms[iz], y, "k")
# # plt.plot(data["P1Ds"].k_kms[iz], fun, lw=3, color="k")

# %%
# keys_plot = out_mle[0].keys()
# keys_no_plot = 

# %%
# keys_plot = [
#     '$\\mathrm{ln}\\,\\tau_0$', 
#     '$\\mathrm{ln}\\,\\sigma^T_0$', 
#     '$\\mathrm{ln}\\,\\gamma_0$',
#     '$\\mathrm{ln}\\,k^F_0$',
#     '$\\mathrm{ln}\\,f^{SiIII}_0$',
#     '$\\mathrm{ln}\\,d^{SiIII}_0$',
#     '$a^{SiIII}_0$',
#     '$\\mathrm{ln}\\,f^{SiII}_0$',
#     '$\\mathrm{ln}\\,d^{SiII}_0$',
#     '$a^{SiII}_0$',
#     '$\\mathrm{ln}\\,f^\\mathrm{HCD}_0$',
#     '$\\mathrm{ln}\\,s^\\mathrm{HCD}_0$',
#     '$\\mathrm{R}_0$',
#     # 'Delta2_star',
#     # 'n_star',
#     # "alpha_star",
#     # '$A_s$',
#     # '$n_s$',
#     # "$\\mathrm{ln}\\,\\mathrm{AGN}_0$",
#     # '$\\Delta^2_\\star$',
#     # '$n_\\star$',
# ]

# %%
len(fitter.paramstrings)

# %%

# %%
full_fit = np.array([0.41380907, 0.19873062, 0.84335202, 0.36286249, 0.19714842,
       0.38384624, 0.23510953, 0.90648141, 0.65477896, 0.71741965,
       0.41192486, 0.79660667, 0.41509414, 0.7768159 , 0.16607856,
       0.85488543, 0.61664629, 0.60074812, 0.80184355, 0.21050039,
       0.31437447, 0.99979126, 0.91889641, 0.76048415, 0.56688202,
       0.94734545])

attime_fit = [np.array([0.23717645, 0.70326833, 0.7579106 , 0.04784281, 0.33061696,
        0.31923253, 0.52919179, 0.57739349, 0.44320083, 0.46603739,
        0.834808  , 0.58872171, 0.48277026, 0.37359484, 0.39801294,
        0.35918721, 0.40609686, 0.95767393]),
 np.array([0.03097194, 0.09437952, 0.21276035, 0.32168535, 0.69355624,
        0.73409756, 0.63207805, 0.31013035, 0.41701602, 0.81712263,
        0.06417422, 0.54217481, 0.54008518, 0.87481782, 0.46036522,
        0.89263792, 0.84411348, 0.02411243]),
 np.array([0.00352233, 0.14928074, 0.26661643, 0.29006051, 0.37467675,
        0.34214149, 0.46434591, 0.39229868, 0.51077533, 0.09089699,
        0.3966171 , 0.44644824, 0.69851398, 0.7905684 , 0.09295659,
        0.30247091, 0.21690522, 0.87825967]),
 np.array([0.0953962 , 0.56177059, 0.17793624, 0.28742799, 0.90738614,
        0.87768855, 0.06095016, 0.99834119, 0.95450826, 0.99971139,
        0.6006195 , 0.03662663, 0.02768684, 0.27095244, 0.49760099,
        0.33378674, 0.30705255, 0.99998938]),
 np.array([0.26433515, 0.03879539, 0.3293281 , 0.99953991, 0.21444878,
        0.2394612 , 0.81118101, 0.40666512, 0.56271483, 0.66872237,
        0.58861186, 0.8508989 , 0.74187866, 0.54154622, 0.69768032,
        0.6111421 , 0.69323454, 0.03167497]),
 np.array([1.30265386e-01, 2.74884300e-01, 3.20929217e-01, 5.35719519e-01,
        3.93105981e-01, 7.97743535e-01, 1.02849871e-04, 5.89548298e-01,
        9.46993432e-01, 3.12257484e-01, 9.99463793e-01, 3.00827431e-04,
        6.79424910e-01, 5.51674146e-01, 3.74288016e-02, 9.97434351e-01,
        8.30407825e-01, 8.95372498e-01]),
 np.array([4.82230497e-01, 3.11385901e-03, 5.15568641e-02, 1.95955223e-01,
        1.14348617e-04, 9.99638746e-01, 2.08381319e-01, 9.22857253e-01,
        8.58436375e-01, 2.99287678e-01, 7.85079766e-01, 9.98708463e-01,
        9.78646269e-01, 2.81120403e-01, 5.40393384e-01, 7.64526716e-01,
        9.92343130e-01, 1.01478767e-02]),
 np.array([3.35892331e-01, 2.43978009e-01, 1.24854364e-01, 9.95720134e-01,
        9.99858165e-01, 9.99754427e-01, 6.58790016e-02, 7.94644062e-03,
        9.93689885e-01, 9.99671385e-01, 9.98994163e-01, 4.61414623e-01,
        8.59835080e-03, 8.92803148e-04, 4.82668577e-03, 3.09695211e-01,
        1.09023561e-01, 1.00000000e+00]),
 np.array([3.82680163e-01, 8.76837159e-01, 6.38588872e-02, 8.34246386e-01,
        3.77769520e-04, 6.46849026e-04, 9.24417441e-04, 5.32934975e-03,
        9.87331045e-01, 5.37821936e-01, 9.09838598e-03, 9.97153390e-01,
        9.99674514e-01, 1.11484056e-01, 9.98894645e-01, 9.95282044e-04,
        9.99904624e-01, 9.99870847e-01]),
 np.array([5.76507893e-01, 4.73693669e-03, 2.03255414e-01, 6.66341339e-01,
        4.35656892e-04, 9.98798847e-01, 9.99974181e-01, 9.25883567e-04,
        9.75864946e-01, 9.99951571e-01, 5.11567261e-02, 1.84650638e-02,
        9.98669172e-01, 3.80135656e-03, 8.01574652e-01, 9.99978668e-01,
        5.90268901e-01, 2.24348947e-04]),
 np.array([3.91359832e-01, 3.56961330e-01, 1.69710214e-01, 9.99336766e-01,
        9.98939588e-01, 8.37993771e-02, 1.23887083e-05, 9.99157993e-01,
        4.58362173e-01, 9.43333462e-05, 4.01077800e-02, 9.99988294e-01,
        9.99889954e-01, 2.95311597e-04, 1.72241117e-02, 7.60094600e-02,
        2.25009827e-03, 9.92195869e-01])]

# %% [markdown]
# ## No priors

# %%
key = fitter.param_dict_rev[fitter.paramstrings[2]]
key

# %%
fitter2.like.free_param_names


# %%
def compare_ev_nuisance(like_full, par_full, like_time, par_time):
    


# %%
coeff_cube = np.array(full_fit[:5])
coeff_no_cube = np.zeros_like(coeff_cube)
for ip in range(len(coeff_cube)):
    coeff_no_cube[ip] = fitter2.like.free_params[ip].value_from_cube(coeff_cube)

tau_full = fitter2.like.theory.model_igm.F_model.get_tau_eff(fitter2.like.data.z, over_coeff=coeff_no_cube[::-1])

tau_atz = np.zeros_like(fitter.like.data.z)
for iz in range(len(tau_atz)):
    coeff_no_cube = fitter.like.free_params[0].value_from_cube(attime_fit[iz][0])
    tau_atz[iz] = fitter.like.theory.model_igm.F_model.get_tau_eff(fitter.like.data.z[iz], over_coeff=coeff_no_cube)

plt.plot(fitter.like.data.z, tau_atz)
plt.plot(fitter.like.data.z, tau_full)

# plt.yscale("log")

# %%
coeff_cube = np.array(full_fit[5:7])
coeff_no_cube = np.zeros_like(coeff_cube)
for ip in range(len(coeff_cube)):
    coeff_no_cube[ip] = fitter2.like.free_params[5 + ip].value_from_cube(coeff_cube[ip])

tau_full = fitter2.like.theory.model_igm.T_model.get_gamma(fitter2.like.data.z, over_coeff=coeff_no_cube[::-1])

tau_atz = np.zeros_like(fitter.like.data.z)
for iz in range(len(tau_atz)):
    coeff_no_cube = fitter.like.free_params[2].value_from_cube(attime_fit[iz][2])
    tau_atz[iz] = fitter.like.theory.model_igm.T_model.get_gamma(fitter.like.data.z[iz], over_coeff=coeff_no_cube)

plt.plot(fitter.like.data.z, tau_atz)
plt.plot(fitter.like.data.z, tau_full)


# %%

# %%
def plot_z_at_time_params(fitter, attime_fit, save_fig=None):

    ofit = {}
    for par in fitter.paramstrings:
        ofit[fitter.param_dict_rev[par]] = 1

    weak_priors = {}
    
    fig, ax = plt.subplots(3, 4, figsize=(16, 16), sharex=True)
    ax = ax.reshape(-1)
    
    dict_out = {}
    jj = 0
    for ii, key in enumerate(fitter.paramstrings):
        if key not in out_mle[0]:
            continue
        dict_out[key] = np.zeros(len(out_mle))
        for iz in range(len(out_mle)):
            ax[jj].scatter(fitter.like.data.z[iz], out_mle[iz][key])
            dict_out[key][iz] = out_mle[iz][key]
        ax[jj].set_ylabel(key)
        ax[jj].set_xlabel(r"$z$")
        jj += 1
    
    jj = 0
    for ii, key in enumerate(fitter.paramstrings):
        if key not in dict_out:
            continue
        print(key, np.round(np.median(dict_out[key]), 4), np.round(np.std(dict_out[key]), 4))
        ax[jj].plot(fitter.like.data.z, fitter.like.data.z[:]*0 + np.median(dict_out[key]))
    
        # ind = np.argwhere(fitter.param_dict_rev[key] == list_props)[0,0]
        # w = np.abs(delta_chi2[ind])
        x = fitter.like.data.z.copy()[:5]
        y = dict_out[key].copy()[:5]
        w = np.ones_like(x)    
        fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
        for kk in range(3):
            mod = np.poly1d(fit)(x)
            std_mod = np.std(mod - y)
            # if "ln_x_" in fitter.param_dict_rev[key]:
            #     _ = (np.abs(y - mod) < 2 * std_mod) & (y > -8)
            # else:
            _ = np.abs(y - mod) < 2 * std_mod
            x = x[_]
            y = y[_]
            w = w[_]
            fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
        # ax[jj].plot(like.data.z, mod)
        mod = np.poly1d(fit)(fitter.like.data.z)    
        ax[jj].errorbar(fitter.like.data.z, mod, std_mod * 2)
        weak_priors[fitter.param_dict_rev[key] + "_cen"] = mod
        weak_priors[fitter.param_dict_rev[key] + "_std"] = std_mod
        jj += 1
        
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig("snr3_fid_weakp.png")
        plt.savefig("snr3_fid_weakp.pdf")
    return weak_priors


# %%
weak_priors = plot_z_at_time_params(fitter, out_mle)

# %%
# weak_priors

# %%
p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_mle_cube = []
out_chi2 = []
list_fix = ["ln_tau_0", "ln_sigT_kms_0", "ln_gamma_0", "ln_kF_0"]   

for ii in range(len(like.data.z)): 
# for ii in range(1): 
    print(ii)
    for par in fitter.like.free_params:
        if par.name not in list_fix:
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
            print(par.name, par.value, par.min_value, par.max_value)
    zmask = np.array([like.data.z[ii]])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=3)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

# %% [markdown]
# Priors, delta2chi = 30!; we had 644

# %%
print(np.sum(out_chi2))
out_chi2

# %%
weak2_priors = plot_z_at_time_params(fitter, out_mle)

# %%

# %%
p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_mle_cube = []
out_chi2 = []
for ii in range(len(like.data.z)):
# for ii in range(9, 10): 
    print(ii)
    for par in fitter.like.free_params:
        if par.name != "R_coeff_0":
            par.value = weak2_priors[par.name + "_cen"][ii]
            par.min_value = weak2_priors[par.name + "_cen"][ii] - 2 * weak2_priors[par.name + "_std"]
            par.max_value = weak2_priors[par.name + "_cen"][ii] + 2 * weak2_priors[par.name + "_std"]
            print(par.name, par.value, par.min_value, par.max_value)
    zmask = np.array([like.data.z[ii]])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=4)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%
print(np.sum(out_chi2))
out_chi2

# %%
weak3_priors = plot_z_at_time_params(fitter, out_mle, save_fig=True)

# %% [markdown]
# We had 675 original, weak priors 685, weak2priors 690

# %%

# %%

# %%

# %%
zmask

# %%
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

zmask = np.array([2.4])
fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)

# %%
# # %%time
fitter.run_sampler(pini=p0, zmask=zmask)

# %%
# diru = None
# plotter = Plotter(fitter, save_directory=diru, zmask=zmask)

# %%
# plotter.plots_minimizer()

# %%
print(np.sum(out_chi2))
out_chi2

# %%
weak3_priors = {}


# fig, ax = plt.subplots(4, 3, figsize=(10, 8))
# fig, ax = plt.subplots(8, 3, figsize=(10, 16), sharex=True)
fig, ax = plt.subplots(4, 4, figsize=(16, 16), sharex=True)
ax = ax.reshape(-1)
dict_out = {}
jj = 0
for ii, key in enumerate(fitter.paramstrings):
    if key not in out_mle[0]:
        continue
    dict_out[key] = np.zeros(len(out_mle))
    for iz in range(len(out_mle)):
        ax[jj].scatter(like.data.z[iz], out_mle[iz][key])
        dict_out[key][iz] = out_mle[iz][key]
    ax[jj].set_ylabel(key)
    ax[jj].set_xlabel(r"$z$")
    jj += 1

jj = 0
for ii, key in enumerate(fitter.paramstrings):
    if key not in dict_out:
        continue
    print(key, np.round(np.median(dict_out[key]), 4), np.round(np.std(dict_out[key]), 4))
    ax[jj].plot(like.data.z, like.data.z[:]*0 + np.median(dict_out[key]))

    # ind = np.argwhere(fitter.param_dict_rev[key] == list_props)[0,0]
    # w = np.abs(delta_chi2[ind])
    x = like.data.z.copy()
    y = dict_out[key].copy()
    w = np.ones_like(x)    
    fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
    for kk in range(3):
        mod = np.poly1d(fit)(x)
        std_mod = np.std(mod - y)
        _ = np.abs(y - mod) < 2 * std_mod
        x = x[_]
        y = y[_]
        w = w[_]
        fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
    # ax[jj].plot(like.data.z, mod)
    mod = np.poly1d(fit)(like.data.z)    
    ax[jj].errorbar(like.data.z, mod, std_mod * 2)
    weak3_priors[fitter.param_dict_rev[key] + "_cen"] = mod
    weak3_priors[fitter.param_dict_rev[key] + "_std"] = std_mod * 3
    jj += 1
    
plt.tight_layout()
plt.savefig("snr3_fid_weakp3.png")

# %% [markdown]
# Get priors

# %% [markdown]
# Normal units

# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/fid/chain_5/2.2/"
folder_priors = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/priors/chain_3/2.2/"
file = "fitter_results.npy"
# sampler_data = np.load(folder + file, allow_pickle=True).item()

# %%
# sampler_data.keys()

# %%
plotter = Plotter(fname_chain=folder_data + file, fname_priors=folder_priors + file)
# plotter.fitter.like.plot_p1d(residuals=True)

# %% [markdown]
# prior out of range?!!

# %%
# plotter.fitter.like.plot_p1d()

# %%
plotter.save_directory = folder_data

# %%
plotter.save_directory = None

# %% [markdown]
# #### priors: 1 and 2 sigma?

# %%
plotter.plot_corner()

# %%
plotter.save_directory = folder_data

# %%
only_plot = [
    '$\\mathrm{ln}\\,\\tau_0$',
    '$\\mathrm{ln}\\,\\sigma^T_0$',
    '$\\mathrm{ln}\\,\\gamma_0$',
    '$\\mathrm{ln}\\,k^F_0$',
    '$\\mathrm{R}_0$'
]
plotter.plot_corner_1z_natural(2.2, only_plot=only_plot)

# %%

# %%

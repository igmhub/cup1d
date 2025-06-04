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

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

# %%

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
zmask = np.array([like.data.z[0]])
# fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)
fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, restart=True, nsamples=1)
# zmask = np.array([2.4])
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %%

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

# %%

# diru = "test_snr3_11_1_1_1"
# diru = "test_snr3_3_3_1_1"
# diru = "test_snr3_3_2_2_2"
# diru = "test_snr3_3_2_1_1_cosmo"
diru = None
plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plots_minimizer()

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
n_metals = {
    "Lya_SiIII": [1, 1, 1],
    "Lya_SiIIa": [1, 1, 1],
    "Lya_SiIIb": [1, 1, 1],
    "SiIIa_SiIII": [1, 1, 1],
    "SiIIb_SiIII": [1, 1, 1],
    "SiIIa_SiIIb": [1, 1, 1],
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





# %%
args.set_baseline()

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
    print(ii)
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
fitter.mle

# %% [markdown]
# 35.7 66.1 43.3 48.7
#
# 258.25019606819825 [[52.62668059 71.89440875 61.29380909 59.75298414]] 245.56788256970646
#
# diff in chi2 of 51.8 (245.57-193.75)

# %% [markdown]
# No priors

# %%
print(np.sum(out_chi2))
out_chi2

# %% [markdown]
# New fid, Delta_chi2=12!!

# %%
print(np.sum(out_chi2))
out_chi2

# %% [markdown]
# FFT direct, starting random, 4 + fulllines 956 
#
# FFT fid, starting random, 4 + fulllines 1005 
#
# QMLE snr3, starting random, 4 + fulllines 633
# - 703 when assuming f is not redshift dependent!

# %%

plt.plot(zs, out_chi2, "o-", label="FFT SB1 direct")

# %%
# out_chi2_dir = out_chi2.copy()
# out_chi2_ind = out_chi2.copy()
# out_chi2_qmle_snr3 = out_chi2.copy()
# out_chi2_qmle_fid = out_chi2.copy()
plt.plot(out_chi2_dir, "o-", label="FFT SB1 direct")
plt.plot(out_chi2_ind, "o-", label="FFT SB1 model")
plt.plot(out_chi2_qmle_snr3, "o-", label="QMLE snr3 SB1 model")
plt.plot(out_chi2_qmle_fid, "o-", label="QMLE fid SB1 model")
plt.legend()
plt.xlabel(r"$z$")
plt.ylabel(r"$\chi^2$")
# plt.savefig("chi2_all.png")

# %%
# plotter.plot_p1d(zmask=zmask)

# %%
# diru = "fft_dirmetal_mpg_z_at_time_fulllines"
# diru = "fft_fid_mpg_z_at_time_fulllines"

# diru = "qmle_snr3_mpg_z_at_time_fulllines"
# diru = "qmle_snr3_mpg_z_at_time_fid_weakp"
# diru = "qmle_fid_mpg_z_at_time_fulllines"
diru = None

plotter = Plotter(fitter, save_directory=diru, zmask=zmask)

# %%
plotter.plot_p1d(out_mle_cube, z_at_time=True, plot_panels=True, residuals=True)

# %%
# for ii in range(len(zs)):
#     plotter.plot_illustrate_contaminants(out_mle_cube[ii].copy(), [zs[ii]], lines_use=lines_use)

# %%
lines_use = [
    "Lya_SiIII",
    "Lya_SiIIa",
    "Lya_SiIIb",
    "SiIIa_SiIIb",
    "SiIIa_SiIII",
    "SiIIb_SiIII",
]
plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.2], lines_use=lines_use)

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

# %% [markdown]
# ## No priors

# %%
def plot_z_at_time_params(fitter, out_mle):

    ofit = {}
    for par in fitter.paramstrings:
        ofit[fitter.param_dict_rev[par]] = 1

    weak_priors = {}
    
    fig, ax = plt.subplots(4, 4, figsize=(16, 16), sharex=True)
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
        x = fitter.like.data.z.copy()
        y = dict_out[key].copy()
        w = np.ones_like(x)    
        fit = np.polyfit(x, y, ofit[fitter.param_dict_rev[key]], w=w)
        for kk in range(3):
            mod = np.poly1d(fit)(x)
            std_mod = np.std(mod - y)
            if "ln_x_" in fitter.param_dict_rev[key]:
                _ = (np.abs(y - mod) < 2 * std_mod) & (y > -8)
            else:
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
    # plt.savefig("snr3_fid_weakp.png")
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
for ii in range(len(like.data.z)): 
# for ii in range(10, 11): 
    print(ii)
    for par in fitter.like.free_params:
        if par.name != "R_coeff_0":
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
            print(par.name, par.value, par.min_value, par.max_value)
    zmask = np.array([like.data.z[ii]])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=2)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %% [markdown]
# Priors, delta2chi = 2; we had 675

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
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%
weak3_priors = plot_z_at_time_params(fitter, out_mle)

# %% [markdown]
# We had 675 original, weak priors 685, weak2priors 690

# %%
print(np.sum(out_chi2))
out_chi2

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
diru = None
plotter = Plotter(fitter, save_directory=diru, zmask=zmask)

# %%
plotter.plots_minimizer()

# %%
plotter.plot_corner()

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

# %%

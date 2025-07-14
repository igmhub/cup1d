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

from cup1d.utils.utils import get_path_repo

from scipy.stats import chi2 as chi2_scipy

from lace.archive.nyx_archive import NyxArchive

# %%
from cup1d.nuisance.mean_flux_class import MeanFlux
from cup1d.nuisance.pressure_class import Pressure
from cup1d.nuisance.thermal_class import Thermal

# %% [markdown]
# ### Set arguments

# %%

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
print(data["P1Ds"].apply_blinding)
if data["P1Ds"].apply_blinding:
    print(data["P1Ds"].blinding)

# %%
# data["P1Ds"].apply_blinding = False
# data["P1Ds"].blinding = False

# %%
fname = None
fname = "p1d_qmle"
data["P1Ds"].plot_p1d(fname=fname)
if args.data_label_hires is not None:
    data["extra_P1Ds"].plot_p1d()


# %%
try:
    data["P1Ds"].plot_igm()
except:
    print("Real data, no true IGM history")

# %%

# %%

# %%

# %% [markdown]
# #### Set fiducial/initial options for the fit

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

# %% [markdown]
# ### Set likelihood

# %% [markdown]
#
# up to z=3.3
# 307 to 350 (349)
#
# "tau_eff" first or second, maybe more
# "sigT_kms" noisy, no idea
#
# "f_Lya_SiIII" second
# "s_Lya_SiIII" first 
#
# "f_Lya_SiII" first + 13, better with nodes
# "s_Lya_SiII" first 
#
# "f_SiIIa_SiIII" first
# "f_SiIIb_SiIII" first
#
# "f_SiIIa_SiIIb" first
# "s_SiIIa_SiIIb" second
#
# "HCD_damp1" second
# "HCD_damp4" first
#

# %%
# args.set_baseline(fit_type="all", fix_cosmo=False)
# args.set_baseline(fit_type="all", fix_cosmo=True)
# args.set_baseline(fit_type="wip", fix_cosmo=True, zmax=3.2)
# args.set_baseline(fit_type="wip", fix_cosmo=True, zmax=4.2)
# args.set_baseline()

# %%

# %%
# np.save("int_vals_modify.npy", vals_modify)

# %%
# fid_tau = np.array([-0.02388212235743159, -0.037639133847819584, -0.03397288381711758, -0.06906946783191442, -0.08050471235281417, -0.1020954274736767, -0.1091780615234634])
# fid_tau = np.array([-0.07694854367384515, -0.4639306935116183, -1.0627446006160628])
# fid_sigT = np.array([0.9833049985389459, -0.43913496622924253, 1.0701790380033116])

# %%
param_attime_all = np.load("fit_baseline_param_attime.npy", allow_pickle=True).item()

# %%
# param_attime_all

# %%
out_fits = np.load("res_weak_priors.npy", allow_pickle=True).item()
out_fits.keys()

# %%
param_attime_all = out_fits["param_attime_all"]
param_attime_all

# %%
# param_attime_all

# %%
import re

def split_string(s):
    match = re.match(r"^(.*)_(\d+)$", s)
    if match:
        return match.group(1), match.group(2)
    else:
        return s, None 


# %%
np.arange(2.2, 3.2001, 0.2)
# 7

# %% [markdown]
# # HERE

# %%

args.set_baseline(fit_type="wip", fix_cosmo=True, zmax=4.2)
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)
len(like.free_param_names)

# %%
free_params = like.free_params.copy()

for p in free_params:
    # if p.name.startswith("tau_eff"):
    #     ii = int(p.name[-1])
    #     p.value = fid_tau[ii]
    #     # pass
    # elif p.name.startswith("sigT_kms"):
    #     ii = int(p.name[-1])
    #     p.value = fid_sigT[ii]
    #     # pass
    # elif p.name[:-2] in vals_modify:
    #     ii = int(p.name[-1])
    #     p.value = vals_modify["pfit_" + p.name[:-2]][-(ii + 1)]
    #     if (p.value > p.max_value):
    #         p.max_value = p.value + 2
    #     elif (p.value < p.min_value):
    #         p.min_value = p.value -

    pname, iistr = split_string(p.name)
    ii = int(iistr)
    # if (pname == "f_SiIIa_SiIII") | (pname == "f_SiIIb_SiIII"):
    if pname in args.opt_props:
        p.fixed = False
        # p.value = param_attime_all[pname][ii] # note order for splines!!!
        p.value = new_vals[pname][ii] # optimized!

    if pname == "sigT_kms":
        p.value = 1
        
    if pname == "HCD_damp1":
        p.value = -1.2

    
    # elif pname in new_vals:
    #     p.fixed = True
    #     p.value = new_vals[pname][ii] # optimized!
    #     if (p.value > p.max_value):
    #         p.max_value = p.value + 2
    #     elif (p.value < p.min_value):
    #         p.min_value = p.value - 2
    # else:
    #     p.fixed = True
    #     # p.value = param_attime_all[p.name[:-2]][-(ii + 1)]
    #     p.value = param_attime_all[pname][ii] # note order for splines!!!
    #     if (p.value > p.max_value):
    #         p.max_value = p.value + 2
    #     elif (p.value < p.min_value):
    #         p.min_value = p.value - 2

    print(p.name, '\t', np.round(p.value, 3), '\t', np.round(p.min_value, 3), '\t', np.round(p.max_value, 3), '\t', p.Gauss_priors_width, p.fixed)

# %%
like.theory.model_igm.models["F_model"].reset_coeffs(free_params)
like.theory.model_igm.models["T_model"].reset_coeffs(free_params)
like.theory.model_cont.hcd_model.reset_coeffs(free_params)
like.theory.model_cont.metal_models["Si_mult"].reset_coeffs(free_params)
like.theory.model_cont.metal_models["Si_add"].reset_coeffs(free_params)

# %%
# for p in like.free_params:
#     print(p.name, '\t', np.round(p.value, 3), '\t', np.round(p.min_value, 3), '\t', np.round(p.max_value, 3), '\t', p.Gauss_priors_width, p.fixed)

# %%
# param_attime_all

# %%
for ii in range(len(free_params)):
    if free_params[ii].fixed:
        for jj, p in enumerate(like.free_params):
            if p.name == free_params[ii].name:
                like.free_params.pop(jj)
                like.free_param_names.pop(jj)
                break
for p in like.free_params:
    print(p.name)

# %% [markdown]
# #### Plot cov to pk

# %%
plot = False

if plot:
    like.plot_cov_to_pk(save_directory=".")
    # like.plot_correlation_matrix()

# %% [markdown]
# #### Plot corr matrix

# %%
plot = False

if plot:
    like.plot_correlation_matrix()

# %% [markdown]
# #### Plot hull

# %%
plot = False

if plot:
    like.plot_hull_fid()

# %% [markdown]
# #### Plot IGM

# %%
plot = False

if plot:
    like.plot_igm(cloud=True)

# %%
input_pars = like.sampling_point_from_parameters().copy()
input_pars

# %%
input_pars = fitter.mle_cube.copy()

# %% [markdown]
# Compare data and fiducial/starting model

# %%
# # %%time
# like.plot_p1d(plot_panels=True, residuals=True)
input_pars = like.sampling_point_from_parameters().copy()
# , plot_fname="test_weak"
like.plot_p1d(plot_panels=True, residuals=True, values=input_pars)
# like.plot_p1d(plot_panels=True, residuals=True)
# like.plot_p1d(residuals=True)

# %%
2.2 31.3 37 73.3
2.4 43.97 40 30.71
2.6 55.84 43 9.06 - 5
2.8 52.23 46 24.48 - 4
3.0 71.06 49 2.14 - 4
3.2 55.24 51 31.77 - 4
3.4 48.42 55 72.26 - 7
3.6 95.99 61 0.28 - 4
3.8 68.74 63 28.92 - 6
4.0 82.0 64 6.43
4.2 73.14 66 25.53

All 677.92 575 0.19

# %%

# %%
# like.plot_hull_fid(like_params=like.free_params)

# %% [markdown]
# 705 chi2 614 deg 67 params prob 0.63%

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
# 5 min to run baseline with nsamples=1

# %%
# %%time
# if like.truth is None:
#     # p0 = np.zeros(len(like.free_params)) + 0.5
#     p0 = np.array(list(like.fid["fit_cube"].values()))
# else:
#     p0 = np.array(list(like.truth["like_params_cube"].values()))*1.01
# p0 = np.array(list(like.fid["fit_cube"].values()))
p0 = like.sampling_point_from_parameters().copy()
# p0[0] = 0.5
# p0[1] = 0.5
# p0[2] = 0.5
# p0 = fitter.mle_cube.copy()
# p0[:] = 0.5
# fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)
# zmask = np.array([like.data.z[0]])
# fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=1)
fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, restart=True, nsamples=1)
# zmask = np.array([2.4])
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %% [markdown]
# 705 chi2 614 deg 67 params prob 0.63%
# 719 after removing 2 sigT, 3 HCD1

# %%
# su = 0
# for ii in range(len(data["P1Ds"].k_kms)):
#     print(len(data["P1Ds"].k_kms[ii]))
#     su += len(data["P1Ds"].k_kms[ii])
# su

# %% [markdown]
# 691 base 1-z at a time with weak priors. Without priors, 673

# %%
new_vals = {}

# %%

like_params = fitter.like.parameters_from_sampling_point(fitter.mle_cube)

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/full_"
folder = fold0 + "ev_baseline1z_params/taueff"
# folder = None
oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "ev_baseline1z_params/sigT"
oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "ev_baseline1z_params/Simult"
oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "ev_baseline1z_params/Siadd"
oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "ev_baseline1z_params/HCD"
oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

# mod = ocTmodel

# for key in mod:
#     if key in args.opt_props:
#         new_vals[key] = mod[key]
#         print(mod[key])

models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
for mod in models:
    for key in mod:
        if key in args.opt_props:
            new_vals[key] = mod[key]

# %%
# new_vals

# %%
np.save("opt_vals.npy", new_vals)

# %% [markdown]
# #### Latest

# %%

# diru = "test_snr3_11_1_1_1"
# diru = "test_snr3_3_3_1_1"
# diru = "test_snr3_3_2_2_2"
# diru = "test_snr3_3_2_1_1_cosmo"
# diru = "allz_snr3_nocosmo_full"
# diru = "qmle3_all_igmonly"
diru = None
plotter = Plotter(fitter, save_directory=diru)
# plotter.plots_minimizer()

# %%
2.2 31.3 37 73.3
2.4 43.97 40 30.71
2.6 55.84 43 9.06
2.8 52.23 46 24.48
3.0 71.06 49 2.14
3.2 55.24 51 31.77
3.4 48.42 55 72.26
3.6 95.99 61 0.28
3.8 68.74 63 28.92
4.0 82.0 64 6.43
4.2 73.14 66 25.53

All 677.92 575 0.19

# %%
plotter.plot_p1d(plot_panels=True, residuals=True)

# %%
like_params = fitter.like.parameters_from_sampling_point(fitter.mle_cube)
fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z, like_params)

# %%
# plotter.plot_igm()
# plotter.plot_p1d(plot_panels=True, residuals=True)
# plotter.plot_p1d_errors()
# plotter.plot_p1d(residuals=True)
# plotter.plot_p1d(residuals=True, plot_panels=True)
# plotter.plot_metal_cont(plot_data=True)

# if args.fix_cosmo == False:
    # plotter.plot_mle_cosmo()
# plotter.plots_minimizer()

# plotter.plot_metal_cont(plot_data=True)
# plotter.plot_hcd_cont(plot_data=True)

# %% [markdown]
# ## Optimize parameters of pipeline with z at a time fits 

# %%
from cup1d.optimize.baseline_ztime import run_grow_model_atz

# %%
# always run with zs starting with 2.2, need to change
zs = np.arange(2.2, 4.4, 0.2)
# zs = np.arange(2.2, 2.4, 0.2)
folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/qmle3_lpo/"
select_props = run_grow_model_atz(folder, zs, verbose=False)

for key in select_props:
    print(np.round(select_props[key]["z"], 2), np.sort(select_props[key]["name"]))

# %% [markdown]
# ## Run a z at a time fit

# %% [markdown]
# #### Same baseline for all z

# %%
# args.set_baseline(fit_type="all", fix_cosmo=False)
# args.set_baseline(fit_type="full")

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
# args.set_baseline(fit_type="at_a_time_igm")
# args.set_baseline(fit_type="at_a_time_orig")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

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

# %%

# %%

out_mle = []
out_mle_cube = []
out_chi2 = []
for ii in range(len(like.data.z)): 
# for ii in range(1): 
# for ii in range(4, 5): 
# for ii in range(2,3): 
# for ii in range(9, 10): 
# for ii in range(2, 3): 
    zmask = np.array([like.data.z[ii]])
    
    print()
    print(ii, like.data.z[ii])
    p0 = np.array(list(like.fid["fit_cube"].values()))
    # p0 = np.array([0.5, 0.8])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=6)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

ndeg_all = 0
props = []
chi2_all = 0
for ii in range(len(out_chi2)):
    ndeg = len(like.data.k_kms[ii]) - len(out_mle_cube[ii])
    prob = chi2_scipy.sf(out_chi2[ii], ndeg)
    print(like.data.z[ii], '&', np.round(out_chi2[ii], 2), '&', ndeg, '&', np.round(prob*100, 2), '\\\\')
    ndeg_all += ndeg
    chi2_all += out_chi2[ii]
    props.append(prob)
prob = chi2_scipy.sf(chi2_all, ndeg_all)
print()
print("All", '&', np.round(chi2_all, 2), '&', ndeg_all, '&', np.round(prob*100, 2), '\\\\')
prob

# %%

# diru = "igm_1z_snr3_nocosmo"
diru = "orig_1z_snr3_nocosmo"
plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube, plot_panels=True, residuals=True, z_at_time=True)

# %%

# plotter.plot_illustrate_contaminants_cum(out_mle_cube[0].copy(), np.array([2.4]))

# %%

# plotter.plot_p1d(residuals=True, zmask=zmask)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.2], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.4], lines_use=lines_use)
# plotter.plot_illustrate_contaminants_each(out_mle_cube[0].copy(), np.array([2.2]))
# plotter.plot_illustrate_contaminants(test, [2.4], lines_use=lines_use)

# %% [markdown]
# #### Different baseline as a function of z

# %%

out_mle = []
out_mle_cube = []
out_chi2 = []
for ii in range(len(data["P1Ds"].z)): 
# for ii in range(10, 11): 
# for ii in range(6, 7): 
# for ii in range(2,3): 
# for ii in range(9, 10): 
# for ii in range(2, 3): 
    zmask = np.array([data["P1Ds"].z[ii]])

    args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

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
        
        # if p.name[:-2] == "HCD_damp4":
        #     if p.name[:-2] in vals_modify:
        #         p.value = vals_modify[p.name[:-2]][ii]
        #         p.min_value = p.value - 1e-3
        #         p.max_value = p.value + 1e-3
            
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
    
    print(ii, like.data.z[ii])
    p0 = np.array(list(like.fid["fit_cube"].values()))
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=6)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

ndeg_all = 0
props = []
chi2_all = 0
for ii in range(len(out_chi2)):
    ndeg = len(like.data.k_kms[ii]) - len(out_mle_cube[ii])
    prob = chi2_scipy.sf(out_chi2[ii], ndeg)
    print(like.data.z[ii], '&', np.round(out_chi2[ii], 2), '&', ndeg, '&', np.round(prob*100, 2), '\\\\')
    ndeg_all += ndeg
    chi2_all += out_chi2[ii]
    props.append(prob)
prob = chi2_scipy.sf(chi2_all, ndeg_all)
print()
print("All", '&', np.round(chi2_all, 2), '&', ndeg_all, '&', np.round(prob*100, 2), '\\\\')
prob

# %%
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
like1 = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

out_mle_cube_reformat = []
for ii in range(len(data["P1Ds"].z)):
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
    like2 = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    _cube = np.zeros(len(all_props))
    for jj, prop in enumerate(like1.free_param_names):
        if prop in like2.free_param_names:
            ind = np.argwhere(prop == np.array(like2.free_param_names))[0,0]
            _cube[jj] = out_mle_cube[ii][ind]
    out_mle_cube_reformat.append(np.array(_cube))

# %%
diru = "first_1z_snr3_nocosmo"

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

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
fitter.mle_cube = out_mle_cube[ii]

plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube_reformat, plot_panels=True, residuals=True, z_at_time=True)

# %%
# dir_out = {
#     "mle_cube":out_mle_cube,
#     "mle":out_mle,
#     "chi2":out_chi2,
#     "mle_cube_reformat":out_mle_cube_reformat,
# }
# np.save("first_1z_snr3_nocosmo/res.npy", dir_out)

# %%
dir_out = np.load("first_1z_snr3_nocosmo/res.npy", allow_pickle=True).item()
out_mle_cube = dir_out["mle_cube"]

# %%
args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

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

mask = np.arange(11)

like_params = []
for mle_cube in out_mle_cube_reformat:
    like_params.append(fitter.like.parameters_from_sampling_point(mle_cube))

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/first_1z_snr3_nocosmo/"
folder = fold0 + "taueff"
oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "sigT"
oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Simult"
oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Siadd"
oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "HCD"
oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
param_attime_all = {}
for mod in models:
    for key in mod:
        param_attime_all[key] = mod[key]

# %%
np.save("first_1z_snr3_nocosmo/fit_baseline_param_attime.npy", param_attime_all)

# %%

# %%
# plotter.plot_p1d(zmask=zmask)
# plotter.plot_metal_cont(smooth_k=False, plot_data=True, zrange=[2.3, 2.5], plot_panels=False)
# plotter.plot_metal_cont(smooth_k=False, plot_data=True, zrange=[2.9, 3.1], plot_panels=False)

# %% [markdown]
# ### Plot evolution of parameters with z

# %%
from cup1d.optimize.plot_params_ztime import plot_z_at_time_params

weak1_priors = plot_z_at_time_params(fitter, out_mle)

# weak2_priors = plot_z_at_time_params(fitter, out_mle)

# %%
np.save("first_1z_snr3_nocosmo/weak_priors.npy", weak1_priors)

# %%
weak_priors = weak1_priors.copy()

# %% [markdown]
# #### Redo 1 z at time fits using weak_priors

# %%
p0 = np.array(list(like.fid["fit_cube"].values()))
out_mle = []
out_mle_cube = []
out_chi2 = []
list_fix = ["tau_eff_0", "sigT_kms_0", "gamma_0", "kF_kms_0"]

for ii in range(len(like.data.z)): 
# for ii in range(10, 11):
    print(ii)

    zmask = np.array([data["P1Ds"].z[ii]])

    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

    like = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )    
    
    for par in like.free_params:
        if par.name not in list_fix:
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
        else:
            if (par.value < par.max_value) & (par.value > par.min_value):
                par.value = weak_priors[par.name + "_cen"][ii]
        print(par.name, par.value, par.min_value, par.max_value)
    
    fitter = Fitter(
        like=like,
        rootdir=output_dir,
        nburnin=args.n_burn_in,
        nsteps=args.n_steps,
        parallel=args.parallel,
        explore=args.explore,
        fix_cosmology=args.fix_cosmo,
    )

    p0 = like.sampling_point_from_parameters().copy()
            
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=5)
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, zmask=zmask, restart=True)
    out_mle.append(fitter.mle)
    out_mle_cube.append(fitter.mle_cube)
    out_chi2.append(fitter.mle_chi2)
    # plotter = Plotter(fitter, zmask=zmask, save_directory="mpg_baseline/"+str(ii))
    # plotter.plots_minimizer()

# %%

ndeg_all = 0
props = []
chi2_all = 0
for ii in range(len(out_chi2)):
    ndeg = len(like.data.k_kms[ii]) - len(out_mle_cube[ii])
    prob = chi2_scipy.sf(out_chi2[ii], ndeg)
    print(like.data.z[ii], '&', np.round(out_chi2[ii], 2), '&', ndeg, '&', np.round(prob*100, 2), '\\\\')
    ndeg_all += ndeg
    chi2_all += out_chi2[ii]
    props.append(prob)
prob = chi2_scipy.sf(chi2_all, ndeg_all)
print()
print("All", '&', np.round(chi2_all, 2), '&', ndeg_all, '&', np.round(prob*100, 2), '\\\\')
prob

# %%

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
like1 = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

out_mle_cube_reformat = []
for ii in range(len(data["P1Ds"].z)):
# for ii in range(10,11):
    args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
    args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")
    like2 = set_like(
        data["P1Ds"],
        emulator,
        args,
        data_hires=data["extra_P1Ds"],
    )
    
    for par in like2.free_params:
        if par.name not in list_fix:
            par.value = weak_priors[par.name + "_cen"][ii]
            par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
            par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
        else:
            if (par.value < par.max_value) & (par.value > par.min_value):
                par.value = weak_priors[par.name + "_cen"][ii]
    _cube = np.zeros(len(all_props))
    for jj, prop in enumerate(like1.free_param_names):
        if prop in like2.free_param_names:
            ind = np.argwhere(prop == np.array(like2.free_param_names))[0,0]
            value = like2.free_params[ind].value_from_cube(out_mle_cube[ii][ind])
            in_cube = like1.free_params[jj].get_value_in_cube(value)
            print(prop, like1.free_params[jj].name)
            if in_cube < 0:
                in_cube = 0
            _cube[jj] = in_cube
    out_mle_cube_reformat.append(np.array(_cube))

# %%
diru = "priors_1z_snr3_nocosmo"

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

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
fitter.mle_cube = out_mle_cube[ii]

plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube_reformat, plot_panels=True, residuals=True, z_at_time=True)

# %%

# %%
dir_out = {
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
    "mle_cube_reformat":out_mle_cube_reformat,
}
np.save("priors_1z_snr3_nocosmo/res.npy", dir_out)

# %%
args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
ii = 0
args.set_baseline(ztar=data["P1Ds"].z[ii], fit_type="at_a_time")

like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)

for par in like.free_params:
    if par.name not in list_fix:
        par.value = weak_priors[par.name + "_cen"][ii]
        par.min_value = weak_priors[par.name + "_cen"][ii] - 2 * weak_priors[par.name + "_std"]
        par.max_value = weak_priors[par.name + "_cen"][ii] + 2 * weak_priors[par.name + "_std"]
    else:
        if (par.value < par.max_value) & (par.value > par.min_value):
            par.value = weak_priors[par.name + "_cen"][ii]

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

mask = np.arange(11)

like_params = []
for mle_cube in out_mle_cube_reformat:
    like_params.append(fitter.like.parameters_from_sampling_point(mle_cube))

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/priors_1z_snr3_nocosmo/"
folder = fold0 + "taueff"
oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "sigT"
oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Simult"
oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Siadd"
oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "HCD"
oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
param_attime_all = {}
for mod in models:
    for key in mod:
        param_attime_all[key] = mod[key]

# %%
np.save("priors_1z_snr3_nocosmo/fit_baseline_param_attime.npy", param_attime_all)

# %%

# %%
# 
weak2_priors = plot_z_at_time_params(fitter, out_mle)

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

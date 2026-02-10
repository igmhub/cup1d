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


# %% [markdown]
# ### Fisher forecast analysis

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

# archive_mock = set_archive(training_set="Pedersen21")
# dat = archive_mock.get_testing_data("mpg_central")

# %%

# %% [markdown]
# ### Mock analysis

# %%
# nyx_training_set = "models_Nyx_Sept2025_include_Nyx_fid_rseed"
# archive_mock = set_archive(training_set=nyx_training_set)
# pip = Pipeline(args, archive=archive_mock)
pip = Pipeline(args)

# %%
pip.fitter.like.plot_p1d()

# %%
# dict_out = {
#     "k_kms": pip.fitter.like.data.k_kms,
#     "Pk_kms": pip.fitter.like.data.Pk_kms,
#     "cov_Pk_kms": pip.fitter.like.data.cov_Pk_kms,
#     "z": pip.fitter.like.data.z,
# }
# np.save("smooth_" + data_label + ".npy", dict_out)

# data = np.load("smooth_" + data_label + ".npy", allow_pickle=True).item()
# data.keys()

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
# p0[:] = 0.5
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
for par in free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %%
pip.fitter.like.plot_p1d(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %% [markdown]
# ### Data analysis

# %%



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

# name_variation = "metals_z"
# name_variation = "all_inflate"
# name_variation = "Turner24"
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


data_label = "DESIY1_QMLE3"
# data_label = "DESIY1_QMLE"
# data_label = "DESIY1_FFT3_dir"

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

pip = Pipeline(args)


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

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%
pip.fitter.like.plot_p1d(
    p0, 
    residuals=True, 
    plot_panels=True, 
    print_chi2=False, 
    fix_cosmo=False, 
    plot_fname=None
)


# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
chain = np.load(base + folder + "chain.npy")


# %%
pip.run_minimizer(p0)

# %%

# %%

# %%

# %%

# %%

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

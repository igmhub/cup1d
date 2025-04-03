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

# %%
combine_posteriors = False
if combine_posteriors:

    folder_in = "/home/jchaves/Proyectos/projects/lya/data/forecast/mock_challenge_DESIY1_1.0fxh/"
    file_mpg_nn = folder_in + "CH24/no_emu_err/chain_1/fitter_results.npy"
    file_mpg_gp = folder_in + "CH24_mpg_gp/no_emu_err/chain_1/fitter_results.npy"
    file_nyx_nn = folder_in + "CH24_NYX/no_emu_err/chain_1/fitter_results.npy"
    file_nyx_gp = folder_in + "CH24_nyx_gp/no_emu_err/chain_1/fitter_results.npy"
    files = [file_mpg_nn, file_mpg_gp, file_nyx_nn, file_nyx_gp]
    arr_lab = ["mpg_nn", "mpg_gp", "nyx_nn", "nyx_gp"]
    
    arr_par = []
    for file in files:
        res = np.load(file, allow_pickle=True).item()
        ds = res["fitter"]["blobs"]["Delta2_star"].reshape(-1)
        ns = res["fitter"]["blobs"]["n_star"].reshape(-1)
        
        par = np.zeros((ds.shape[0], 2))
        par[:, 0] = ds
        par[:, 1] = ns
        arr_par.append(par)
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    for ii in range(len(arr_par)):
        corner(
            arr_par[ii], 
            fig=fig, 
            color="C"+str(ii), 
            plot_datapoints=False, 
            plot_density=False, 
            levels=np.array([0.393, 0.864]),
        );
        ax[0, 0].scatter(
            res["truth"]['$\\Delta^2_\\star$'],
            0,
            # res["truth"]['$n_\\star$'],
            color="C"+str(ii),
            label=arr_lab[ii],
            s=10
        )
        
    ax[1, 0].axhline(
        res["truth"]['$n_\\star$'],
        color="k",
        linestyle=":",
    )
    ax[1, 0].axvline(
        res["truth"]['$\\Delta^2_\\star$'],
        color="k",
        linestyle=":",
    )
    ax[0, 0].axvline(
        res["truth"]['$\\Delta^2_\\star$'],
        color="k",
        linestyle=":",
    )
    ax[1, 1].axvline(
        res["truth"]['$n_\\star$'],
        color="k",
        linestyle=":",
    )
    ax[0, 0].legend()
    
    
    labels=[r"$\Delta^2_\star$", r"$n_\star$"]
    ax[1, 0].set_xlabel(labels[0], fontsize=15)
    ax[1, 0].set_ylabel(labels[1], fontsize=15)
    ax[1, 1].set_xlabel(labels[0], fontsize=15)
    
    plt.tight_layout()
    plt.savefig("forecast_emus.pdf")

# %%

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

# args = Args(emulator_label="CH24_nyx_gp", training_set="Nyx23_Jul2024")
args = Args(emulator_label="CH24_mpg_gp", training_set="Cabayol23")
# args = Args(emulator_label="CH24", training_set="Cabayol23")
# args = Args(emulator_label="CH24_NYX", training_set="Nyx23_Jul2024")
output_dir = "."

emulator = set_emulator(
    emulator_label=args.emulator_label,
)
archive = None

# %%

# %% [markdown]
# ### Set emulator

# %% [markdown]
# #### Set either mock data or real data

# %%
# for forecast, just start label of observational data with mock
choose_forecast = True 
# choose_forecast = False
# to analyze data from simulations
choose_mock = False
# to analyze data from observations
choose_data = False
# to analyze data from mock challenge
choose_challenge = False
# to analyze data from desiy1
choose_desiy1 = False

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
    # true_sim = "nyx_central"
    true_sim = "mpg_central"
    true_sim_cosmo = "mpg_central"
    args.true_label_mF=true_sim
    args.true_label_T=true_sim
    args.true_label_kF=true_sim
    
    # true_sim_cosmo = "Planck18_low"
    # args.true_label_kF="kF_both"
    args.mF_model_type="chunks"
    true_cosmo = set_cosmo(cosmo_label=true_sim_cosmo)
    
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
    # args.data_label="nyx_central"
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
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    # fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_bar_ic_grid_3.fits.gz"
    # fname = "mockchallenge-0."+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
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
    # in NERSC
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"
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
data["P1Ds"].plot_p1d()
if args.data_label_hires is not None:
    data["extra_P1Ds"].plot_p1d()

# %%
try:
    data["P1Ds"].plot_igm()
except:
    print("Real data, no true IGM history")

# %% [markdown]
# #### Set fiducial/initial options for the fit

# %%
# # std
# p2 = np.array([-0.00109798,  0.00691753])
# # bias + std
# p2 = np.array([-0.0058123,  0.0237336])

# %%
# cosmology
args.ic_correction=False

# args.emu_cov_factor = None
args.emu_cov_factor = 1
# args.emu_cov_type = "diagonal"
# args.emu_cov_type = "block"
args.emu_cov_type = "full"
# 1 sigma
# args.emu_cov_factor = np.array([-6.93721149e-02, -2.43173756e-04,  4.91541477e-02, -2.90469219e+00])
# 2 sigma
# args.emu_cov_factor = np.array([-4.78752514e-02, -8.60779305e-04,  5.60692510e-02, -2.56919148e+00])


# args.fid_cosmo_label="mpg_2"
# if "Nyx" in emulator.emulator_label:
#     args.fid_cosmo_label="nyx_central"
#     args.fid_igm_label="nyx_central"
#     args.vary_alphas=True
# else:
#     args.fid_cosmo_label="mpg_central"
#     args.fid_igm_label="mpg_central"
#     args.vary_alphas=False

args.fix_cosmo=False
# args.fix_cosmo=True
args.vary_alphas=False
# args.vary_alphas=True
# args.fid_cosmo_label="Planck18"
# args.fid_cosmo_label="Planck18_low"
# sim_fid = "nyx_central"
sim_fid = "mpg_central"
args.fid_cosmo_label=sim_fid
# sim_fid = "nyx_seed"
# sim_fid = "nyx_3"
# args.fid_cosmo_label=sim_fid
args.fid_label_mF=sim_fid
args.fid_label_T=sim_fid
args.fid_label_kF=sim_fid
# args.fid_label_kF="kF_both"


# args.fid_cosmo_label="mpg_central"
# args.fid_sim_igm_label_mF="mpg_central"
# args.fid_sim_igm_label_T="mpg_central"
# args.fid_sim_igm_label_kF="mpg_central"

# args.fid_cosmo_label="mpg_central"
# args.fid_label_mF="mpg_central"
# args.fid_label_mF="nyx_0"
# args.fid_label_T="mpg_central"
# # args.fid_sim_igm_label_T="nyx_0"
# args.fid_sim_igm_label_kF="mpg_central"

# args.fid_cosmo_label="nyx_seed"

# args.fid_igm_label="mpg_22"
# args.fid_cosmo_label="mpg_22"

# args.fid_igm_label="mpg_central"
# args.ic_correction=True
# args.fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# args.use_star_priors = None
# args.use_star_priors = {}
# Planck18 0.354 -2.300 -0.2155
# 5 sigma 0.056 0.011 0.0028
# blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params()
# amin = blob["alpha_star"] - 0.0028
# amax = blob["alpha_star"] + 0.0028
# args.use_star_priors["alpha_star"] = [amin, amax]


# IGM
# args.fid_igm_label="nyx_13"
# args.fid_igm_label="mpg_2"
# args.fid_igm_label="nyx_seed"
# args.fid_igm_label="nyx_3"
# args.fid_igm_label="nyx_3_1"
if choose_data == False:
    args.igm_priors = "hc"
else:
    args.igm_priors = "data"

# args.hcd_model_type = "Rogers2017"
args.hcd_model_type = "new"
# args.mF_model_type = "pivot"
args.mF_model_type = "chunks"
# contaminants
# from 1 to 6, -11 to -4
# from -5 to 0
# args.fid_HCD=[0, -2]
# from -5 to 2
# args.fid_SN=[0, -4]
# args.fid_AGN=[0, -5]

    
args.n_tau=len(data["P1Ds"].z)
args.n_sigT=1
args.n_gamma=1
args.n_kF=1

args.n_x_SiIII=0
args.n_d_SiIII=0
args.n_a_SiIII=0
args.n_d_dla = 0
args.n_s_dla = 0

# args.n_x_SiIII=1
# args.n_d_SiIII=1
# args.n_a_SiIII=1
# args.n_d_dla = 1
# args.n_s_dla = 1
args.fid_SiIII_X=[0, -10] # fine
args.fid_SiIII_D=[0, 5]
args.fid_SiIII_A=[0, 1]
args.fid_A_damp = [0, -9]
args.fid_A_scale = [0, 5]

# args.n_x_SiIII=1
# args.n_d_SiIII=1
# args.n_a_SiIII=0
# args.n_d_dla = 1
# args.n_s_dla = 1
# args.fid_SiIII_X=[0, -5]
# args.fid_SiIII_D=[0, 5]
# args.fid_SiIII_A=[0, 1]
# args.fid_A_damp = [0, -1]
# args.fid_A_scale = [0, 5]
# args.n_x_SiIII=2
# args.n_d_SiIII=2
# args.n_a_SiIII=2
# args.n_d_dla = 2
# args.n_s_dla = 2
# args.n_tau=2
# args.n_sigT=2
# args.n_gamma=2
# args.n_kF=2
# args.n_x_SiIII = 2
# args.n_d_SiIII = 2
# args.n_a_SiIII = 2

# args.n_SiII = 0
# args.n_dla=0
# args.n_sn=0
# args.n_agn=0


free_parameters = set_free_like_parameters(args, emulator.emulator_label)
free_parameters

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
# for ii in range(len(cov0)):
for ii in range(10):
    plt.plot(like.data.k_kms[ii], np.diag(cov1[ii])/np.diag(cov0[ii]), label=like.data.z[ii])
    # plt.plot(like.data.k_kms[ii], np.diag(like.cov_Pk_kms[ii]), "C"+str(ii), label=like.data.z[ii])
    # plt.plot(like.data.k_kms[ii], np.diag(like.cov_Pk_kms[ii]) - np.diag(like.emu_cov_Pk_kms[ii]), "C"+str(ii)+"--")
plt.xscale("log")
plt.yscale("log")
plt.legend()


# %%
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# plt.imshow(correlation_from_covariance(like.emu_full_cov_Pk_kms))
plt.imshow(correlation_from_covariance(like.full_cov_Pk_kms))
plt.colorbar()

# %%
# plt.imshow(correlation_from_covariance(like.emu_full_cov_Pk_kms))
plt.imshow(correlation_from_covariance(like.full_cov_Pk_kms-like.emu_full_cov_Pk_kms))
plt.colorbar()

# %%
is_pos_def(like.full_cov_Pk_kms)

# %%
# plt.plot(like.data.full_k_kms, like.full_cov_Pk_kms[0])
# plt.plot(like.data.full_k_kms, like.full_cov_Pk_kms[0]-like.emu_full_cov_Pk_kms[0])
# plt.plot(like.data.full_k_kms, like.emu_full_cov_Pk_kms[0])
ii =20
plt.plot(like.full_cov_Pk_kms[ii])
plt.plot(like.full_cov_Pk_kms[ii]-like.emu_full_cov_Pk_kms[ii])
plt.plot(like.emu_full_cov_Pk_kms[ii])
# plt.xscale("log")
plt.yscale("log")
plt.xlim(

# %%
# emu_call, M_of_z = like.theory.get_emulator_calls(like.data.z)
# p1 = np.zeros(
#     (
#         like.theory.hull.nz,
#         len(like.theory.hull.params),
#     )
# )
# for jj, key in enumerate(like.theory.hull.params):
#     p1[:, jj] = emu_call[key]

# like.theory.hull.plot_hulls(p1)

# %%
# like.plot_igm(cloud=True)

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %%
# emu_call = np.load("emu_call_fiducial.npy", allow_pickle=True).item()
# emu_call

# %%

# %% [markdown]
# Compare data and fiducial/starting model

# %%
# like.plot_p1d(residuals=False)
like.plot_p1d(residuals=True)

# %%

# %% [markdown]
# ### Relative error

# %%
for ii in range(10):
    col = "C"+str(ii)
    print(data["P1Ds"].z[ii])
    rat = np.sqrt(np.diag(data["P1Ds"].cov_Pk_kms[ii]))/data["P1Ds"].Pk_kms[ii]
    plt.plot(data["P1Ds"].k_kms[ii], rat, col)
    
    dkms_dMpc = like.theory.fid_cosmo["cosmo"].dkms_dMpc(
        like.data.z[ii]
    )
    logk_Mpc = np.log(like.data.k_kms[ii] * dkms_dMpc)
    rat2 = np.exp(np.poly1d(like.emu_cov_factor)(logk_Mpc))
    
    plt.plot(data["P1Ds"].k_kms[ii], rat2, col+"--")
    # plt.plot(data["P1Ds"].k_kms[ii], np.sqrt(rat**2 + rat2**2))
plt.xscale("log")
plt.yscale("log")

# %%
# z = like.data.z
# k_kms = like.data.k_kms
# like.theory.model_cont.agn_model.plot_contamination(z, k_kms)

# %%
# z = like.data.z
# k_kms = like.data.k_kms
# like.theory.model_cont.hcd_model.plot_contamination(z, k_kms);

# %%
# like.plot_igm(cloud=True)

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
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %% [markdown]
# For Nyx fiducial
#
# ['As', 'ns', 'nrun', 'ln_tau_0', 'ln_tau_1', 'ln_tau_2', 'ln_tau_3', 'ln_tau_4', 'ln_tau_5', 'ln_tau_6', 'ln_tau_7', 'ln_tau_8', 'ln_tau_9', 'ln_tau_10', 'ln_sigT_kms_0', 'ln_gamma_0', 'ln_kF_0']
#
# ['As', 'ns', 'nrun', 'ln_tau_0', 'ln_tau_1', 'ln_tau_2', 'ln_tau_3', 'ln_tau_4', 'ln_tau_5', 'ln_tau_6', 'ln_tau_7', 'ln_tau_8', 'ln_tau_9', 'ln_tau_10', 'ln_sigT_kms_0', 'ln_gamma_0', 'ln_kF_0', 'ln_x_SiIII_0', 'ln_d_SiIII_0', 'a_SiIII_0', 'ln_A_damp_0', 'ln_A_scale_0']

# %% [markdown]
# #### 2 min for the fit when varying all taus

# %% [markdown]
# For DESI FFT
#
#
# Nyx 378.97810084186125
# args.n_tau=11
# args.n_sigT=1
# args.n_gamma=1
# args.n_kF=1
# args.n_x_SiIII=1
# args.n_d_SiIII=1
# args.n_a_SiIII=0 # fid 1
# args.n_d_dla = 1
# args.n_s_dla = 1
#
# Cabayol 366.31532112361907
# args.n_tau=10
# args.n_sigT=1
# args.n_gamma=1
# args.n_kF=1
# args.n_x_SiIII=1
# args.n_d_SiIII=1
# args.n_a_SiIII=1
# args.n_d_dla = 1
# args.n_s_dla = 1

# %%
# fitter.save_fitter()

# %%
plotter = Plotter(fitter, save_directory=None)
if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()
plotter.plots_minimizer()

# %%
# fitter.write_chain_to_file()

# %%
# fft Cabayol23 free cosmo z_max = 4.3
# SiIII dSiIII DLA
# 1 1 2 1502
# 1 0 2 2521 # We need 1 params for damping! dchi2 = 1000
# 1 1 0 2272 # We need (at least) 1 param for DLAs! dchi2 = 700
# 1 1 1 1700 # We need 2 params for DLAs dchi2 = 100
# 0 0 0 6726

# fft Planck18 z_max = 4.3
# Cabayol23+ 1 1 2 1504
# pedersen23 1 1 2 1501
# Nyx_cov    1 1 2 1667

# fft Planck18 zmin=2.3 z_max = 4.3
# Nyx_cov    1 1 2 1327

# qmle Planck18 z_max = 4.3
# Cabayol23+ 1 1 2 1813
# Nyx_cov    1 1 2 2100

# %%
# fil = np.load(fitter.save_directory + "/minimizer_results.npy", allow_pickle=True).item()
# for key in fil:
#     print(key, fil[key])

# %%
# fitter.save_minimizer()

# %%

# %%

# %%
plotter.plot_p1d(residuals=False, plot_every_iz=1)
plotter.plot_p1d(residuals=True, plot_every_iz=1)

# %%
plotter.plot_igm()

# %%

# %%
# plotter.plot_hcd_cont(plot_data=True)

# %%
# plotter.plot_metal_cont(smooth_k=True, plot_data=True)

# %%
# plotter.plot_agn_cont(plot_data=True)

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/test_qmle/"
folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/ca_test_fft_111/"
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/ca_test_qmle_112/"
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/pe_test_fft_112/"
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/nyx_test_fft_112/"
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/nyx_test_qmle_112/"
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/nyx_zmin_test_fft_112/"
plotter = Plotter(fitter, save_directory=folder)

# %%
plotter.plots_minimizer()


# %% [markdown]
# ### Run sampler
# It takes less than 2 min on my laptop without any parallelization

# %%
# %%time

def func_for_sampler(p0):
    res = fitter.like.get_log_like(values=p0, return_blob=True)
    return res[0], *res[2]

run_sampler = True
if run_sampler:    
    _emcee_sam = fitter.run_sampler(pini=fitter.mle_cube, log_func=func_for_sampler)

# %%
fitter.write_chain_to_file()

# %%
plotter = Plotter(fitter, save_directory=None)

# %%

# %%

# %%

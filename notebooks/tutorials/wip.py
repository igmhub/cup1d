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

from cup1d.likelihood.pipeline import (
    set_archive,
    set_P1D,
    set_cosmo,
    set_free_like_parameters,
    set_like,
)
from cup1d.p1ds.data_DESIY1 import P1D_DESIY1

from cup1d.likelihood.input_pipeline import Args

# %%
from cup1d.nuisance import AGN_model
import cup1d
import os

# %% [markdown]
# ### Set emulator

# %%
# set output directory for this test
output_dir = "."

# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# the nyx emulator has not properly been validated yet
# path nyx files in NERSC /global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/
args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Jul2024")

archive = set_archive(args.training_set)

emulator = set_emulator(
    emulator_label=args.emulator_label,
    archive=archive,
)

if emulator.emulator_label == "Nyx_alphap":
    emulator.list_sim_cube = archive.list_sim_cube
    emulator.list_sim_cube.remove("nyx_14")
else:
    emulator.list_sim_cube = archive.list_sim_cube

# %% [markdown]
# #### Set either mock data or real data

# %%
choose_forecast = False
choose_mock = False
choose_data = False
choose_challenge = False
choose_desiy1 = True

if choose_forecast:
    # for forecast, just start label of observational data with mock
    args.data_label = "mock_Chabanier2019"
    # args.data_label="mock_Karacayli2024"
    # args.data_label_hires = "mock_Karacayli2022"

    # you need to provide true cosmology, IGM history, and contaminants
    true_cosmo = set_cosmo(cosmo_label="nyx_central")
    # true_cosmo = set_cosmo(cosmo_label="mpg_central")
    # args.true_igm_label="mpg_central"
    args.true_igm_label="nyx_central"
    # from -11 to -4
    args.true_SiIII=[[0, 0], [-10, -10]]
    args.true_SiII=[[0, 0], [-10, -10]]
    # from -7 to 0
    args.true_HCD=[0, -6]
    # from -5 to 2
    args.true_SN=[0, -4]
    # from -5 to 1.5
    args.true_AGN=[0, -5]
elif choose_mock:    
    true_cosmo=None
    # to analyze data from simulations
    args.data_label = "mpg_central"    
    # args.data_label="nyx_central"
    # args.data_label="nyx_seed"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None

    # provide cosmology only to cull the data
    args.true_cosmo_label="mpg_central"
    # args.true_cosmo_label="nyx_central"
    # args.true_cosmo_label="nyx_seed"

    # you may provide contaminants
    # from 1 to 6, -11 to -4
    args.true_SiIII=[[0, 0], [2, -10]]
    args.true_SiII=[[0, 0], [2, -10]]
    # from -5 to 0
    args.true_HCD=[0, -4]
    # from -5 to 2
    args.true_SN=[0, -4]
    args.true_AGN=[0, -5]
    
elif choose_data:    
    true_cosmo=None
    args.data_label = "Chabanier2019"
    # args.data_label="Karacayli2024"
    args.data_label_hires = "Karacayli2022"
    args.z_max = 3.9

# you do not need to provide the archive for obs data 
data = {"P1Ds": None, "extra_P1Ds": None}

if choose_challenge:    
    # folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
    # fname = "mock_challenge_0.2_nonoise_fiducial.fits"
    # fname = "mock_challenge_0.2_nonoise_CGAN_4096_base.fits"
    # fname = "mock_challenge_0.2_nonoise_cosmo_grid_3.fits"
    # fname = "mock_challenge_0.2_nonoise_bar_ic_grid_3.fits"
    # fname = "mock_challenge_0.2_noise-42-0_fiducial.fits"
    # true_sim_label="nyx_central"
    # true_sim_label="nyx_seed"
    # true_sim_label="nyx_3"
    data["P1Ds"] = P1D_DESIY1(
        fname = folder + fname, 
        true_sim_label=true_sim_label
    )
elif choose_desiy1:
    fname = None
    # in NERSC
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"

    if fname is None:
        print("choose appropriate folder")
    else:    
        args.z_min = 2.0
        args.z_max = 5.3
        args.data_label_hires = None
        # args.data_label_hires = "Karacayli2022"
        
        data["P1Ds"] = P1D_DESIY1(
            fname=fname, 
            z_min=args.z_min, 
            z_max=args.z_max
        )
        
        
        # data["extra_P1Ds"] = set_P1D(
        #     args.data_label_hires,
        #     args,
        #     archive=archive,
        #     # true_cosmo=true_cosmo,
        #     # emulator=emulator,
        #     cull_data=False
        # )
else:
    data["P1Ds"] = set_P1D(
        args.data_label,
        args,
        archive=archive,
        true_cosmo=true_cosmo,
        emulator=emulator,
        cull_data=False
    )
    if args.data_label_hires is not None:
        data["extra_P1Ds"] = set_P1D(
            args.data_label_hires,
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
# cosmology
args.ic_correction=False

args.emu_cov_factor = 0.02
# args.fid_cosmo_label="mpg_central"
args.fid_cosmo_label="nyx_central"
# args.fid_cosmo_label="nyx_seed"

# args.fid_cosmo_label="nyx_3"
# args.ic_correction=True
# args.fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# IGM
# args.fid_igm_label="mpg_central"
args.fid_igm_label="nyx_central"
# args.fid_igm_label="nyx_seed"
# args.fid_igm_label="nyx_3"
# args.fid_igm_label="nyx_3_1"
if choose_data == False:
    args.igm_priors = "hc"
else:
    args.type_priors = "data"
args.type_priors = "hc"

# contaminants
# args.fid_SiIII=[0, -10]
# args.fid_SiII=[0, -10]
# args.fid_HCD=[0, -6]
# args.fid_SN=[0, -4]
# args.fid_AGN=[0, -5]


args.fid_SiIII=[[0, 0], [4, -5]]
args.fid_SiII=[[0, 0], [2, -10]]
args.fid_HCD=[0, -2]
args.fid_SN=[0, -4]
args.fid_AGN=[0, -5]

# parameters
args.vary_alphas=False
args.vary_alphas=True
args.fix_cosmo=False
# args.fix_cosmo=True
# args.n_tau=1
# args.n_sigT=1
# args.n_gamma=1
# args.n_kF=1
args.n_tau=2
args.n_sigT=2
args.n_gamma=2
args.n_kF=2
args.n_SiIII = 1
args.n_d_SiIII = 1
args.n_SiII = 0
args.n_dla=1
args.n_sn=0
args.n_agn=0

free_parameters = set_free_like_parameters(args)
free_parameters

# %% [markdown]
# ### Set likelihood

# %%
like = set_like(
    data["P1Ds"],
    emulator,
    fid_cosmo,
    free_parameters,
    args,
    data_hires=data["extra_P1Ds"],
)

# %% [markdown]
# Sampling parameters

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %% [markdown]
# Compare data and fiducial/starting model

# %% [markdown]
# priors at z for which no data! XD

# %%
like.plot_p1d(residuals=False, plot_every_iz=1, print_chi2=False)
like.plot_p1d(residuals=True, plot_every_iz=2, print_ratio=False)

# %%
like.plot_igm()


# %% [markdown]
# ### Set fitter

# %%
def func_for_sampler(p0):
    res = fitter.like.get_log_like(values=p0, return_blob=True)
    return res[0], *res[2]

# for sampler, no real fit, just test
# args.n_steps=1000
# args.n_burn_in=1500
# args.parallel=False
# args.explore=False
# args.n_steps=500
# args.n_burn_in=500
# args.parallel=False
# args.explore=True

args.n_steps=1
args.n_burn_in=0
args.parallel=False
args.explore=True

fitter = Fitter(
    like=like,
    rootdir=output_dir,
    save_chain=False,
    nburnin=args.n_burn_in,
    nsteps=args.n_steps,
    parallel=args.parallel,
    explore=args.explore,
    fix_cosmology=args.fix_cosmo,
)

# %% [markdown]
# ### Run minimizer

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
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=16)

# %% [markdown]
# - GP Minimization improved: 7495.505449898462 1413.4703537937303
# - Nyx_alphap Minimization improved: 4381.154069685914 1455.4486410843758

# %%
if args.fix_cosmo == False:
    fitter.plot_mle_cosmo()

# %%
fitter.plot_p1d(residuals=False, plot_every_iz=1)

# %%
fitter.plot_p1d(residuals=True, plot_every_iz=2)

# %%
fitter.plot_igm(cloud=True)

# %% [markdown]
# ### Run sampler
# It takes less than 2 min on my laptop without any parallelization

# %%
# %%time
run_sampler = True
if run_sampler:    
    _emcee_sam = fitter.run_sampler(pini=fitter.mle_cube, log_func=func_for_sampler)

# %% [markdown]
# Todo
#
# - add nuisance
# - new mle?

# %%
# sampler.write_chain_to_file()

# %%
# import h5py
# nyx_file = "/home/jchaves/Proyectos/projects/lya/data/nyx/models_Nyx_Oct2023.hdf5"
# ff = h5py.File(nyx_file, "r")
# sim_avail = list(ff.keys())

# zkeys = list(ff["cosmo_grid_0"].keys())

# snap = ff["cosmo_grid_0"][zkeys[0]]
# list_scalings = list(snap.keys())

# z = np.zeros((len(list_scalings), len(zkeys)))
# fbar = np.zeros((len(list_scalings), len(zkeys)))

# for ii in range(len(list_scalings)):
#     for jj in range(len(zkeys)):
#         z[ii, jj] = float(zkeys[jj][-3:])
#         snap = ff["cosmo_grid_0"][zkeys[jj]]
#         if list_scalings[ii] in snap:
#             if "T_0" in snap[list_scalings[ii]].attrs.keys():
#                 fbar[ii, jj] = snap[list_scalings[ii]].attrs["T_0"]            
#             else:
#                 print(list_scalings[ii], zkeys[jj]) 


# for ii in range(len(list_scalings)):
#     if "new" in list_scalings[ii]:
#         col = "red"
#     elif "native" in list_scalings[ii]:
#         col = "k"
#     else:
#         col = "C1"
#     _ = np.argwhere(fbar[ii, :] != 0)[:,0]
#     if(len(_) > 0):
#         plt.plot(z[ii, _], fbar[ii, _], col, label=list_scalings[ii], alpha=0.75)
# # plt.legend()
# plt.xlabel("z")
# plt.ylabel("T_0")
# plt.savefig("nyx_T0.pdf")
# # plt.ylabel("gamma")
# # plt.savefig("nyx_gamma.pdf")

# %%

# %%
# check IGM histories
# zs=data["P1Ds"].z

# plt.plot(zs, like.theory.model_igm.F_model.get_tau_eff(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["tau_eff"], "--")

# plt.plot(zs, like.theory.model_igm.T_model.get_sigT_kms(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["sigT_kms"], "--")

# plt.plot(zs, like.theory.model_igm.T_model.get_gamma(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["gamma"], "--")

# plt.plot(zs, like.theory.model_igm.P_model.get_kF_kms(zs))
# plt.plot(like.truth["igm"]["z"], like.truth["igm"]["kF_kms"], "--")

# %% [markdown]
# Results fiducial
#
# -- w/ emu error 
#
# - block-covariance chi2 48
#
# Delta2_star 0.36652 0.36004 0.00647
#
# n_star -2.30407 -2.29877 0.0053
#
# alpha_star -0.21519 -0.21614 0.00096
#
# - full-covariance chi2 50
#
# Delta2_star 0.36911 0.36004 0.00907
#
# n_star -2.30417 -2.29877 0.0054
#
# alpha_star -0.21405 -0.21614 0.00209
#
# -- w/o emu error 
#
# - block-covariance chi2 147
#
# Delta2_star 0.37197 0.36004 0.01193
#
# n_star -2.30516 -2.29877 0.00639
#
# alpha_star -0.21374 -0.21614 0.00241
#
# - full-covariance chi2 181
#
# Delta2_star 0.3809 0.36004 0.02086
#
# n_star -2.30277 -2.29877 0.00401
#
# alpha_star -0.2111 -0.21614 0.00504

# %%
def make_p1d_err_plot(p1ds_err, kMpc_test):
    """
    Plot the P1D errors with 16th and 84th percentiles shaded.
    
    Parameters:
    p1ds_err (np.array): Array of P1D errors
    kMpc_test (np.array): k values in Mpc^-1
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate median, 16th and 84th percentiles
    p1d_median = np.nanmedian(p1ds_err.reshape(-1, len(kMpc_test)), axis=0)
    perc_16 = np.nanpercentile(p1ds_err.reshape(-1, len(kMpc_test)), 16, axis=0)
    perc_84 = np.nanpercentile(p1ds_err.reshape(-1, len(kMpc_test)), 84, axis=0)
    
    # Plot median line
    ax.plot(kMpc_test, p1d_median,  color='crimson')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax.fill_between(kMpc_test, perc_16, perc_84, alpha=0.3, color='crimson')
    
    ax.set_xlabel('k (Mpc$^{-1}$)')
    ax.set_ylabel('Relative Error in P1D')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'p1d_errors_all.pdf', bbox_inches='tight')
    plt.close()

# %%

# %%

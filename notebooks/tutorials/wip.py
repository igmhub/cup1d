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

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
args.set_baseline(fit_type="global", fix_cosmo=True, P1D_type="DESIY1_QMLE3")

# %%
pip = Pipeline(args)

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
# ### Set likelihood

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

# %%
# param_attime_all

# %%

# %% [markdown]
# # HERE

# %%

args.set_baseline(fit_type="global", fix_cosmo=True, zmax=4.2)
# args.set_baseline(fit_type="andreu", fix_cosmo=True, zmax=4.2)
# args.set_baseline(fit_type="andreu2", fix_cosmo=True, zmax=4.2)
# args.set_baseline(fit_type="global", fix_cosmo=False, zmax=4.2)
# args.set_baseline(fit_type="andreu2", fix_cosmo=False, zmax=4.2)
like = set_like(
    data["P1Ds"],
    emulator,
    args,
    data_hires=data["extra_P1Ds"],
)
len(like.free_param_names)

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
from cup1d.optimize.set_ic import set_ic_from_fullfit, set_ic_from_z_at_time

# %% [markdown]
# #### IC from 1z at a time fit

# %%
fname = "first_1z_snr3_nocosmo/res.npy"
fitter = set_ic_from_z_at_time(args, like, data, emulator, fname, verbose=True)

# %%
# for key in igm_hist_mpg.keys():
#     plt.plot(igm_hist_mpg[key]["z"], igm_hist_mpg[key]["tau_eff"], "k", alpha=0.2)
    
# plt.errorbar(igm["z"], igm["tau_eff"], igm["err_tau_eff"])
# z = fitter.like.data.z
# best_tau = fitter.like.theory.model_igm.models["F_model"].get_tau_eff(z, like_params=like_params)
# plt.plot(z, best_tau)
# plt.xlabel("z")
# plt.ylabel("tau_eff")


# %%

# %%

# %% [markdown]
# #### IC from full fit

# %%
fname = "allz_snr3_nocosmo_global/res.npy"
fitter = set_ic_from_fullfit(like, data, emulator, fname, type_fit="global", verbose=True)


# %%
# for p in like.free_params:
#     print(p.name, '\t', np.round(p.value, 3), '\t', np.round(p.min_value, 3), '\t', np.round(p.max_value, 3), '\t', p.Gauss_priors_width, p.fixed)

# %%
# for ii in range(len(free_params)):
#     if free_params[ii].fixed:
#         for jj, p in enumerate(like.free_params):
#             if p.name == free_params[ii].name:
#                 like.free_params.pop(jj)
#                 like.free_param_names.pop(jj)
#                 break
# for p in like.free_params:
#     print(p.name)

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

# %% [markdown]
# Compare data and fiducial/starting model

# %%
# # %%time
# like.plot_p1d(plot_panels=True, residuals=True)
input_pars = fitter.like.sampling_point_from_parameters().copy()
# input_pars = fitter.mle_cube[2:]
# input_pars = fitter.mle_cube.copy()
# , plot_fname="test_weak"
fitter.like.plot_p1d(plot_panels=True, residuals=True, values=input_pars)
# like.plot_p1d(plot_panels=True, residuals=True)
# like.plot_p1d(residuals=True)


# %%

# %%

# %%
# like.plot_hull_fid(like_params=like.free_params)

# %% [markdown]
# 705 chi2 614 deg 67 params prob 0.63%

# %% [markdown]
# ### Set fitter

# %%
# xx = np.concatenate([sig_diff, [0]])
# yy = np.concatenate([chi2_arr, [chi2_cen]])
# ind = np.argsort(xx)
# xx = xx[ind]
# yy = yy[ind]

# plt.plot(xx, yy - chi2_cen, "o:")
# # rfit = np.polyfit(xx, yy, 2)
# # x = np.linspace(-2, 2, 100)
# # plt.plot(x, np.poly1d(rfit)(x) - chi2_cen)
# plt.axhline(scipy_chi2.ppf(0.68, 2))
# plt.axhline(scipy_chi2.ppf(0.95, 2))
# plt.axvline(1)
# plt.axvline(2)
# plt.axvline(-1)
# plt.axvline(-2)
# # plt.plot(sig_diff, scipy_chi2(sig_diff))
# plt.xlabel(r"$\sigma$ around best")
# plt.ylabel(r"$\Delta\chi^2$")

# %% [markdown]
# ### Run minimizer

# %% [markdown]
# 5 min to run baseline with nsamples=1

# %%
p0 = fitter.mle_cube.copy()

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
fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, restart=True, nsamples=0)
# zmask = np.array([2.4])
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0, zmask=zmask)
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %%
# %%time
err = fitter.like.get_error(fitter.mle_cube.copy())
err

# %%
err

# %%
pstar = fitter.like.theory.fid_cosmo["cosmo"].get_linP_params()
print(pstar['Delta2_star'], pstar['n_star'])

target_params = {
    'Delta2_star': 0.42,
     'n_star': -2.33,
}

fitter.like.theory.rescale_fid_cosmo(target_params)

pstar = fitter.like.theory.fid_cosmo["cosmo"].get_linP_params()
print(pstar['Delta2_star'], pstar['n_star'])


# %%

# %%
def prof_like(grid, p0):
    fitter.run_minimizer(fitter.like.minus_log_prob, p0=p0, restart=True, nsamples=0)


# %%
fitter.mle_cosmo

# %%
fitter.like.minus_log_prob(fitter.mle_cube)

# %%

# %%
np.sqrt(0.021**2/0.04**2 + 0.009**2/0.018**2)

# %%
np.sqrt(1)

# %%
from scipy.differentiate import hessian

# %%
hess = derivative(fitter.like.minus_log_prob, fitter.mle_cube)


# %%

# %%

# %%

# %%
def get_points_profile_like(


# %%
Andreu2 809

'Delta2_star': 0.4315486076928048,  'n_star': -2.303017944612121,

Global 719

'Delta2_star': 0.4110166684732482,  'n_star': -2.311688458372555,

# %%
diff_cosmo(fitter.mle)


# %%
def diff_cosmo(mle):
    andreu2 = {'Delta2_star': 0.4315486076928048,  'n_star': -2.303017944612121}
    print(np.round(mle["Delta2_star"] - andreu2["Delta2_star"], 3))
    print(np.round(mle["n_star"] - andreu2["n_star"], 3))


# %% [markdown]
# 758 wip0
#
# 777 Andreu
#
# 763 wip1 (wip2) 0.5%!
# to 759 when tau for each redshift, 0.046%, not better
#
# 741 wip2 0.1%
#
# 716 global 0.69%
#
# 719 global + cosmo 0.69%
#
# 815 andreu2 0.02%
#
# 809 andreu2 + cosmo, 0.04%

# %%
mask = np.arange(11)

like_params = fitter.like.parameters_from_sampling_point(fitter.mle_cube)

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/allz_snr3_cosmo_global/"
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
dir_out = {
    "mle_cube":fitter.mle_cube,
    "mle":fitter.mle,
    "chi2":fitter.mle_chi2,
    "param_attime_all":param_attime_all,
}
np.save("allz_snr3_nocosmo_global/res.npy", dir_out)
# np.save("allz_snr3_nocosmo_andreu2/res.npy", dir_out)

# %%

# %%
# new_vals = {}
# like_params = fitter.like.parameters_from_sampling_point(fitter.mle_cube)

# fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/full_"
# folder = fold0 + "ev_baseline1z_params/taueff"
# # folder = None
# oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
# folder = fold0 + "ev_baseline1z_params/sigT"
# oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
# folder = fold0 + "ev_baseline1z_params/Simult"
# oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
# folder = fold0 + "ev_baseline1z_params/Siadd"
# oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
# folder = fold0 + "ev_baseline1z_params/HCD"
# oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

# # mod = ocTmodel

# # for key in mod:
# #     if key in args.opt_props:
# #         new_vals[key] = mod[key]
# #         print(mod[key])

# models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
# for mod in models:
#     for key in mod:
#         if key in args.opt_props:
#             new_vals[key] = mod[key]
# np.save("opt_vals.npy", new_vals)

# %%
# new_vals

# %%

# %% [markdown]
# #### Latest

# %%

# diru = "test_snr3_11_1_1_1"
# diru = "test_snr3_3_3_1_1"
# diru = "test_snr3_3_2_2_2"
# diru = "test_snr3_3_2_1_1_cosmo"
# diru = "allz_snr3_nocosmo_full"
# diru = "qmle3_all_igmonly"
# diru = "allz_snr3_nocosmo_andreu"
# diru = "wip0"
# diru = "allz_snr3_nocosmo_global"
diru = "allz_snr3_cosmo_global"
# diru = "allz_snr3_cosmo_andreu2"
plotter = Plotter(fitter, save_directory=diru)
plotter.plot_p1d(plot_panels=True, residuals=True)
# plotter.plots_minimizer()

# %%
for zz in like.data.z:
    plotter.plot_illustrate_contaminants_cum(fitter.mle_cube.copy(), np.array([zz]))

# %%
plotter.plot_mle_cosmo()

# %%
# plotter.plot_p1d(plot_panels=True, residuals=True)

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
args.set_baseline(fit_type="at_a_time_orig")

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

out_mle = []
out_mle_cube = []
out_chi2 = []
# for ii in range(len(like.data.z)): 
# for ii in range(1): 
for ii in range(7, 8): 
# for ii in range(2,3): 
# for ii in range(9, 10): 
# for ii in range(2, 3): 
    zmask = np.array([like.data.z[ii]])
    
    print()
    print(ii, like.data.z[ii])
    p0 = np.array(list(like.fid["fit_cube"].values()))
    # p0 = np.array([0.5, 0.8])
    # fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0_arr[ii], zmask=zmask, restart=True)
    fitter.run_minimizer(log_func_minimize=fitter.like.minus_log_prob, p0=p0, zmask=zmask, restart=True, nsamples=4)
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
# diru = "orig_1z_snr3_nocosmo"
diru = None
plotter = Plotter(fitter, save_directory=diru, zmask=zmask)
plotter.plot_p1d(values=out_mle_cube, plot_panels=True, residuals=True, z_at_time=True)

# %%

plotter.plot_illustrate_contaminants_cum(out_mle_cube[0].copy(), np.array([3.6]))

# %%

# plotter.plot_p1d(residuals=True, zmask=zmask)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.2], lines_use=lines_use)
# plotter.plot_illustrate_contaminants(out_mle_cube[0].copy(), [2.4], lines_use=lines_use)
# plotter.plot_illustrate_contaminants_each(out_mle_cube[0].copy(), np.array([2.2]))
# plotter.plot_illustrate_contaminants(test, [2.4], lines_use=lines_use)

# %%
ii = 0

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

# %% [markdown]
# #### Different baseline as a function of z

# %%

out_mle = []
out_mle_cube = []
out_chi2 = []
for ii in range(len(data["P1Ds"].z)): 
# for ii in range(1): 
# for ii in range(10, 11): 
# for ii in range(1): 
# for ii in range(2,3): 
# for ii in range(7, 8): 
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
print_results(like, out_chi2, out_mle_cube)

# %%
print_results(like, out_chi2, out_mle_cube)

# %%
from cup1d.optimize.show_results import reformat_cube

# %%
out_mle_cube_reformat = reformat_cube(args, data, emulator, out_mle_cube)

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
# diru = None

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
for iz in range(len(like.data.z)):
    args.set_baseline(ztar=data["P1Ds"].z[iz], fit_type="at_a_time")
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
    fitter.mle_cube = out_mle_cube[iz]
    
    plotter = Plotter(fitter, save_directory=diru, zmask=[like.data.z[iz]])
    plotter.plot_illustrate_contaminants_cum(out_mle_cube[iz].copy(), np.array([like.data.z[iz]]))

# %%
dir_out = {
    "mle_cube":out_mle_cube,
    "mle":out_mle,
    "chi2":out_chi2,
    "mle_cube_reformat":out_mle_cube_reformat,
}
np.save("first_1z_snr3_nocosmo/res.npy", dir_out)

# %%
dir_out = np.load("first_1z_snr3_nocosmo/res.npy", allow_pickle=True).item()
out_mle_cube = dir_out["mle_cube"]
out_mle_cube_reformat = dir_out["mle_cube_reformat"]

# %%

# %%

# %%

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
fitter.like.theory.model_igm.models["F_model"].list_coeffs

# %%
np.save("first_1z_snr3_nocosmo/fit_baseline_param_attime.npy", param_attime_all)

# %%

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
np.log10(10**-1.5 / 5)

# %%
np.save("first_1z_snr3_nocosmo/weak_priors.npy", weak1_priors)

# %% [markdown]
# #### Redo 1 z at time fits using weak_priors

# %%
weak_priors = weak1_priors.copy()

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
weak2_priors = plot_z_at_time_params(fitter, out_mle)

# %%
weak_priors = weak2_priors.copy()

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
from cup1d.optimize.show_results import print_results

# %%
print_results(like, out_chi2, out_mle_cube)

# %%

# %%

# %%
weak3_priors = plot_z_at_time_params(fitter, out_mle)

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

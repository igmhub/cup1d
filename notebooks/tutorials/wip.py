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
from cup1d.likelihood.pipeline_z import Pipeline_z

# %%
# args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="Nyx_alphap_cov", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args.vary_alphas = False
args.data_label = "DESIY1"

# version = "6"
# folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
# fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
# args.p1d_fname = folder + fname

# in NERSC
# QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
# FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
# args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"
args.z_min = 2.1
args.z_max = 4.3
args.cov_only_diag = True


args.fid_SiII=[[0, 0], [-4, -10]] # null
args.fid_SN=[0, -4] # null
args.fid_AGN=[0, -4] # null
# args.fid_AGN=[0, -2]
# args.fid_SiIII=[[0, 0], [-3, -10]] # null
# args.fid_SiIII=[[0, 0], [-3, -5]] # 1 null
args.fid_SiIII=[[0, 0], [5, -5]]
# args.fid_HCD=[0, -4] # null
args.fid_HCD=[0, -2] # 1 null
# args.fid_HCD=[3, -1.5]

args.n_tau = 1
args.n_sigT = 1
args.n_gamma = 1
args.n_kF = 1
args.n_SiIII=1
args.n_d_SiIII=1
args.n_dla=1

args.fix_cosmo = True
args.fid_cosmo_label = "Planck18"
# args.fid_igm_label_mF = "mpg_central"
# args.fid_igm_label_T = "mpg_central"
# args.fid_igm_label_kF = "mpg_central"
args.fid_igm_label_mF = "nyx_central"
args.fid_igm_label_T = "nyx_central"
args.fid_igm_label_kF = "nyx_central"
args.ic_correction = False

# %%
pip = Pipeline(args)

# %%

# %%
key_avoid = [
    '$\\Delta^2_\\star$', 
    '$n_\\star$', 
    '$\\alpha_\\star$', 
    '$f_\\star$', 
    '$g_\\star$',
    '$H_0$'
]

out_folder_base = "desi_fft_z"
# list_z = pip.fitter.like.data.z
list_z = np.array([2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2])
print("list_z = {}".format(list_z))

# only minimizer for now, need to implement sampler
for ii, z in enumerate(list_z):
    print("Reading z = {}".format(z))
    fname = os.path.join(out_folder_base, "z{}".format(z), "chain_1", "minimizer_results.npy") 
    data = np.load(fname, allow_pickle=True).item()

    # create results
    if ii == 0:
        results = {}
        for key in data["mle"]:
            if key in key_avoid:
                continue
            results[key] = np.zeros((len(list_z)))
        results['lnprob_mle'] = np.zeros((len(list_z)))

    for key in data["mle"]:
        if key in key_avoid:
            continue
        results[key][ii] = data["mle"][key]
    results['lnprob_mle'][ii] = data['lnprob_mle']

# %%
results.keys()

# %% [markdown]
# also save the evaluation of the IGM and contaminants for the redshifs of interest

# %%
for jj, key in enumerate(results):
    if key == 'lnprob_mle':
        continue
    fig, ax = plt.subplots()
    ax.plot(list_z, results[key], "o:", label="Fit")
    for ii in range(3):
        x = list_z[:-1].copy()
        y = results[key][:-1].copy()
        if key == '$\\mathrm{ln}\\,f^{SiIII}_0$':
            _ = y > -5.5
            x = x[_]
            y = y[_]
            ax.set_ylim(-5.5)
        elif key == '$\\mathrm{ln}\\,d^{SiIII}_0$':
            _ = y > 4
            x = x[_]
            y = y[_]
            ax.set_ylim(4)
        z = np.polyfit(x, y, ii)        
        p = np.poly1d(z)
        ax.plot(list_z, p(list_z), "C"+str(ii+1), label="polyfit ndeg="+str(ii), alpha=0.5)
    ax.legend(loc="upper right")
        
    ax.set_xlabel("z")
    ax.set_ylabel(key)
    plt.savefig("desi_fft_z/"+str(jj)+".png")

# %%
data.keys()

# %%

args.z_min = 2.1
args.z_max = 2.3
# args.z_min = 2.1
# args.z_max = 4.3
pip = Pipeline_z(args, out_folder="desi_fft_z")

# %%
# fname_chain = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/v6/Nyx_alphap_cov/mockchallenge-0.6_nonoise_fiducial/chain_2/sampler_results.npy"
# plotter = Plotter(save_directory="test", fname_chain=fname_chain)
# plotter.plot_corner(only_cosmo=True)

# %% [markdown]
# ### Set archive

# %%
# args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Jul2024")
args = Args(emulator_label="Nyx_alphap_cov", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")

# %%
# path nyx files in NERSC /global/cfs/cdirs/desi/science/lya/y1-p1d/likelihood_files/nyx_files/
archive = set_archive(args.training_set)

# %% [markdown]
# ### Set emulator

# %%
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
# #### Set either mock data or real data

# %%
choose_forecast = False
choose_mock = False
choose_data = False
choose_challenge = True
choose_desiy1 = False

if choose_forecast:
    args.data_label_hires = None
    # for forecast, just start label of observational data with mock
    # args.data_label = "mock_Chabanier2019"
    # args.data_label="mock_Karacayli2024"
    # args.data_label_hires = "mock_Karacayli2022"
    args.data_label="mock_DESIY1"
    args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms.fits"

    # you need to provide true cosmology, IGM history, and contaminants
    true_cosmo = set_cosmo(cosmo_label="nyx_central")
    args.true_igm_label="nyx_central"
    # true_cosmo = set_cosmo(cosmo_label="mpg_22")
    # true_cosmo = set_cosmo(cosmo_label="Planck18")
    # args.true_igm_label="nyx_central"
    # args.true_igm_label="mpg_22"
    # true_cosmo = set_cosmo(cosmo_label="mpg_central")
    # args.true_igm_label="mpg_central"
    # from -11 to -4
    args.true_SiIII=[[0, 0], [-10, -10]]
    args.true_SiII=[[0, 0], [-10, -10]]
    # from -7 to 0
    args.true_HCD=[0, -6]
    # from -5 to 2
    args.true_SN=[0, -4]
    # from -5 to 1.5
    args.true_AGN=[0, -5]
    args.z_min = 2.1
    args.z_max = 4.3
elif choose_mock:    
    true_cosmo=None
    # to analyze data from simulations
    args.data_label = "mpg_central"    
    args.true_cosmo_label="mpg_central"
    args.true_igm_label="mpg_central"
    # args.data_label="nyx_central"
    # args.data_label="nyx_seed"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None

    # provide cosmology only to cull the data
    # args.true_cosmo_label="mpg_central"
    # args.true_cosmo_label="nyx_central"
    true_cosmo = set_cosmo(cosmo_label=args.data_label)
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
elif choose_challenge:
    args.data_label = "challenge_DESIY1"
    version = "6"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    args.p1d_fname = folder + fname
    if "fiducial" in args.p1d_fname:
        true_sim_label = "nyx_central"
    elif "CGAN" in args.p1d_fname:
        true_sim_label = "nyx_seed"
    elif "grid_3" in args.p1d_fname:
        true_sim_label = "nyx_3"
    else:
        true_sim_label = None
    true_cosmo = set_cosmo(cosmo_label=true_sim_label)
    args.true_igm_label=true_sim_label
    args.z_min = 2.1
    args.z_max = 4.3
elif choose_desiy1:
    true_cosmo = None
    args.true_igm_label= None
    args.data_label = "DESIY1"
    # in NERSC
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v2.fits"
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
# cosmology
args.ic_correction=False

args.emu_cov_factor = 0.0
# args.fid_cosmo_label="mpg_2"
if "Nyx" in emulator.emulator_label:
    args.fid_cosmo_label="nyx_central"
    args.fid_igm_label="nyx_central"
    args.vary_alphas=True
else:
    args.fid_cosmo_label="mpg_central"
    args.fid_igm_label="mpg_central"
    args.vary_alphas=False

args.vary_alphas=False
args.fid_cosmo_label="nyx_central"
args.fid_igm_label_mF="nyx_central"
args.fid_igm_label_T="nyx_central"
args.fid_igm_label_kF="nyx_central"
# args.fid_igm_label_kF="mpg_central"

# args.fid_cosmo_label="nyx_seed"

# args.fid_igm_label="mpg_22"
# args.fid_cosmo_label="mpg_22"

# args.fid_igm_label="mpg_central"
# args.ic_correction=True
# args.fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# args.use_star_priors = None
args.use_star_priors = {}
# Planck18 0.354 -2.300 -0.2155
# 5 sigma 0.056 0.011 0.0028
blob = CAMB_model.CAMBModel(zs=[3], cosmo=fid_cosmo).get_linP_params()
amin = blob["alpha_star"] - 0.0028
amax = blob["alpha_star"] + 0.0028
args.use_star_priors["alpha_star"] = [amin, amax]


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

# contaminants
# from 1 to 6, -11 to -4
args.fid_SiIII=[[0, 0], [-4, -10]]
args.fid_SiII=[[0, 0], [-4, -10]]
# from -5 to 0
args.fid_HCD=[0, -4]
# from -5 to 2
args.fid_SN=[0, -4]
args.fid_AGN=[0, -5]

    
args.fix_cosmo=False
# args.fix_cosmo=True
args.n_tau=2
args.n_sigT=0
args.n_gamma=0
args.n_kF=0
# args.n_tau=2
# args.n_sigT=2
# args.n_gamma=2
# args.n_kF=2
args.n_SiIII = 0
args.n_d_SiIII = 0
args.n_SiII = 0
args.n_dla=0
args.n_sn=0
args.n_agn=0

# args.fid_SiII=[[0, 0], [-4, -10]] # null
# args.fid_SN=[0, -4] # null
# args.fid_AGN=[0, -4] # null
# # args.fid_SiIII=[[0, 0], [-3, -10]] # null
# # args.fid_SiIII=[[0, 0], [-3, -5]] # 1 null
# args.fid_SiIII=[[0, 0], [5, -5]]
# # args.fid_HCD=[0, -4] # null
# # args.fid_HCD=[0, -2] # 1 null
# args.fid_HCD=[3, -1.5]

# args.n_SiII=0
# args.n_d_SiII=0
# args.n_sn=0
# args.n_agn=0
# args.n_tau=2
# args.n_sigT=2
# args.n_gamma=2
# args.n_kF=2
# args.n_SiIII=1
# args.n_d_SiIII=1
# args.n_dla=2

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

# %% [markdown]
# Sampling parameters

# %%
for p in like.free_params:
    print(p.name, p.value, p.min_value, p.max_value)

# %% [markdown]
# Compare data and fiducial/starting model

# %%
like.plot_p1d(residuals=False)
like.plot_p1d(residuals=True)

# %%
# z = like.data.z
# k_kms = like.data.k_kms
# like.theory.model_cont.agn_model.plot_contamination(z, k_kms)

# %%
like.plot_igm(cloud=True)

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
# fitter.like.data.truth

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
# fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, nsamples=4)

# %%

# %%
plotter = Plotter(fitter)
if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()

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

# %%

# %% [markdown]
# inside in terms of deltap and np parameters, but outside in terms of deltastar and nstar

# %%
like.theory.emulator.list_sim_cube[0][:3]

# %%
ii = 0
jj = 0
zs = fitter.like.data.z
pini = fitter.chain[ii,jj,:].copy()
pini[1] = 0.95
like_params = fitter.like.parameters_from_sampling_point(pini)

# %%
pini

# %%
fitter.blobs[ii,jj]

# %%
for par in like_params:
    print(par.name, par.value)

# %%

hull = fitter.like.theory.hull

# %%

# %%

# %%

# %% [markdown]
# why outside?

# %%
# fil = np.load("/home/jchaves/Proyectos/projects/lya/data/mock_challenge/v2/Nyx_alphap_cov/mock_challenge_0.2_nonoise_fiducial/chain_6/sampler_results.npy", allow_pickle=True).item()
# fil.keys()

# %%
# fil["posterior"].keys()
# fil["posterior"]["nrun"].max()
# plt.hist(fil["posterior"]["nrun"].reshape(-1), bins=100);

# %%
# plt.scatter(fil["posterior"]["alpha_star"].reshape(-1), fil["posterior"]["lnprob"].reshape(-1), s=1)

# %%
plotter = Plotter(fitter, save_directory=fitter.save_directory)

# %%
plotter.plots_sampler()

# %%
# chain = plotter.plot_corner(only_cosmo=True)

# %%
fitter.write_chain_to_file()

# %%
fil = np.load("/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/chain_2/chain.npy", allow_pickle=True).item()

# %% [markdown]
# ## Test pipeline

# %%
# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")


# %%
archive = set_archive(args.training_set)

# %%

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

# %%
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
args.archive=archive
args.emulator=emulator

folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0.2/"
fname = "mock_challenge_0.2_nonoise_fiducial.fits"
# fname = "mock_challenge_0.2_nonoise_CGAN_4096_base.fits"
# fname = "mock_challenge_0.2_nonoise_cosmo_grid_3.fits"
# fname = "mock_challenge_0.2_nonoise_bar_ic_grid_3.fits"
# fname = "mock_challenge_0.2_noise-42-0_fiducial.fits"
true_sim_label="nyx_central"
args.data_label = "DESI_Y1"
args.p1d_fname = folder + fname


# cosmology
args.ic_correction=False

args.emu_cov_factor = 0.0
# args.fid_cosmo_label="mpg_central"
args.fid_cosmo_label="nyx_central"
# args.fid_cosmo_label="nyx_seed"

# args.fid_cosmo_label="nyx_3"
# args.ic_correction=True
# args.fid_cosmo_label="Planck18"
# fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# IGM
args.fid_igm_label="mpg_central"
# args.fid_igm_label="nyx_central"
# args.fid_igm_label="nyx_seed"
# args.fid_igm_label="nyx_3"
# args.fid_igm_label="nyx_3_1"

# parameters
args.vary_alphas=False
# args.vary_alphas=True
# args.fix_cosmo=False
args.fix_cosmo=True

args.n_tau=2
args.n_sigT=2
args.n_gamma=2
args.n_kF=2

args.n_steps=100
args.n_burn_in=40
args.parallel=False
args.explore=True

# %%
out_folder = "/home/jchaves/Proyectos/projects/lya/data/tests"
pip = Pipeline(args, make_plots=True, out_folder=out_folder)

# %%
pip.run_minimizer()

# %%
pip.run_sampler()

# %%
plotter = Plotter(pip.fitter, save_directory=pip.out_folder)
plotter.plots_sampler()

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

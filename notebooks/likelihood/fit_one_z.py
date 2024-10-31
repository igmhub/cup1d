# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: cup1d
#     language: python
#     name: cup1d
# ---

# %% [markdown]
# # Fit DESI data, one redshift at a time

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
import cup1d
import os

# %% [markdown]
# ### Set emulator

# %%
# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
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

# %%
# P1D data at NERSC
# QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
# FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
desi_y1_fname = "/global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"

# set output directory for this test
add_hr=True
if add_hr:
    outdir='p1d_desi_hr'
else:
    outdir='p1d_desi'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# ignore IC correction for now
args.ic_correction=False
# option to include emulator covariance
args.emu_cov_factor = 0.0001

# set fiducial cosmology to Planck
args.fid_cosmo_label="Planck18"
fid_cosmo = set_cosmo(cosmo_label=args.fid_cosmo_label)

# set fiducial IGM to Nyx central (not relevant if fitting one redshift at a time)
args.fid_igm_label="nyx_central"
# not sure what this does
args.igm_priors = "hc"
args.type_priors = "hc"

# settings for contaminants (not sure exactly what this does)
args.fid_SiIII=[[0, 0], [7, -4]]
args.fid_SiII=[[0, 0], [0, -10]]
args.fid_HCD=[0, -2]
args.fid_SN=[0, -4]
args.fid_AGN=[0, -5]

# define free parameters (fix cosmo, one parameter per model)
args.vary_alphas=False
args.fix_cosmo=True
args.n_tau=1
args.n_sigT=1
args.n_gamma=1
args.n_kF=1
args.n_SiIII = 1
args.n_d_SiIII = 1
args.n_SiII = 0
args.n_dla=1
args.n_sn=0
args.n_agn=0

free_parameters = set_free_like_parameters(args)
free_parameters


# %%
def fit_one_z(zmin,zmax,show_all_plots=True):
    
    # setup data to fit (one z only)
    args.z_min = zmin
    args.z_max = zmax
    data = {"P1Ds": None, "extra_P1Ds": None}
    data["P1Ds"] = P1D_DESIY1(
            fname=desi_y1_fname, 
            z_min=args.z_min, 
            z_max=args.z_max
        )
    # add high-res P1D
    if add_hr:
        args.data_label_hires = "Karacayli2022"
        data["extra_P1Ds"] = set_P1D(
                args.data_label_hires,
                args,
                archive=archive,
                cull_data=False
            )
    
    if show_all_plots:
        data["P1Ds"].plot_p1d()
        if args.data_label_hires is not None:
            data["extra_P1Ds"].plot_p1d()
    
    # set likelihood 
    like = set_like(
            data["P1Ds"],
            emulator,
            fid_cosmo,
            free_parameters,
            args,
            data_hires=data["extra_P1Ds"],
        )
    
    for p in like.free_params:
        print(p.name, p.value, p.min_value, p.max_value)
      
    if show_all_plots:
        like.plot_p1d(residuals=False, plot_every_iz=1, print_chi2=False)
    
    # set fitter (only the minimizer will be used)
    fitter = Fitter(
            like=like,
            rootdir=outdir,
            save_chain=False,
            nburnin=0,
            nsteps=1,
            parallel=False,
            explore=True,
            fix_cosmology=args.fix_cosmo,
        )
    
    # run the minimizer
    p0 = np.array(list(like.fid["fit_cube"].values()))
    fitter.run_minimizer(log_func_minimize=fitter.like.get_chi2, p0=p0)
    
    return fitter


# %%
fitter=fit_one_z(2.1,2.3,show_all_plots=False)


# %%
def collect_results(fitter):

    like=fitter.like
    # redshift used
    z=like.data.z[0]
    # best-fit values (in unit cube)
    best_fit_cube=fitter.get_best_fit(stat_best_fit="mle")
    results={'z':z,'best_fit_cube':best_fit_cube}

    # likelihood parameters of best-fit
    like_params=like.parameters_from_sampling_point(best_fit_cube)
    results['mF'] = like.theory.model_igm.F_model.get_mean_flux(z=z,like_params=like_params)
    results['T0'] = like.theory.model_igm.T_model.get_T0(z=z,like_params=like_params)
    results['sigT_kms'] = like.theory.model_igm.T_model.get_sigT_kms(z=z,like_params=like_params)
    results['gamma'] = like.theory.model_igm.T_model.get_gamma(z=z,like_params=like_params)
    results['kF'] = like.theory.model_igm.P_model.get_kF_kms(z=z,like_params=like_params)
    results['f_SiIII'] = like.theory.model_cont.metal_models[0].get_amplitude(z=z,like_params=like_params)
    results['damp_SiIII'] = like.theory.model_cont.metal_models[0].get_damping(z=z,like_params=like_params)
    results['amp_HCD'] = like.theory.model_cont.hcd_model.get_A_damp(z=z,like_params=like_params)
    
    return results


# %%
z_edges=np.arange(2.1,4.7,0.2)
print(z_edges)
runs={}
for zmin, zmax in zip(z_edges[:-1],z_edges[1:]):
    fitter=fit_one_z(zmin,zmax,show_all_plots=False)
    z=fitter.like.data.z[0]
    runs[z]={'fitter':fitter}

# %%
for run in runs.values():
    fitter=run['fitter']
    run['results']=collect_results(fitter)

# %%
for z, run in runs.items():
    values=run['results']['best_fit_cube']
    for residuals, tag in zip([True,False],['_res','']):
        plot_fname=outdir+'/p1d{}_{}.png'.format(tag,z)
        run['fitter'].like.plot_p1d(values=values,residuals=residuals,plot_fname=plot_fname)

# %%
all_zs=[run['results']['z'] for run in runs.values()]
all_mFs=[run['results']['mF'] for run in runs.values()]
#print(all_mFs)
plt.plot(all_zs,all_mFs)
plt.xlabel('z')
plt.ylabel('<F>(z)')
plt.savefig(outdir+'/best_fit_mF.png')

# %%
all_T0s=[run['results']['T0'] for run in runs.values()]
plt.plot(all_zs,all_T0s)
plt.xlabel('z')
plt.ylabel('T0 [K]')
plt.savefig(outdir+'/best_fit_T0.png')

# %%
all_gammas=[run['results']['gamma'] for run in runs.values()]
plt.plot(all_zs,all_gammas)
plt.xlabel('z')
plt.ylabel('gamma')
plt.savefig(outdir+'/best_fit_gamma.png')

# %%
all_kFs=[run['results']['kF'] for run in runs.values()]
plt.plot(all_zs,all_kFs)
plt.xlabel('z')
plt.ylabel('kF [s/km]')
plt.savefig(outdir+'/best_fit_kF.png')

# %%
all_f_SiIIIs=[run['results']['f_SiIII'] for run in runs.values()]
plt.plot(all_zs,all_f_SiIIIs)
plt.xlabel('z')
plt.ylabel('f_SiIII')
plt.savefig(outdir+'/best_fit_f_SiIII.png')

# %%
all_damp_SiIIIs=[run['results']['damp_SiIII'] for run in runs.values()]
plt.plot(all_zs,all_damp_SiIIIs)
plt.xlabel('z')
plt.ylabel('damp_SiIII')
plt.savefig(outdir+'/best_fit_damp_SiIII.png')

# %%
all_amp_HCDs=[run['results']['amp_HCD'] for run in runs.values()]
plt.plot(all_zs,all_amp_HCDs)
plt.xlabel('z')
plt.ylabel('HCD_amp')
plt.savefig(outdir+'/best_fit_HCD_amp.png')

# %%

# %%

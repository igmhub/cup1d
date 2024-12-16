# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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

from cup1d.likelihood.pipeline_z import Pipeline_z

# %%

# %%
# args = Args(emulator_label="Nyx_alphap", training_set="Nyx23_Jul2024")
args = Args(emulator_label="Nyx_alphap_cov", training_set="Nyx23_Jul2024")
# args = Args(emulator_label="Cabayol23+", training_set="Cabayol23")
# args = Args(emulator_label="Pedersen23_ext", training_set="Cabayol23")
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

# args.fid_SiII=[[0, 0], [-4, -10]] # null
# args.fid_SiII=[[0, 0], [0, -6]] # null
# args.n_SiII=0
# args.n_d_SiII=0
# args.fid_SN=[0, -4] # null
# args.fid_AGN=[0, -4] # null
# args.fid_AGN=[0, -2]
# args.fid_SiIII=[[0, 0], [-3, -10]] # null
# args.fid_SiIII=[[0, 0], [-3, -5]] # 1 null
args.fid_SiIII_X=[0, -5]
args.fid_SiIII_D=[0, 5]
args.fid_SiIII_A=[0, 1.5]
# args.fid_HCD=[0, -4] # null
args.fid_HCD=[0, -2] # 1 null
# args.fid_HCD=[3, -1.5]

args.n_tau = 1
args.n_sigT = 1
args.n_gamma = 1
args.n_kF = 1
args.n_x_SiIII=1
args.n_d_SiIII=1
args.n_a_SiIII=1
args.n_dla=1


args.fix_cosmo = True
args.fid_cosmo_label = "Planck18"
# args.fid_igm_label_mF = "mpg_central"
# args.fid_igm_label_T = "mpg_central"
# args.fid_igm_label_kF = "mpg_central"
args.fid_sim_igm_label_mF = "nyx_central"
args.fid_sim_igm_label_T = "nyx_central"
args.fid_sim_igm_label_kF = "nyx_central"
args.ic_correction = False

# %%

# %%

# args.z_min = 2.1
# args.z_max = 2.9
args.z_min = 3.7
args.z_max = 3.9
pip = Pipeline_z(args, out_folder="desi_fft_z")

# %%

# %%

# %%
# pip = Pipeline(args)

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
    fname = os.path.join(out_folder_base, "z{}".format(z), "chain_1", "fitter_results.npy") 
    data = np.load(fname, allow_pickle=True).item()

    # create results
    if ii == 0:
        results = {}
        for key in data["fitter"]["mle"]:
            if key in key_avoid:
                continue
            results[key] = np.zeros((len(list_z)))

        for key in data["IGM"]:
            if key in key_avoid:
                continue
            results[key] = np.zeros((len(list_z)))
        
        for key in data["nuisance"]["SiIII"]:
            if key in key_avoid:
                continue
            results["SiIII_" + key] = np.zeros((len(list_z)))
            
        results['lnprob_mle'] = np.zeros((len(list_z)))
        results['HCD'] = np.zeros((len(list_z)))

    for key in data["fitter"]["mle"]:
        if key in key_avoid:
            continue
        results[key][ii] = data["fitter"]["mle"][key]

    for key in data["IGM"]:
        if key in key_avoid:
            continue
        results[key][ii] = data["IGM"][key][0]
    
    for key in data["nuisance"]["SiIII"]:
        if key in key_avoid:
            continue
        results["SiIII_" + key][ii] = data["nuisance"]["SiIII"][key][0]
        
    results['lnprob_mle'][ii] = data["fitter"]['lnprob_mle']
    results['HCD'][ii] = data["nuisance"]["HCD"][0]


# %% [markdown]
# also save the evaluation of the IGM and contaminants for the redshifs of interest

# %%
for jj, key in enumerate(results):
    if key in ['lnprob_mle', "z"]:
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
results.keys()


# %%
def signed_exp(x):
    return np.tanh(x) * np.exp(np.exp(np.abs(x)))

def fun(x):
    return np.sinh(x)


# %%

# %%
x = np.linspace(-1, 1, 100)
plt.plot(x, signed_exp(x))
plt.plot(x, fun(x))
# plt.yscale("log")

# %%
signed_exp(-0.02)

# %%
z_X = 3
xz = np.log((1 + list_z) / (1 + z_X))
# ln_poly = np.poly1d([signed_exp(-2), np.exp(6.2)])
# y = results["SiIII_d"]
# ln_poly = np.poly1d([signed_exp(-0.01), np.exp(-4.3)])
# y = results["SiIII_f"]
# ln_poly = np.poly1d([3, 1])
# y = results["SiIII_a"]
ln_poly = np.poly1d([3.0, -1.6])
y = results["HCD"]
ln_out = ln_poly(xz)
# ln_out = ln_poly(xz)
plt.plot(list_z, np.exp(ln_out))
# plt.plot(list_z, ln_out)
plt.plot(list_z, y)

# %%
results["SiIII_d"]

# %%
z_0 = 2
z = 2.1
a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
b_0 = np.array([36.449, 81.388, 162.95, 429.58])
b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
# compute the z-dependent correction terms
a_z = a_0 * ((1 + z) / (1 + z_0)) ** a_1
b_z = b_0 * ((1 + z) / (1 + z_0)) ** b_1

# %%
k_kms = pip.pip2.fitter.like.data.k_kms[0]
y0 = (a_z[0] * np.exp(b_z[0] * k_kms) - 1) ** -2
y1 = (a_z[1] * np.exp(b_z[1] * k_kms) - 1) ** -2
y2 = (a_z[2] * np.exp(b_z[2] * k_kms) - 1) ** -2

# %%
adamp = 1.9
plt.plot(k_kms, adamp * y0)
adamp = 0.5
plt.plot(k_kms, adamp * y1)
# plt.plot(k_kms, adamp * y2)
adamp = 0.4
plt.plot(k_kms, adamp * (y0 + y1))
adamp = 0.11
plt.plot(k_kms, adamp * (y0 + y1 + y2))

# adamp = 4e-2
# plt.plot(k_kms, adamp * (np.exp(k_kms * 100)-1)**-2 )

adamp = 1.7
a = 300
plt.plot(k_kms, adamp/np.exp(k_kms * a) )

plt.xscale("log")

# %%
k_kms * 1e2

# %%
data.keys()

# %% [markdown]
# 2 par for each of nuisance

# %%

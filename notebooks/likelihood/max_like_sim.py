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
# ## Example of likelihood maximization with iMinuit
#
# This notebook contains the basic syntax required to run iMinuit on simulated P1D data

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 140
mpl.rcParams['figure.dpi'] = 140
import numpy as np
import time
# our own modules
from lace.archive import gadget_archive
from lace.archive import nyx_archive
from lace.emulator import gp_emulator
from lace.emulator import nn_emulator
from cup1d.data import data_gadget
from cup1d.data import data_nyx
from cup1d.likelihood import lya_theory
from cup1d.likelihood import likelihood
from cup1d.likelihood import iminuit_minimizer

# %% [markdown]
# ### Set up mock data
#
# Begin by picking a simulation to use as mock data, and creating a corresponding data object

# %%
# specify simulation to use to generate synthetic data
test_sim_label="growth"
if type(test_sim_label)==int:
    drop_sim=str(test_sim_label)
    print('will drop sim number {} from emulator'.format(drop_sim))
else:
    drop_sim=None

# %%
# add high-k measurement (will change emulator settings as well)
add_hires=False
if add_hires:
    kmax_Mpc=8
    polyfit_ndeg=7
else:
    kmax_Mpc=4
    polyfit_ndeg=5
    extra_data=None

# %%
# specify simulation suite and P1D mesurements
z_max=4.5
data_label='Chabanier2019'
use_nyx=False
if use_nyx:
    if drop_sim:
        drop_sim="nyx_"+drop_sim
    archive=nyx_archive.NyxArchive(verbose=True)
    data=data_nyx.Nyx_P1D(archive=archive,sim_label="nyx_"+str(test_sim_label),
                                z_max=z_max,data_cov_label=data_label,
                                polyfit_kmax_Mpc=kmax_Mpc,
                                polyfit_ndeg=polyfit_ndeg)
    # option to add extra P1D (high-resolution)
    if add_hires:
        extra_data=data_nyx.Nyx_P1D(archive=archive,
                                sim_label="nyx_"+str(test_sim_label),
                                z_max=z_max,
                                polyfit_kmax_Mpc=kmax_Mpc,
                                polyfit_ndeg=polyfit_ndeg,
                                data_cov_label='Karacayli2022')
else:
    if drop_sim:
        drop_sim="mpg_"+drop_sim
    archive=gadget_archive.GadgetArchive(postproc='Cabayol23')
    data=data_gadget.Gadget_P1D(archive=archive,sim_label="mpg_"+str(test_sim_label),
                                z_max=z_max,data_cov_label=data_label,
                                polyfit_kmax_Mpc=kmax_Mpc,
                                polyfit_ndeg=polyfit_ndeg)
    # option to add extra P1D (high-resolution)
    if add_hires:
        extra_data=data_gadget.Gadget_P1D(archive=archive,
                                sim_label="mpg_"+str(test_sim_label),
                                z_max=z_max,
                                polyfit_kmax_Mpc=kmax_Mpc,
                                polyfit_ndeg=polyfit_ndeg,
                                data_cov_label='Karacayli2022')

# %%
data.plot_p1d()

# %%
if extra_data:
    extra_data.plot_p1d()

# %% [markdown]
# ### Emulator and training set
#
# Create a set of training data to train an emulator

# %%
emu_params=["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "lambda_P"]

# %%
use_gp=False
if use_gp:
    if use_nyx:
        raise ValueError('can not use GP with Nyx emulator')
    emulator=gp_emulator.GPEmulator(archive=archive,emu_params=emu_params,
                                    drop_sim=drop_sim,kmax_Mpc=kmax_Mpc,ndeg=polyfit_ndeg)
else:
    emulator=nn_emulator.NNEmulator(archive=archive,emu_params=emu_params,
                                    drop_sim=drop_sim,kmax_Mpc=kmax_Mpc,ndeg=polyfit_ndeg)

# %%
emulator = nn_emulator.NNEmulator(
        training_set="Cabayol23", emulator_label="Cabayol23"
    )

# %% [markdown]
# ### Set up a likelihood
#
# Here we chose which parameters we want to sample, over which range and chose a prior. We pass the data and emulator objects to the likelihood.

# %%
# stick to primordial power-law parameters here
free_param_names=["As","ns"]
# specify the number of free parameters per IGM function (default=2)
n_igm=2
for i in range(n_igm):
    for par in ["tau","sigT_kms","gamma","kF"]:
        free_param_names.append('ln_{}_{}'.format(par,i))

# %%
theory = lya_theory.Theory(
    zs=data.z,
    emulator=emulator,
    free_param_names=free_param_names,
    sim_igm='mpg',
)

# %%
# option to include/remove a Gaussian prior (in unit cube)
prior_Gauss_rms=None
# option to include/ignore emulator covariance (it might bias the results)
emu_cov_factor=0
like=likelihood.Likelihood(data=data,theory=theory,
                            free_param_names=free_param_names,
                            prior_Gauss_rms=prior_Gauss_rms,
                            emu_cov_factor=emu_cov_factor,
                            extra_p1d_data=extra_data)

# %%
like.plot_p1d(residuals=True,plot_every_iz=2)

# %%
if extra_data:
    like.extra_p1d_like.plot_p1d(residuals=True,plot_every_iz=2)

# %% [markdown]
# # Try iminuit minimizer

# %%
test_values=len(free_param_names)*[0.5]
ini_chi2=like.get_chi2(values=test_values)
print('chi2 =',ini_chi2)

# %%
minimizer = iminuit_minimizer.IminuitMinimizer(like)

# %%
minimizer.minimize(compute_hesse=True)

# %%
minimizer.minimizer.errors

# %%
np.diag(np.array(minimizer.minimizer.covariance))

# %%
best_fit_values=np.array(minimizer.minimizer.values)
best_chi2=like.get_chi2(values=best_fit_values)
print('chi2 improved from {} to {}'.format(ini_chi2,best_chi2))

# %%
minimizer.plot_best_fit(plot_every_iz=2)

# %%
if extra_data:
    like.extra_p1d_like.plot_p1d(values=best_fit_values,residuals=True,plot_every_iz=2)

# %%
like.truth

# %%
np.array(minimizer.minimizer.values)

# %%
minimizer.plot_ellipses('As','ns')

# %%
like.truth

# %% [markdown]
# ### Access the actual minimizer object from iminuit

# %%
minimizer.minimizer.migrad()

# %%
#minimizer.minimizer.draw_mncontour("x0", "x1")

# %%
#minimizer.minimizer.draw_mnprofile("x0")

# %%

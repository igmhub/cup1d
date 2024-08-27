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
# This notebook contains the basic syntax required to run iMinuit on eBOSS P1D data

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 140
mpl.rcParams['figure.dpi'] = 140
import numpy as np
import time
# our own modules
from lace.emulator import gp_emulator
from lace.emulator import nn_emulator
from cup1d.data import data_Chabanier2019
from cup1d.data import data_Karacayli2022
from cup1d.likelihood import lya_theory
from cup1d.likelihood import likelihood
from cup1d.likelihood import iminuit_minimizer

# %% [markdown]
# ### Set up data (eBOSS P1D measurement from Chabanier et al. 2019)

# %%
zmin=2.7
zmax=3.3
plot_every_iz=1

# %%
data=data_Chabanier2019.P1D_Chabanier2019(zmin=zmin,zmax=zmax)

# %%
data.plot_p1d()

# %% [markdown]
# ### Set up high-res extra data (P1D measurement from Karacayli et al. 2022)

# %%
add_hires=True
if add_hires:
    extra_data=data_Karacayli2022.P1D_Karacayli2022(zmin=zmin,zmax=zmax)
    extra_data.plot_p1d()
    emu_kmax_Mpc=8
    emu_ndeg=7
else:
    extra_data=None
    emu_kmax_Mpc=4
    emu_ndeg=5

# %% [markdown]
# ### Setup an emulator (Nyx, LaCE-GP or LaCE-NN)

# %%
emulator_label="LaCE-GP"
if emulator_label=="LaCE-GP":
#    emulator=gp_emulator.GPEmulator(training_set="Cabayol23", kmax_Mpc=emu_kmax_Mpc, ndeg=emu_ndeg)
    emulator=gp_emulator.GPEmulator(training_set="Pedersen21", kmax_Mpc=emu_kmax_Mpc, ndeg=emu_ndeg)
elif emulator_label=="LaCE-NN":
    assert not add_hires,"NN emulator not trained beyond k=4 1/Mpc"
    emulator=nn_emulator.NNEmulator(training_set="Cabayol23", emulator_label="Cabayol23")
elif args.emulator_label=="Nyx":
    assert not add_hires,"Nyx emulator not trained beyond k=4 1/Mpc"
    emulator=nn_emulator.NNEmulator(training_set="Nyx23", emulator_label="Cabayol23_Nyx")
else:
    raise ValueError("wrong emulator_label",emulator_label)

# %% [markdown]
# ### Setup a likelihood
#
# Here we chose which parameters we want to sample, over which range and chose a prior. We pass the data and emulator objects to the likelihood.

# %%
free_cosmo=False
if free_cosmo:
    # stick to primordial power-law parameters here
    free_param_names=["As","ns"]
else:
    free_param_names=[]
# specify the number of free parameters per IGM function (default=2)
n_igm=1
for i in range(n_igm):
    for par in ["tau","sigT_kms","gamma","kF"]:
        free_param_names.append('ln_{}_{}'.format(par,i))
# add metal line contaminations
free_param_names.append('ln_SiIII_0')

# %%
theory=lya_theory.Theory(zs=data.z,emulator=emulator,free_param_names=free_param_names)

# %%
theory.metal_models[0].get_dv_kms()

# %%
# option to include/remove a Gaussian prior (in unit cube)
prior_Gauss_rms=None
# option to include/ignore emulator covariance (it might bias the results)
emu_cov_factor=0
like=likelihood.Likelihood(data=data,
                            theory=theory,
                            free_param_names=free_param_names,
                            prior_Gauss_rms=prior_Gauss_rms,
                            emu_cov_factor=emu_cov_factor,
                            extra_p1d_data=extra_data)

# %%
like.plot_p1d(residuals=True,plot_every_iz=plot_every_iz)

# %%
if extra_data:
    like.extra_p1d_like.plot_p1d(residuals=True,plot_every_iz=plot_every_iz)

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
best_fit_values=np.array(minimizer.minimizer.values)
print('best fit values',best_fit_values)
best_chi2=like.get_chi2(values=best_fit_values)
print('chi2 improved from {} to {}'.format(ini_chi2,best_chi2))

# %%
minimizer.plot_best_fit(plot_every_iz=plot_every_iz,residuals=True)

# %%
if extra_data:
    like.extra_p1d_like.plot_p1d(values=best_fit_values,residuals=False,plot_every_iz=plot_every_iz)

# %%
minimizer.plot_ellipses('ln_tau_0','ln_gamma_0')

# %%
minimizer.plot_ellipses('ln_sigT_kms_0','ln_gamma_0')

# %%
minimizer.plot_ellipses('ln_sigT_kms_0','ln_kF_0')

# %%
minimizer.plot_ellipses('ln_tau_0','ln_SiIII_0')

# %% [markdown]
# ### Access the actual minimizer object from iminuit

# %%
minimizer.minimizer.migrad()

# %%
#minimizer.minimizer.draw_mncontour("x0", "x1")

# %%
#minimizer.minimizer.draw_mnprofile("x0")

# %%
for ip,par in enumerate(like.free_params):
    print(ip,best_fit_values[ip],par.value_from_cube(best_fit_values[ip]),par.info_str(all_info=True))

# %%
like_params=like.parameters_from_sampling_point(best_fit_values)

# %%
F_model = like.theory.F_model_fid.get_new_model(like_params)
T_model = like.theory.T_model_fid.get_new_model(like_params)
P_model = like.theory.P_model_fid.get_new_model(like_params)

# %%
X_model = like.theory.metal_models[0].get_new_model(like_params)

# %%
print('<F>=',F_model.get_mean_flux(z=3.0))
print('T_0=',T_model.get_T0(z=3.0))
print('sigT_kms=',T_model.get_sigT_kms(z=3.0))
print('gamma=',T_model.get_gamma(z=3.0))
print('kF_kms=',P_model.get_kF_kms(z=3.0))
print('f_SiIII=',X_model.get_amplitude(z=3.0))

# %%

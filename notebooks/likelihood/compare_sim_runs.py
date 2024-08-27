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
# # Compare fits to mock data using different emulators

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 140
mpl.rcParams['figure.dpi'] = 140
import numpy as np
# our own modules
from lace.archive import gadget_archive
from lace.emulator import gp_emulator
from lace.emulator import nn_emulator
from cup1d.data import data_gadget
from cup1d.likelihood import lya_theory
from cup1d.likelihood import likelihood
from cup1d.likelihood import iminuit_minimizer

# %% [markdown]
# ### Setup test P1D data from Gadget sim

# %%
# specify simulation to use to generate synthetic data
test_sim_label="central"
if type(test_sim_label)==int:
    drop_sim=test_sim_label
    print('will drop sim number {} from emulator'.format(drop_sim))
else:
    drop_sim=None

# %%
# specify simulation suite and P1D mesurements
archive=gadget_archive.GadgetArchive(postproc='Cabayol23')

# %%
# add high-k measurement (will change emulator settings as well)
add_hires=False
if add_hires:
    kmax_Mpc=8
    polyfit_ndeg=7
else:
    kmax_Mpc=4
    polyfit_ndeg=5

# %%
# specify simulation suite and P1D mesurements
z_max=4.5
data_cov_label='Chabanier2019'
data=data_gadget.Gadget_P1D(archive=archive,
                                sim_label="mpg_"+test_sim_label,
                                z_max=z_max,
                                polyfit_kmax_Mpc=kmax_Mpc,
                                polyfit_ndeg=polyfit_ndeg,
                                data_cov_label=data_cov_label)

# %%
data.plot_p1d()

# %% [markdown]
# ### Setup emulators
#
# Each run will have its own emulator, that will be the only different thing in the run

# %%
runs={}
runs['GP']={'emulator':gp_emulator.GPEmulator(archive=archive,drop_sim=drop_sim,
                                              kmax_Mpc=kmax_Mpc,ndeg=polyfit_ndeg)}

# %%
runs['NN']={'emulator':nn_emulator.NNEmulator(archive=archive,drop_sim=drop_sim,
                                              kmax_Mpc=kmax_Mpc,ndeg=polyfit_ndeg)}

# %% [markdown]
# ### Set free parameters and theory

# %%
# stick to primordial power-law parameters here
free_param_names=["As","ns"]
# specify the number of free parameters per IGM function (default=2)
n_igm=2
for i in range(n_igm):
    for par in ["tau","sigT_kms","gamma","kF"]:
        free_param_names.append('ln_{}_{}'.format(par,i))

# %%
for label,run in runs.items():
    run['theory']=lya_theory.Theory(zs=data.z,emulator=run['emulator'],free_param_names=free_param_names)

# %% [markdown]
# ### Set up a likelihood
#
# Here we chose which parameters we want to sample, over which range and chose a prior. We pass the data and theory objects to the likelihood.

# %%
# option to include/remove a Gaussian prior (in unit cube)
prior_Gauss_rms=0.2
# option to include/ignore emulator covariance (it might bias the results)
emu_cov_factor=0
for label,run in runs.items():
    run['likelihood']=like=likelihood.Likelihood(data=data,theory=run['theory'],
                            free_param_names=free_param_names,
                            prior_Gauss_rms=prior_Gauss_rms,
                            emu_cov_factor=emu_cov_factor)

# %%
# check starting point for free parameters (should be equal to truth)
test_values=len(free_param_names)*[0.5]
for label,run in runs.items():
    print('run',label)
    like=run['likelihood']
    for p in like.parameters_from_sampling_point(values=test_values):
        print(p.info_str(all_info=True))
    print('chi2 =',like.get_chi2(values=test_values))
    like.plot_p1d(values=test_values,residuals=True,plot_every_iz=2)
    print('----------------')

# %% [markdown]
# ### Run iminuit minimizer

# %%
# choose starting point for free parameters (within 0.5 +/- ini_sigma, in the unit cube)
ini_sigma=0.0
ini_values=2*ini_sigma*np.random.random(len(free_param_names))+0.5-ini_sigma
for label,run in runs.items():
    print('minimize run',label)
    # what is the chi2 of the starting point?
    ini_chi2=run['likelihood'].get_chi2(values=ini_values)
    run['minimizer']=iminuit_minimizer.IminuitMinimizer(run['likelihood'],ini_values=ini_values)
    run['minimizer'].minimize(compute_hesse=True)
    # what is the chi2 of the best-fit? (should be close to 0)
    best_fit_values=np.array(run['minimizer'].minimizer.values)
    best_chi2=run['likelihood'].get_chi2(values=best_fit_values)
    print('chi2 improved from {} to {}'.format(ini_chi2,best_chi2))
    

# %%
for label,run in runs.items():
    print('run',label)
    run['minimizer'].plot_best_fit(plot_every_iz=2)

# %%
for label,run in runs.items():
    plt.figure()
    print('run',label)
    run['minimizer'].plot_ellipses('As','ns')
    plt.savefig(label+'_As_ns.png')

# %%

# %%

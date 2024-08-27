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
from lace.emulator import gp_emulator
from lace.emulator import nn_emulator
from cup1d.data import mock_data
from cup1d.likelihood import lya_theory
from cup1d.likelihood import likelihood
from cup1d.likelihood import iminuit_minimizer

# %% [markdown]
# ### Setup emulators
#
# Each run will have its own emulator, that will be the only different thing in the run

# %%
# setup run for Pedersen21 emulator
runs={}
runs['Pedersen21']={'emulator':gp_emulator.GPEmulator(training_set='Pedersen21',emulator_label='Pedersen21')}

# %%
# setup run for Pedersen23 emulator
runs['Pedersen23']={'emulator':gp_emulator.GPEmulator(training_set='Pedersen21',emulator_label='Pedersen23')}

# %%
runs['LaCE-GP']={'emulator':gp_emulator.GPEmulator(training_set='Cabayol23',emulator_label='Cabayol23')}

# %%
runs['test-C23-P23']={'emulator':gp_emulator.GPEmulator(training_set='Cabayol23',emulator_label='Pedersen23')}

# %%
runs['test-P12-C23']={'emulator':gp_emulator.GPEmulator(training_set='Pedersen21',emulator_label='Cabayol23')}

# %%
# setup run for Cabayol23 NN emulator
runs['Cabayol23']={
    'emulator':nn_emulator.NNEmulator(
        training_set='Cabayol23',
        emulator_label='Cabayol23', 
        model_path="NNmodels/Cabayol23/Cabayol23.pt",
        train=False,
    )
}

# %%
# setup run for the Nyx emulator
runs['Nyx']={
    'emulator':nn_emulator.NNEmulator(
        training_set='Nyx23_Oct2023',
        emulator_label='Nyx_v0', 
        model_path="NNmodels/Nyx23_Oct2023/Nyx_v0.pt",
        train=False,
    )
}

# %% [markdown]
# ### Create mock P1D data
#
# Create mock data, mimicking a particular dataset, using the emulator to create the signal

# %%
# data_label="QMLE_Ohio"
data_label="Chabanier2019"

# %%
for label,run in runs.items():
    print('label',label)
    run['data']=mock_data.Mock_P1D(emulator=run['emulator'],data_label=data_label)

# %%
for label,run in runs.items():
    data=run['data']
    print(label)
    data.plot_p1d()


# %%
def plot_p1d(runs,iz):
    for label,run in runs.items():
        data=run['data']
        z=data.z[iz]
        print(label,iz,z)
    
        k_kms=data.k_kms
        Pk_kms=data.get_Pk_iz(iz)
        plt.plot(k_kms[:10],Pk_kms[:10],label=label)
        
    plt.title('z = {}'.format(z))
    plt.legend()
    plt.xlabel('k [s/km]')
    plt.ylabel('P(k) [km/s]')


# %%
plot_p1d(runs,2)

# %%

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
    run['theory']=lya_theory.Theory(zs=run['data'].z,emulator=run['emulator'],free_param_names=free_param_names)

# %% [markdown]
# ### Set up a likelihood
#
# Here we chose which parameters we want to sample, over which range and chose a prior. We pass the data and theory objects to the likelihood.

# %%
# option to include/remove a Gaussian prior (in unit cube)
prior_Gauss_rms=None
# option to include/ignore emulator covariance (it might bias the results)
emu_cov_factor=0
for label,run in runs.items():
    run['likelihood']=like=likelihood.Likelihood(data=run['data'],theory=run['theory'],
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
#     plt.savefig(label+'_As_ns.png')

# %%

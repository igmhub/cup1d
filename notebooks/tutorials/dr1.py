# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial for DR1 data

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Load P1D measurements and set likelihood

# %%
config_file = os.path.join(
    get_path_repo("cup1d"), "configs", "cm2026", "cm2026_base.yaml"
)
args = Args.from_yaml(config_file, verbose=False)
analysis = Analysis(args)

# %% [markdown]
# ## Plot P1D data 
#
# Get parameters from a point of the parameter space close to the best fit

# %%
p0 = analysis.fitter.sampling_point_from_parameters().copy()
free_params = analysis.fitter.parameters_from_sampling_point(p0)
analysis.like.get_chi2(free_params)

# %% [markdown]
# Plot model for these parameters

# %%
analysis.like.plot_p1d(free_params)

# %% [markdown]
# #### If you want to extract the data

# %%
# measurements in z bins (no correlation between z bins)
key = list(analysis.data.keys())[0]

k_kms = analysis.data[key].k_kms
Pk_kms = analysis.data[key].Pk_kms
cov_Pk_kms = analysis.like.cov_Pk_kms[key]
print(len(k_kms), len(Pk_kms), len(cov_Pk_kms))
print(k_kms[0].shape, Pk_kms[0].shape, cov_Pk_kms[0].shape)

# %%
# measurements in full array (correlation between z bins)
k_kms = analysis.data[key].full_k_kms
Pk_kms = analysis.data[key].full_Pk_kms
cov_Pk_kms = analysis.like.full_cov_Pk_kms[key]
print(k_kms.shape, Pk_kms.shape, cov_Pk_kms.shape)

# %% [markdown]
# #### Components of the covariance matrix

# %%
analysis.like.plot_cov_to_pk()

# %%
# access the components of the covariance matrix
key = list(analysis.data.keys())[0]

## stat + sys + emu
cov_Pk_kms_tot = analysis.like.cov_Pk_kms[key]
## stat
cov_Pk_kms_stat = analysis.data[key].covstat_Pk_kms
## syst
cov_Pk_kms_syst = []
for ii in range(len(cov_Pk_kms_stat)):
    cov_Pk_kms_syst.append(analysis.data[key].cov_Pk_kms[ii] - cov_Pk_kms_stat[ii])
## emu
cov_Pk_kms_emu = analysis.like.cov_emu_Pk_kms[key]

print(
    len(cov_Pk_kms_tot), len(cov_Pk_kms_stat), len(cov_Pk_kms_syst), len(cov_Pk_kms_emu)
)

# %% [markdown]
# ### Get predictions from the model

# %%
# list of model parameters

for par in analysis.like.free_params.values():
    print(par["name"], par["value"], par["min_value"], par["max_value"])

# %% [markdown]
# #### Evaluate the model for some input parameters

# %%
# evaluate model for the initial value of the input parameters
zs = analysis.data[key].z
k_kms = analysis.data[key].k_kms
ini_free_params = {
    name: parameter["value"]
    for name, parameter in analysis.like.free_params.items()
}

ini_model_Pk_kms = analysis.theory.get_p1d_kms(
    zs, k_kms, like_params=ini_free_params
)[0]

# %%
# evaluate model for other values of input parameters, only changing As
zs = analysis.data[key].z
k_kms = analysis.data[key].k_kms

new_free_params = ini_free_params.copy()
new_free_params["As"] *= 1.1

new_As_model_Pk_kms = analysis.theory.get_p1d_kms(
    zs, k_kms, like_params=new_free_params
)[0]

# %% [markdown]
# Show ratio of both predictions

# %%
ii = 0
plt.plot(
    k_kms[ii],
    new_As_model_Pk_kms[ii] / ini_model_Pk_kms[ii],
    label="z = " + str(zs[ii]),
)
plt.ylabel("$P_\mathrm{1D}(k)/P_\mathrm{1D}^\mathrm{ini}(k)$")
plt.xlabel("$k_\parallel$ [km/s]")
plt.legend()
plt.show()

# %% [markdown]
# ### Compressed parameters

# %%
blob = analysis.theory.get_blob_for_parameters(ini_free_params)
ini_Delta2_star = blob[0]
ini_n_star = blob[1]
print(ini_Delta2_star, ini_n_star)

# %%
blob = analysis.theory.get_blob_for_parameters(new_free_params)
new_Delta2_star = blob[0]
new_n_star = blob[1]
print(new_Delta2_star, new_n_star)

# %%
# as expected, 10% larger value of the new parameter
new_Delta2_star/ini_Delta2_star

# %% [markdown]
# ## Run minimizer

# %% [markdown]
# Get value of parameters close to best fit again

# %%
p0 = analysis.fitter.sampling_point_from_parameters().copy()
free_params = analysis.fitter.parameters_from_sampling_point(p0)
analysis.like.get_chi2(free_params)

# %% [markdown]
# Run minimizer starting from this point, it should stop the minimization soon

# %%
analysis.run_minimizer(p0)

# %% [markdown]
# Evaluate for the new best fit

# %%
p1 = analysis.fitter.mle_cube
analysis.like.plot_p1d(analysis.fitter.parameters_from_sampling_point(p1))

# %% [markdown]
# Read chain

# %%
path = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/blobs.npy"
base_chain = np.load(path)

results = {}
results["Delta2_star"] = base_chain["Delta2_star"].reshape(-1)
results["n_star"] = base_chain["n_star"].reshape(-1)

for par in results:
    print(np.median(results[par]))

# %% [markdown]
# Apply unblinding

# %%
from cup1d.utils.blinding import apply_unblinding

results_unblind = apply_unblinding(analysis.like.blind, results)

# %%
for par in results_unblind:
    print(np.median(results_unblind[par]))

# %% [markdown]
# Compare the unblinded sampler samples with the unblinded MLE. The blinding
# offsets are additive, so they are removed from the MLE location but not from
# its covariance-derived errors.

# %%
analysis.fitter.estimate_mle_errors(method="gauss_newton")
mle_cosmo_unblind = apply_unblinding(
    analysis.like.blind, analysis.fitter.mle_cosmo.copy()
)
from cup1d.postprocessing import plot_cosmo_sampler_and_fit

samples = np.column_stack(
    [results_unblind["Delta2_star"], results_unblind["n_star"]]
)
fig = plot_cosmo_sampler_and_fit(
    samples,
    mle_cosmo_unblind,
    analysis.fitter.mle_cosmo_covariance[:2, :2],
    sampler_label="Sampler contours",
    fit_label="Minimizer",
)
plt.show()

# %% [markdown]
# ## Reload the saved minimizer result
#
# The YAML configuration rebuilds the likelihood, after which the saved fit
# state is restored. Loading does not create another output directory.

# %%
minimizer_results = os.path.join(
    analysis.fitter.save_directory, "minimizer_results.npy"
)
restored_analysis = Analysis.from_results(minimizer_results)
print(restored_analysis.fitter.mle_chi2)
print(restored_analysis.fitter.mle)

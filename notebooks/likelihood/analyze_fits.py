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
# # Analyse a saved fit
#
# This notebook reconstructs an analysis from its YAML configuration and uses
# the maximum-likelihood point saved in `fitter_results.npy`. The YAML file
# must describe the fit being inspected: the saved results do not contain the
# complete analysis configuration.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Select the configuration and saved fit
#
# Set `fit_directory` to the directory containing `fitter_results.npy`. The
# default points to the CM2026 baseline configuration and to the standard DR1
# QMLE3 global fit. Change both settings together when inspecting a variation.

# %%
repository_path = Path(get_path_repo("cup1d"))
config_path = repository_path / "configs" / "cm2026" / "cm2026_base.yaml"

fit_directory = Path(
    "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
    "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7"
)
results_path = fit_directory / "fitter_results.npy"

if not results_path.is_file():
    raise FileNotFoundError(
        f"Could not find {results_path}. Edit fit_directory in this cell."
    )


# %% [markdown]
# ## Reconstruct the analysis and load the maximum-likelihood point
#
# `mle_cube` is stored in the unit cube used by the likelihood. Evaluating its
# chi-squared again checks that the selected configuration matches the fit and
# makes the point available to the regular fitting interface.

# %%
args = Args.from_yaml(config_path, verbose=False)
analysis = Analysis(args)

fit_results = np.load(results_path, allow_pickle=True).item()
mle_cube = np.asarray(fit_results["fitter"]["mle_cube"])
best_fit_parameters = analysis.fitter.parameters_from_sampling_point(mle_cube)
chi2 = analysis.like.get_chi2(best_fit_parameters)
analysis.fitter.set_mle(mle_cube, chi2)

print(f"chi2 = {chi2:.3f}")
print(f"number of free parameters = {len(analysis.like.free_params)}")


# %% [markdown]
# ## Inspect the fitted parameters
#
# The likelihood converts the unit-cube point into the physical parameter
# values used by the model.

# %%
for name, value in best_fit_parameters.items():
    print(f"{name:20s} = {value:g}")


# %% [markdown]
# ## Compare the best-fit model with the P1D measurements
#
# The residual-panel view is useful for locating redshift or wavenumber ranges
# that dominate the goodness of fit. No figures are saved unless a filename is
# explicitly supplied to the plotting method.

# %%
analysis.like.plot_p1d(best_fit_parameters, residuals=True, plot_panels=True, print_chi2=False)


# %% [markdown]
# ## Plot the fitted IGM history
#
# For a minimizer result we show the maximum-likelihood history. The plot also
# overlays Gaikwad et al. (2021) and Turner et al. (2024) mean-flux data as
# tau_eff, and the Gaikwad temperature measurements as sigma_T. To show
# posterior bands, load a sampler chain and pass it as `chain_uformat`.

# %%
analysis.like.plot_igm(free_params=best_fit_parameters)


# %% [markdown]
# ## Inspect the covariance used by the likelihood
#
# This plot separates the statistical, systematic, and emulator contributions
# to the P1D covariance where they are available.

# %%
analysis.like.plot_cov_to_pk()

# %%

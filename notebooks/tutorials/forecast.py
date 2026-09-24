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
# # Tutorial: forecast analysis
#
# This tutorial builds an in-memory synthetic P1D data set on the DESI DR1
# QMLE3 grid and covariance, minimizes the CUP1D likelihood, and compares the
# initial and minimized predictions. The forecast has no random noise by
# default; set `add_noise: true` in the YAML configuration to make one noisy
# realization. Its contaminants begin at their null-model values rather than
# the DR1-informed initial values used by the observational baseline.

# %% [markdown]
# Load the YAML configuration for the synthetic forecast and construct the
# analysis. Creating `Analysis` generates the forecast P1D from the true
# theory model.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo

config_path = (
    Path(get_path_repo("cup1d"))
    / "configs"
    / "forecasts"
    / "DESIY1_QMLE3_CH24_mpgcen_gpr.yaml"
)
args = Args.from_yaml(config_path, synthetic=True, verbose=False)
analysis = Analysis(args)


# %% [markdown]
# ## Inspect the forecast P1D
#
# Evaluate the likelihood at its initial point and plot the forecast together
# with the corresponding model prediction.

# %%
initial_point = analysis.fitter.sampling_point_from_parameters().copy()
initial_parameters = analysis.fitter.parameters_from_sampling_point(initial_point)
initial_chi2 = analysis.like.get_chi2(initial_parameters)
print(f"Initial chi2 = {initial_chi2:.3f}")
analysis.like.plot_p1d(initial_parameters, residuals=True, plot_panels=True)


# %% [markdown]
# ## Minimize the forecast likelihood
#
# Start from the initial point and minimize over all redshift bins. This can
# take several minutes for the full global fit.

# %%
analysis.run_minimizer(initial_point, restart=True)
best_fit_point = analysis.fitter.mle_cube
print(f"Best-fit chi2 = {analysis.fitter.mle_chi2:.3f}")


# %% [markdown]
# ## Inspect the minimized model
#
# Plot the forecast P1D and the model evaluated at the best-fit parameters.

# %%
best_fit_parameters = analysis.fitter.parameters_from_sampling_point(best_fit_point)
analysis.like.plot_p1d(best_fit_parameters, residuals=True, plot_panels=True)


# %% [markdown]
# ## Compare the fitted and true IGM histories
#
# Plot the best-fit IGM evolution together with the truth used to construct
# the forecast. External observational measurements are omitted.

# %%
best_fit_parameters = analysis.fitter.parameters_from_sampling_point(best_fit_point)
analysis.like.plot_igm(
    free_params=best_fit_parameters,
    plot_external_data=False,
    plot_truth=True,
    variation_label="Best fit",
)


# %% [markdown]
# ## Compare cosmologies in the emulator training set
#
# Show the best-fit compressed linear-power cosmology, the true cosmology used
# for the forecast, and every simulation cosmology that trained the emulator.

# %%
from cup1d.postprocessing import Plotter

plotter = Plotter(analysis.fitter)
plotter.plot_mle_cosmo(plot_errors=True, error_method="gauss_newton")

# %%
analysis.fitter.mle_cosmo_errors

# %%

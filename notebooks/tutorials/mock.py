# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Tutorial: mock analysis
#
# This tutorial fits a synthetic P1D mock based on the MPG central simulation.
# The mock uses the DESI DR1 QMLE3 covariance and Planck18 for the current
# theory-to-velocity conversion. Its contaminants start at null-model values,
# and it is noiseless by default.
# Set `add_noise: true` in the YAML configuration to analyze a noisy mock
# realization instead.

# %% [markdown]
# Load the YAML configuration and construct the mock data set. `Analysis`
# loads the simulation P1D, applies the selected mock-data processing, and
# builds the likelihood.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo

config_path = (
    Path(get_path_repo("cup1d"))
    / "configs"
    / "mocks"
    / "mpg_central"
    / "DESIY1_QMLE3_CH24_mpgcen_gpr.yaml"
)
args = Args.from_yaml(config_path, synthetic=True, verbose=False)
analysis = Analysis(args)


# %% [markdown]
# ## Inspect the mock P1D
#
# Evaluate the initial likelihood point and plot the mock P1D with the model
# prediction evaluated at that point.

# %%
initial_point = analysis.like.sampling_point_from_parameters().copy()
initial_chi2 = analysis.like.get_chi2(initial_point)
print(f"Initial chi2 = {initial_chi2:.3f}")
analysis.like.plot_p1d(initial_point, residuals=True, plot_panels=True)


# %% [markdown]
# ## Minimize the mock likelihood
#
# Minimize the global likelihood from the initial point. This can take several
# minutes for the full global fit.

# %%
analysis.run_minimizer(initial_point, restart=True)
best_fit_point = analysis.fitter.mle_cube
print(f"Best-fit chi2 = {analysis.fitter.mle_chi2:.3f}")


# %% [markdown]
# ## Inspect the minimized model
#
# Compare the mock P1D with the best-fit model after minimization.

# %%
analysis.like.plot_p1d(best_fit_point, residuals=True, plot_panels=True)


# %% [markdown]
# ## Compare the fitted and true IGM histories
#
# Plot the best-fit IGM evolution together with the truth used to construct
# the mock. External observational measurements are omitted.

# %%
best_fit_parameters = analysis.like.parameters_from_sampling_point(best_fit_point)
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
# for the mock, and every simulation cosmology that trained the emulator.

# %%
from cup1d.postprocessing import Plotter

plotter = Plotter(analysis.fitter)
plotter.plot_mle_cosmo()

# %%

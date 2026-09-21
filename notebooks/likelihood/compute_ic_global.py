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
# # Compute initial conditions from global fits
#
# The first stage fits each redshift independently and writes
# `mpg_ic_at_a_time.npy`. The `global_all` diagnostic below reads those values
# at the eleven DR1 redshifts. The reduced `global_opt` fit then produces
# `mpg_ic_global_red.npy`, which is used as the initial point for the baseline
# analysis.
#
# Generate the files reproducibly with:
#
# ```bash
# python scripts/create_at_a_time_initial_conditions.py \
#     configs/cm2026/variations/at_a_time_global_QMLE3.yaml
# python scripts/create_global_initial_conditions.py \
#     configs/cm2026/variations/global_opt_ic_QMLE3.yaml
# ```
#
# The second YAML has `file_ic: null`, the direct replacement for the legacy
# `ic_global=False`: it prevents an old global IC file from being read while
# creating its replacement. This notebook is for inspecting the two stages;
# it never writes an IC file.

# %% [markdown]
# Load the YAML-based interface and utilities used to locate the configurations.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Inspect the all-node starting point
#
# This diagnostic has one node at every DR1 redshift and loads the at-a-time
# IC file. It confirms that the independent-redshift results are transferred
# correctly to the full global parameterization.

# %%
config_dir = Path(get_path_repo("cup1d")) / "configs" / "cm2026" / "variations"
all_args = Args.from_yaml(config_dir / "global_all_QMLE3.yaml", verbose=False)
all_analysis = Analysis(all_args)
all_initial_point = all_analysis.like.sampling_point_from_parameters().copy()
all_initial_chi2 = all_analysis.like.get_chi2(all_initial_point)
print(f"All-node initial chi2 = {all_initial_chi2:.3f}")


# %% [markdown]
# Plot the P1D residuals for the all-node model initialized from the local fits.

# %%
all_analysis.like.plot_p1d(
    all_initial_point,
    residuals=True,
    plot_panels=True,
    print_chi2=False,
)


# %% [markdown]
# ## Fit the reduced global model
#
# The reduced global model is the one used to create the standard global IC
# file. Its YAML explicitly disables input global ICs, so this cell can safely
# regenerate the fit from its native parameter defaults.

# %%
global_args = Args.from_yaml(
    config_dir / "global_opt_ic_QMLE3.yaml", verbose=False
)
global_analysis = Analysis(global_args)
global_initial_point = global_analysis.like.sampling_point_from_parameters().copy()
global_initial_chi2 = global_analysis.like.get_chi2(global_initial_point)
print(f"Reduced global initial chi2 = {global_initial_chi2:.3f}")


# %% [markdown]
# Run the global minimization. It can take several minutes. The command-line
# script above performs this same operation and saves `mpg_ic_global_red.npy`.

# %%
global_analysis.run_minimizer(global_initial_point, restart=True)
global_point = global_analysis.fitter.mle_cube
print(f"Reduced global minimized chi2 = {global_analysis.fitter.mle_chi2:.3f}")


# %% [markdown]
# Inspect the minimized reduced-global P1D prediction.

# %%
global_analysis.like.plot_p1d(
    global_point,
    residuals=True,
    plot_panels=True,
    print_chi2=False,
)

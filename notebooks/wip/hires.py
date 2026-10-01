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
# ### Combining low- and high-resolution data
#
# Select either the LaCE Gaussian-process emulator or ForestFlow. Both YAML
# files use the same $k_\parallel \leq 0.07\,\mathrm{s\,km^{-1}}$ data cut.

# %%
# %load_ext autoreload
# %autoreload 2

import os
from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Load P1D measurements and set likelihood

# %%
# Choose ``"lace"`` for the ``lace_mpg`` alias or ``"forestflow"`` for
# the ``forest_mpg`` alias.
config_names = {
    "lace": "hires.yaml",
    "forestflow": "hires_forestflow.yaml",
}

# emulator_family = "lace"
emulator_family = "forestflow"

config_file = os.path.join(
    get_path_repo("cup1d"), "configs", "hires", config_names[emulator_family]
)
args = Args.from_yaml(config_file, verbose=False)
analysis = Analysis(args)
print(f"Using {args.emulator_label} with k_parallel <= {args.kmax_ikms} s/km")

# %% [markdown]
# ## Plot P1D data
#
# Get parameters from a point of the parameter space close to the best fit

# %%
# %%time
free_params = analysis.fitter.initial_parameters()
analysis.like.get_chi2(free_params)

# %%
# %%time
analysis.like.get_chi2(free_params)

# %% [markdown]
# Plot model for these parameters

# %%
analysis.like.plot_p1d(free_params)

# %% [markdown]
# ## Run minimizer

# %% [markdown]
# Run minimizer starting from this point, it should stop the minimization soon

# %%
# %%time
analysis.run_minimizer(free_params)

# %%
best_params = analysis.fitter.mle_cube
analysis.like.get_chi2(best_params)

# %%
analysis.like.plot_p1d(best_params)

# %%

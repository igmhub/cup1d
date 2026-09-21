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
# # Analysis variations
#
# This notebook shows the emulator prior domains and compares selected DESI
# DR1 analysis variations. The chain locations, labels, contour loading, and
# plotting live in `cup1d.postprocessing.VariationPlotter`.

# %%
# %load_ext autoreload
# %autoreload 2

from cup1d.postprocessing import EmulatorPriorPlotter, VariationPlotter


# %% [markdown]
# ## Emulator priors
#
# Map the primordial-spectrum ranges covered by the MPG and Nyx emulators to
# the $\Delta^2_\star$--$n_\star$ plane, and show their simulation cosmologies.

# %%
emulator_priors = EmulatorPriorPlotter()
emulator_priors.compute_priors()
emulator_priors.plot_priors()


# %% [markdown]
# ## Available variation comparisons
#
# Each key below is a named group of chains. The class loads a chain only when
# it is requested, so inspecting this list does not require the results files.

# %%
variation_plotter = VariationPlotter(prior_plotter=emulator_priors)
available_groups = variation_plotter.available_groups()
for group, members in available_groups.items():
    print(f"{group:24s}: {', '.join(members)}")


# %% [markdown]
# ## Plot one or more comparisons
#
# Pass a single group name for one comparison, or a list to make a panel for
# every requested group. For example, change the list below to
# `['data', 'emulator', 'metals']`. To save PDF and PNG files, add
# `save_figures=True` (and optionally set `save_directory`).

# %%
variation_plotter.plot(["cosmology_bounds"], save_figures=False)

# %%

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
# # Historical linear-power constraints
#
# Compare constraints on the linear matter-power amplitude
# $\Delta^2_\star$ and slope $n_\star$, both evaluated at $z=3$ and
# $k_\star=0.009\,\mathrm{s}\,\mathrm{km}^{-1}$. The figure combines the
# Planck LCDM posterior, DESI DR1 P1D, and the SDSS, BOSS, and eBOSS P1D
# measurements used in earlier analyses.

# %%
# %load_ext autoreload
# %autoreload 2
from cup1d.postprocessing.historical_linear_power import (
    HistoricalLinearPowerPlotter,
)

# %% [markdown]
# Load all inputs and make the comparison figure. The plotter reads the Planck
# chain, DESI DR1 contour polygons and posterior samples, and the blinding
# offsets from their standard locations; it unblinds the DESI quantities
# internally. The blue markers are MPG simulation cosmologies. This cell does
# not write anything to disk.

# %%
historical_plotter = HistoricalLinearPowerPlotter().load_data()
figure, axes = historical_plotter.plot()

# %% [markdown]
# The plotter returns the Matplotlib figure and axes, so standard interactive
# editing or an explicit ``figure.savefig(...)`` call remains possible. To
# export the compact publication-data dictionary for Figure 18, explicitly run
# the cell below; it writes ``data/zenodo/fig_18.npy``.

# %%
# historical_plotter.save_data_to_zenodo()

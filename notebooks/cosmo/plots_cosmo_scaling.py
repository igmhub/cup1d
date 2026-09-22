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
# # Cosmological dependence after matching star parameters
#
# How much spectral dependence remains when backgrounds differ but two or
# three compressed parameters agree? At z_star=3 and k_star=0.009 s/km we
# compare no scaling, matching Delta2_star/n_star, and also matching alpha_star.
# Each background is computed with LaCE Cosmology; primordial changes reuse
# it through RescaledCosmology. The plots show P/P_fid - 1.
#
# P1D here is a Gaussian-damped projection of linear CDM+baryon power,
# not a prediction of the observed Lyman-alpha flux power. ForestFlow supplies
# the projection. This notebook requires an updated local ForestFlow install.

# %% [markdown]
# ## Configuration
#
# Preserve the original massless-neutrino fiducial rather than using today's
# default cosmology. k grids, integration limits and damping scale are in s/km.
# The 3D curves use one common log grid over the original displayed range.
# CAMB's comoving upper limit is increased automatically if required to cover
# sqrt(k_parallel^2 + k_perp^2) for any background.
#
# D=H(z)/(1+z) differs by background: k_Mpc=D*k_kms, P3D_kms=D^3*P3D_Mpc,
# whereas P1D_kms=D*P1D_Mpc. Damping exp[-(k_kms/0.4)^2] applies to total k
# before projection. Direct evaluation avoids the old tabulation endpoint
# clamping. The original 99-point integral changed residuals by up to 3.9e-4
# relative to 397 points, so this notebook uses 397 points. Set n_k_perp=99
# to inspect the original resolution; the transverse bounds remain unchanged.

# %%
import numpy as np
from matplotlib import pyplot as plt
from cup1d.postprocessing.cosmo_scaling import CosmoScalingPlotter

fiducial_parameters = dict(
    H0=67.66,
    mnu=0.0,
    omch2=0.119,
    ombh2=0.0224,
    omk=0.0,
    As=2.105e-9,
    ns=0.9665,
    nrun=0.0,
    pivot_scalar=0.05,
    w=-1.0,
)
plotter = CosmoScalingPlotter(
    fiducial_parameters=fiducial_parameters,
    z_star=3.0,
    k_star_kms=0.009,
    scans=dict(
        omch2=np.linspace(0.1071, 0.1309, 6),
        H0=np.linspace(57.66, 77.66, 6),
        mnu=np.linspace(0.06, 0.3, 6),
    ),
    k_parallel_kms=np.linspace(0.001, 0.04, 40),
    k_linear_kms=np.geomspace(0.001, 0.1, 300),
    kpressure_kms=0.4,
    k_perp_min=1e-6,
    k_perp_max=5.0,
    n_k_perp=397,
    camb_kmax_mpc=400.0,
)

# %% [markdown]
# ## Fiducial projection
#
# The reference curve shows dimensionless k_parallel P1D/pi.

# %%
fig, ax = plotter.plot_fiducial()
plt.show()

# %% [markdown]
# ## omch2: damped 1D projection
#
# Vary the physical CDM density while keeping H0, baryon density and neutrino mass fixed.
# Panels compare no matching, matching amplitude/slope, and matching all three
# star parameters to the same fiducial. Values and primordial corrections are
# cached; each matching calculation verifies the recovered target parameters.

# %%
fig, axes, omch2_p1d_data = plotter.plot("omch2", "p1d")
plt.show()

# %% [markdown]
# ## omch2: linear 3D spectrum
#
# The same backgrounds and matching cases are compared at identical velocity
# wavenumbers. Dotted vertical lines mark the star pivot. These curves use
# undamped linear power; damping is applied only when projecting to P1D.

# %%
fig, axes, omch2_linear_data = plotter.plot("omch2", "linear")
plt.show()

# %% [markdown]
# ## H0: damped 1D projection
#
# Vary H0 while keeping the physical matter densities fixed. Labels show h=H0/100.
# Panels compare no matching, matching amplitude/slope, and matching all three
# star parameters to the same fiducial. Values and primordial corrections are
# cached; each matching calculation verifies the recovered target parameters.

# %%
fig, axes, h0_p1d_data = plotter.plot("H0", "p1d")
plt.show()

# %% [markdown]
# ## H0: linear 3D spectrum
#
# The same backgrounds and matching cases are compared at identical velocity
# wavenumbers. Dotted vertical lines mark the star pivot. These curves use
# undamped linear power; damping is applied only when projecting to P1D.

# %%
fig, axes, h0_linear_data = plotter.plot("H0", "linear")
plt.show()

# %% [markdown]
# ## H0: detailed matched residuals
#
# Solid curves match amplitude, slope and curvature; dashed curves match
# amplitude and slope. Colors identify the same background in both cases.

# %%
fig, ax, _ = plotter.plot_matched_detail("H0")
plt.show()

# %% [markdown]
# ## mnu: damped 1D projection
#
# Vary the summed neutrino mass in eV, compensating omch2 with CAMB's neutrino density to hold omch2+omnuh2 fixed, as in the latest original notebook.
# Panels compare no matching, matching amplitude/slope, and matching all three
# star parameters to the same fiducial. Values and primordial corrections are
# cached; each matching calculation verifies the recovered target parameters.

# %%
fig, axes, mnu_p1d_data = plotter.plot("mnu", "p1d")
plt.show()

# %% [markdown]
# ## mnu: linear 3D spectrum
#
# The same backgrounds and matching cases are compared at identical velocity
# wavenumbers. Dotted vertical lines mark the star pivot. These curves use
# undamped linear power; damping is applied only when projecting to P1D.

# %%
fig, axes, mnu_linear_data = plotter.plot("mnu", "linear")
plt.show()

# %% [markdown]
# ## mnu: detailed matched residuals
#
# Solid curves match amplitude, slope and curvature; dashed curves match
# amplitude and slope. Colors identify the same background in both cases.

# %%
fig, ax, _ = plotter.plot_matched_detail("mnu")
plt.show()

# %% [markdown]
# ## Optional figure and Zenodo export
#
# Nothing is saved by default. Set save_path in a plotting call to save a
# figure, or save_data=True to write its exact curves under data/zenodo.
# This only writes local files; it does not upload to Zenodo.
# Historical filenames: H0 fig_A1a/b, mnu fig_A2a/b, omch2 fig_A3a/b
# (a=linear, b=P1D). The dictionaries include the correct x coordinates,
# y0_i/y1_i/y2_i residuals, units, background and matching parameters.

# %%
# Example: uncomment only when you want to save.
# fig, axes, data = plotter.plot("omch2", "linear", save_path="scaling.pdf", save_data=True)
# plotter.save_data_to_zenodo(omch2_linear_data, output_directory="/your/export/directory")

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
# # Compute star parameters
#
# The star parameters describe the linear matter power spectrum at a chosen
# redshift and velocity-space pivot scale. They are calculated directly by
# the LaCE cosmology class.

# %% [markdown]
# Select the fiducial cosmology and the same pivot settings used by the
# CM2026 analysis.

# %%
from lace.cosmo import cosmology

fiducial_cosmology = cosmology.Cosmology(cosmo_label="Planck18")
z_star = 3.0
kp_kms = 0.009

# %% [markdown]
# Compute the dimensionless amplitude, slope, and running of the linear power
# spectrum at the selected pivot.

# %%
star_params = fiducial_cosmology.get_linP_kms_params(z_star, kp_kms)
star_params

# %%

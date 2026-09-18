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
# # McDonald et al. (2005) HCD-contamination model
#
# This notebook illustrates the phenomenological model used to account for
# contamination by High Column Density (HCD) absorbers in the Ly-alpha forest
# power spectrum.  It follows the prescription motivated by McDonald et al.
# (2005), with the scale dependence used in subsequent BOSS analyses.

# %% [markdown]
# At each redshift, the model multiplies the uncontaminated one-dimensional
# flux-power prediction by
#
# $$C_\mathrm{HCD}(k,z)=1+A_\mathrm{damp}(z)
# \left[0.018+\frac{1}{15000k-8.9}\right].$$
#
# Here $k$ is in s/km.  The amplitude is parameterized as
#
# $$\ln A_\mathrm{damp}(z)=\mathrm{poly}\left[
# \ln\!\left(\frac{1+z}{1+z_0}\right)\right], \qquad z_0=3.$$
#
# The polynomial coefficients are the nuisance parameters sampled by the
# likelihood.  Setting the final coefficient to a sufficiently negative value
# switches off the contamination entirely.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from cup1d.models.contaminants.HCD.hcd_model_McDonald2005 import (
    HCD_Model_McDonald2005,
)

# %% [markdown]
# First construct a one-parameter model.  A coefficient of zero corresponds to
# an HCD amplitude of one at the reference redshift.

# %%
hcd_model = HCD_Model_McDonald2005(ln_A_damp_coeff=[0.0])

# %% [markdown]
# The model exposes its coefficient as a likelihood parameter, including the
# default bounds used in an inference run.

# %%
for p in hcd_model.params:
    print(p.info_str())

# %% [markdown]
# Evaluate the corresponding HCD amplitude at a redshift well away from the
# reference value.

# %%
hcd_model.get_A_damp(z=10)

# %% [markdown]
# Plot the fiducial multiplicative correction at the reference redshift.

# %%
k_kms = np.linspace(0.001, 0.02, 1000)
contamination = hcd_model.get_contamination(z=3, k_kms=k_kms)
fig, ax = plt.subplots()
ax.plot(k_kms, contamination)
ax.axhline(1, color="k", linestyle=":")
ax.set(xlabel=r"$k_\parallel$ [s/km]", ylabel="HCD contamination", ylim=(0.9, 1.2))

# %% [markdown]
# Now compare this constant-amplitude model with a two-coefficient example.
# The second coefficient introduces redshift evolution in the amplitude.

# %%
hcd_model_test = HCD_Model_McDonald2005(ln_A_damp_coeff=[1, -0.1])

# %% [markdown]
# Compare the two models at three redshifts.  The redshift dependence is most
# easily seen by keeping the same vertical range in each panel.

# %%
k_kms = np.linspace(0.001, 0.03, 1000)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for redshift, ax in zip([2, 3, 4], axes):
    fiducial_contamination = hcd_model.get_contamination(redshift, k_kms)
    test_contamination = hcd_model_test.get_contamination(redshift, k_kms)
    ax.plot(k_kms, fiducial_contamination, label="one coefficient")
    ax.plot(k_kms, test_contamination, label="two coefficients")
    ax.axhline(1, color="k", linestyle=":")
    ax.set_title(f"z = {redshift}")
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
axes[0].set_ylabel("HCD contamination")
axes[0].legend()
fig.tight_layout()

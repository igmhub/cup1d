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
# # User interface: the compressed likelihood
#
# This notebook illustrates how to use a compressed Ly-alpha likelihood.  It is
# intended for users who want to compare a cosmological model with a published
# constraint without re-running the full P1D analysis.
#
# A compressed likelihood has already marginalized over the astrophysical and
# instrumental nuisance parameters.  It retains only a small set of parameters
# describing the linear matter power, so it no longer contains redshift bins,
# band powers, mean-flux parameters, thermal-history parameters, or
# reionization parameters.

# %% [markdown]
# Summary:
# - Given an input cosmological model, it computes the parameters describing
#   the linear power spectrum (linP).
# - Given a set of linP parameters, it evaluates a precomputed likelihood.
#   Here these are Gaussian approximations in ($\Delta_\star^2$, $n_\star$): the
#   amplitude and slope of the linear power at $z_\star=3$ and
#   $k_\star=0.009\,\mathrm{s/km}$.

# %% jupyter={"outputs_hidden": false}

import matplotlib.pyplot as plt
import numpy as np

from cup1d.likelihood import marginal
from lace.cosmo import cosmology

# %% [markdown]
# ## Plot marginalised likelihoods
#
# Each constraint is experiment-specific and has been marginalized over its
# nuisance parameters.  The functions below provide Gaussian approximations to
# published constraints; they are useful for quick comparisons, but they do
# not replace the full likelihood of the corresponding analysis.

# %% [markdown]
# ### Compare published constraints, including DESI DR1
#
# McDonald et al. (2005) is only approximately Gaussian.  The DESI DR1
# Gaussian is defined by $\Delta_\star^2=0.379\pm0.032$,
# $n_\star=-2.309\pm0.019$, and their correlation $r=-0.1738$, all at the
# common pivot.

# %%
# Define a common grid and evaluate every compressed likelihood on it.
z_star = 3.0
k_star_kms = 0.009
n_star_grid, delta2_star_grid = np.mgrid[-2.5:-2.1:200j, 0.2:0.7:200j]

likelihoods = {
    "McDonald 2005": ("tab:green", marginal.gaussian_chi2_McDonald2005),
    "Palanque-Delabrouille 2015": (
        "tab:red",
        marginal.gaussian_chi2_PalanqueDelabrouille2015,
    ),
    "Chabanier 2019": ("tab:blue", marginal.gaussian_chi2_Chabanier2019),
    "DESI DR1": ("tab:purple", marginal.gaussian_chi2_DESI_DR1),
}
likelihood_results = {
    name: (color, likelihood(n_star_grid, delta2_star_grid))
    for name, (color, likelihood) in likelihoods.items()
}

# %% jupyter={"outputs_hidden": false}
# Plot the 68%, 95%, and 99.7% contours.  Each function returns both its
# metadata and the grid of delta-chi-squared values under the ``chi2`` key.
thresholds = [2.30, 6.17, 11.8]
fig, ax = plt.subplots(figsize=(10, 8))
for name, (color, result) in likelihood_results.items():
    ax.contour(
        n_star_grid, delta2_star_grid, result["chi2"], levels=thresholds, colors=color
    )
    ax.plot([], [], color=color, label=name)
ax.set_ylim(delta2_star_grid.min(), delta2_star_grid.max())
ax.grid()
ax.legend(loc="upper left")
ax.set_title(r"Linear-power constraints at ($z_\star=3$, $k_\star=0.009$ s/km)")
ax.set_xlabel(r"$n_\star$")
ax.set_ylabel(r"$\Delta_\star^2$")

# %% [markdown]
# ### Palanque-Delabrouille et al. (2015) view
#
# This zoomed view shows the Gaussian approximation used for that published
# result.  It is not intended as an exact reproduction of the original figure.

# %% jupyter={"outputs_hidden": false}
fig, ax = plt.subplots(figsize=(8, 8))
result = likelihood_results["Palanque-Delabrouille 2015"][1]
ax.contour(delta2_star_grid, n_star_grid, result["chi2"], levels=thresholds, colors="tab:red")
ax.plot([], [], color="tab:red", label="Palanque-Delabrouille et al. 2015")
ax.set(xlim=(0.14, 0.44), ylim=(-2.41, -2.29))
ax.grid()
ax.legend(loc="upper left")
ax.set_title(r"Linear-power constraints at ($z_\star=3$, $k_\star=0.009$ s/km)")
ax.set_xlabel(r"$\Delta_\star^2$")
ax.set_ylabel(r"$n_\star$")

# %% [markdown]
# ### Chabanier et al. (2019) view
#
# This is the corresponding zoomed view for the Chabanier et al. constraint.

# %% jupyter={"outputs_hidden": false}
fig, ax = plt.subplots(figsize=(10, 6))
result = likelihood_results["Chabanier 2019"][1]
ax.contour(delta2_star_grid, n_star_grid, result["chi2"], levels=thresholds[:2], colors="tab:blue")
ax.plot([], [], color="tab:blue", label="Chabanier et al. 2019")
ax.set(xlim=(0.24, 0.42), ylim=(-2.36, -2.3))
ax.grid()
ax.legend(loc="upper left")
ax.set_title(r"Linear-power constraints at ($z_\star=3$, $k_\star=0.009$ s/km)")
ax.set_xlabel(r"$\Delta_\star^2$")
ax.set_ylabel(r"$n_\star$")

# %% [markdown]
# ## Compute a prediction from a Planck18 cosmology
#
# The following cells use LaCE to calculate the same two linear-power
# parameters for a fiducial Planck18 cosmology, at precisely the pivot used by
# the compressed likelihoods above.

# %% jupyter={"outputs_hidden": false}
# Instantiate the current LaCE Planck18 cosmology interface.
fiducial_cosmology = cosmology.Cosmology(cosmo_label="Planck18")

# %% [markdown]
# ### Compute linear-power parameters at $z_\star=3$ and $k_\star=0.009$ s/km

# %% jupyter={"outputs_hidden": false}
# Ask LaCE directly for the local linear-power parameters at the common pivot.
planck_star_params = fiducial_cosmology.get_linP_kms_params(z_star, k_star_kms)
print("Ly-alpha parameters for the Planck18 cosmology", planck_star_params)

# %% jupyter={"outputs_hidden": false}
# Compare the Planck18 prediction with the two recent compressed constraints.
fig, ax = plt.subplots(figsize=(10, 8))
for name in ("Chabanier 2019", "DESI DR1"):
    color, result = likelihood_results[name]
    ax.contour(delta2_star_grid, n_star_grid, result["chi2"], levels=thresholds, colors=color)
    ax.plot([], [], color=color, label=name)
ax.plot(
    planck_star_params["Delta2_star"],
    planck_star_params["n_star"],
    "o",
    color="tab:red",
    label="Planck18",
)
ax.set(xlim=(0.2, 0.45), ylim=(-2.4, -2.28))
ax.grid()
ax.legend(loc="upper left")
ax.set_title(r"Linear-power constraints at ($z_\star=3$, $k_\star=0.009$ s/km)")
ax.set_xlabel(r"$\Delta_\star^2$")
ax.set_ylabel(r"$n_\star$")

# %%

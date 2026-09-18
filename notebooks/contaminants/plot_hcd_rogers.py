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
# # Rogers et al. (2018) HCD-contamination model
#
# This notebook demonstrates the HCD-contamination model based on Rogers et
# al. (2018).  Unlike the McDonald et al. model, it separates the contribution
# from four HCD populations, whose distinct damping scales produce different
# scale dependence in the one-dimensional flux power spectrum.

# %% [markdown]
# The multiplicative correction is
#
# $$C_\mathrm{HCD}(k,z)=1+c_\mathrm{HCD}(z)+
# \sum_{i=1}^4 f_i^\mathrm{HCD}(z)D_i(k,z),$$
#
# where the four damping functions $D_i$ represent Lyman-limit systems,
# sub-DLAs, small DLAs, and large DLAs.  The amplitude coefficients
# $f_i^\mathrm{HCD}$ are nuisance parameters.  This notebook activates one
# population at a time to show their characteristic shapes at $z=3$.

# %% [markdown]
# Import the reorganized contamination model and the plotting dependencies.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from cup1d.models.contaminants.HCD.hcd_model_rogers_class import HCD_Model_Rogers

# %% [markdown]
# ## Contributions from the four HCD populations
#
# The values below are illustrative amplitudes, not a joint best fit.  All
# other populations are effectively disabled by assigning them a very small
# amplitude.  The resulting curves reproduce the qualitative comparison in
# Figure 6 of Rogers et al. (2018).

# %%
k_kms = np.logspace(-3, np.log10(0.04), 100)
population_labels = ["LLS", "sub-DLAs", "small DLAs", "large DLAs"]
line_styles = ["-", ":", "-.", "--"]
amplitudes = [0.1, 3e-2, 1e-2, 1e-2]
redshift = np.array([3.0])

fig, ax = plt.subplots(figsize=(8, 6))
for index, (label, line_style, amplitude) in enumerate(
    zip(population_labels, line_styles, amplitudes), start=1
):
    coefficients = {
        "HCD_damp1": [0, -11.5],
        "HCD_damp2": [0, -11.5],
        "HCD_damp3": [0, -11.5],
        "HCD_damp4": [0, -11.5],
        "HCD_const": [0, 0],
    }
    coefficients[f"HCD_damp{index}"] = [0, np.log(amplitude)]

    hcd_model = HCD_Model_Rogers(coeffs=coefficients)
    contamination = hcd_model.get_contamination(redshift, [k_kms])
    ax.plot(
        k_kms,
        contamination,
        label=rf"$f^\mathrm{{HCD}}_\mathrm{{{label}}}={amplitude:g}$",
        linestyle=line_style,
        linewidth=2,
    )

ax.axhline(1, color="k", linestyle=":")
ax.set_xscale("log")
ax.set_xlabel(r"$k_\parallel$ [s/km]")
ax.set_ylabel("HCD contamination")
ax.legend()
fig.tight_layout()

# %%

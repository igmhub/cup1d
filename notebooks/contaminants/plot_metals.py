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
# # Metal-contamination models
#
# Absorption by metal transitions produces oscillatory structure in the
# one-dimensional Ly-alpha forest power spectrum. The oscillation frequency is
# set by the velocity separation of the two transitions, while its amplitude
# and redshift evolution are nuisance parameters in the likelihood.

# %% [markdown]
# cup1d has two complementary silicon-contamination models:
#
# - `SiMult` describes terms that multiply the Ly-alpha power, including
#   Ly-alpha--SiIII and Ly-alpha--SiII correlations.
# - `SiAdd` describes the additive SiII--SiII contribution.
#
# This notebook displays representative components at $z=3$. The coefficient
# values are illustrative and are chosen only to make the scale dependence
# visible; they are not a recommended fiducial model or fit.

# %% [markdown]
# Import the current contaminant classes and the velocity-separation helper.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from cup1d.models.contaminants.metals.si_add import SiAdd, vel_diff
from cup1d.models.contaminants.metals.si_mult import SiMult

# %% [markdown]
# The expected oscillation spacing is approximately $2\pi/\Delta v$. The
# following table lists several relevant transition pairs and provides a quick
# check of the physical scales represented by the models.

# %%
transitions = {
    "Ly-alpha--SiIII": (1215.67, 1206.51),
    "Ly-alpha--SiIIb": (1215.67, 1193.28),
    "Ly-alpha--SiIIa": (1215.67, 1190.42),
    "SiIIa--SiIIb": (1190.42, 1193.28),
    "SiIIa--SiIII": (1190.42, 1206.51),
    "SiIIb--SiIII": (1193.28, 1206.51),
}
for label, wavelengths in transitions.items():
    delta_v = vel_diff(*wavelengths)
    print(
        f"{label:18s}: Δv = {delta_v:7.1f} km/s, "
        f"2π/Δv = {2 * np.pi / delta_v:.5f} s/km"
    )

# %% [markdown]
# Construct the multiplicative and additive models. The `remove` dictionaries
# selectively retain a single family of terms, which makes the origin of each
# oscillatory feature easier to see.

# %%
k_kms = np.linspace(1e-3, 0.04, 1000)
redshift = np.array([3.0])
mean_flux = np.array([0.75])

multiplicative_coefficients = {
    "f_Lya_SiIII": [0, -4.17],
    "s_Lya_SiIII": [0, 4.90],
    "f_Lya_SiII": [0, -3.66],
    "s_Lya_SiII": [0, 5.65],
    "f_SiIIa_SiIII": [0, 0.84],
    "f_SiIIb_SiIII": [0, 0.59],
}
additive_coefficients = {
    "f_SiIIa_SiIIb": [0, 0.40],
    "s_SiIIa_SiIIb": [0, 4.43],
}
silicon_multiplicative = SiMult(coeffs=multiplicative_coefficients)
silicon_additive = SiAdd(coeffs=additive_coefficients)

components = [
    (
        "Ly-alpha--SiIII",
        silicon_multiplicative,
        {
            "SiIII_Lya": 1,
            "SiIIa_Lya": 0,
            "SiIIb_Lya": 0,
            "SiIII_SiIIa": 0,
            "SiIII_SiIIb": 0,
        },
        1,
    ),
    (
        "Ly-alpha--SiII",
        silicon_multiplicative,
        {
            "SiIII_Lya": 0,
            "SiIIa_Lya": 1,
            "SiIIb_Lya": 1,
            "SiIII_SiIIa": 0,
            "SiIII_SiIIb": 0,
        },
        1,
    ),
    (
        "SiIII--SiII",
        silicon_multiplicative,
        {
            "SiIII_Lya": 0,
            "SiIIa_Lya": 0,
            "SiIIb_Lya": 0,
            "SiIII_SiIIa": 1,
            "SiIII_SiIIb": 1,
        },
        1,
    ),
    ("SiII--SiII", silicon_additive, {}, 0),
]

# %% [markdown]
# Plot each component separately. Multiplicative corrections are shown around
# one, whereas the additive contribution is shown around zero.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, (label, model, remove, reference_level) in zip(axes.flat, components):
    contamination = model.get_contamination(
        redshift, [k_kms], mF=mean_flux, remove=remove
    )[0]
    ax.plot(k_kms, contamination, linewidth=1.5)
    ax.axhline(reference_level, color="k", linestyle=":")
    ax.set_title(label)
    ax.set_ylabel("metal contamination")
for ax in axes[-1]:
    ax.set_xlabel(r"$k_\parallel$ [s/km]")
fig.tight_layout()

# %%

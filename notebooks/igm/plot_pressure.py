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
# # Pressure-smoothing parameterization
#
# Pressure smoothing suppresses small-scale structure in the intergalactic
# medium. In cup1d it is characterized by a filtering wavenumber,
# $k_\mathrm{F}$, expressed here in velocity units (s/km). Its reciprocal,
# $\lambda_\mathrm{F}=1/k_\mathrm{F}$, is the corresponding velocity-space
# filtering length.

# %% [markdown]
# cup1d starts from a fiducial pressure-smoothing history and rescales it as
#
# $$k_\mathrm{F}(z)=k_\mathrm{F}^\mathrm{fid}(z)
# f_\mathrm{F}(z).$$
#
# The scaling factors $f_\mathrm{F}$ are specified at redshift nodes and
# linearly interpolated between them. This is the same node-based approach used
# for the mean-flux model. In a production analysis the fiducial history comes
# from the selected simulation; the smooth curve below is only for illustration.

# %% [markdown]
# Import the current pressure model.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from cup1d.models.igm.pressure_class import Pressure

# %% [markdown]
# Define a smooth illustrative fiducial history and a helper that constructs a
# `Pressure` model with user-selected node values.

# %%
def fiducial_kF_kms(redshift):
    """Return a smooth illustrative filtering-wavenumber history in s/km."""
    redshift = np.asarray(redshift)
    return 0.015 * ((1 + redshift) / 4) ** 0.5


def make_pressure_model(node_values, redshift_nodes, fiducial_redshift):
    """Build a pressure model around the illustrative fiducial history."""
    fiducial_igm = {
        "kF_kms_z": fiducial_redshift,
        "kF_kms": fiducial_kF_kms(fiducial_redshift),
    }
    properties = {
        "kF_kms_otype": "const",
        "kF_kms_ztype": "interp_spl",
        "kF_kms_znodes": redshift_nodes,
    }
    return Pressure(
        coeffs={"kF_kms": node_values},
        prop_coeffs=properties,
        fid_igm=fiducial_igm,
        fid_vals={},
    )


# %% [markdown]
# Construct a fiducial model and a three-node variation. The variation changes
# $k_\mathrm{F}$ by +20% at the low-redshift node and -10% at the high-redshift
# node, while preserving the fiducial value at the middle node.

# %%
redshift = np.linspace(2.0, 4.5, 100)
redshift_nodes = np.array([2.0, 3.25, 4.5])
fiducial_model = make_pressure_model(
    np.ones_like(redshift_nodes), redshift_nodes, redshift
)
perturbed_model = make_pressure_model(
    np.array([1.2, 1.0, 0.9]), redshift_nodes, redshift
)

# %% [markdown]
# Compare the two histories in terms of both the filtering wavenumber and its
# inverse length. Vertical lines mark the locations of the nuisance nodes.

# %%
fiducial_kF = fiducial_model.get_kF_kms(redshift)
perturbed_kF = perturbed_model.get_kF_kms(redshift)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(redshift, fiducial_kF, label="fiducial history")
axes[0].plot(redshift, perturbed_kF, label="three-node variation")
axes[0].set(
    xlabel="z",
    ylabel=r"$k_\mathrm{F}$ [s/km]",
    title="Filtering wavenumber",
)

axes[1].plot(redshift, 1 / fiducial_kF, label="fiducial history")
axes[1].plot(redshift, 1 / perturbed_kF, label="three-node variation")
axes[1].set(
    xlabel="z",
    ylabel=r"$\lambda_\mathrm{F}$ [km/s]",
    title="Filtering length",
)
for ax in axes:
    for node in redshift_nodes:
        ax.axvline(node, color="0.7", linestyle=":", zorder=0)
    ax.legend()
fig.tight_layout()

# %% [markdown]
# The last panel shows directly the node-interpolated scaling factor applied to
# the fiducial history.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(redshift, perturbed_kF / fiducial_kF)
ax.axhline(1, color="k", linestyle=":")
for node in redshift_nodes:
    ax.axvline(node, color="0.7", linestyle=":", zorder=0)
ax.set(xlabel="z", ylabel=r"$k_\mathrm{F}/k_\mathrm{F}^\mathrm{fid}$")
fig.tight_layout()

# %%

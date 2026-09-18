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
# # Temperature--density relation
#
# The intergalactic-medium temperature is commonly described by the power-law
# relation
#
# $$T(\rho)=T_0\left(\frac{\rho}{\bar\rho}\right)^{\gamma-1},$$
#
# where $T_0$ is the temperature at mean density and $\gamma$ describes the
# density dependence. cup1d models thermal broadening through
# $\sigma_\mathrm{T}$ in km/s, which is equivalent to $T_0$, and models the
# redshift evolution of both $\sigma_\mathrm{T}$ and $\gamma$.

# %% [markdown]
# As with the mean-flux and pressure models, cup1d rescales fiducial histories
# by factors defined at redshift nodes and linearly interpolated between them.
# The fiducial thermal history below is taken from the tabulated Gaikwad et al.
# (2021) measurements. In a production run it is normally supplied by the
# configured simulation history instead.

# %% [markdown]
# Import the current thermal model and the Gaikwad et al. measurements.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from lace.cosmo.thermal_broadening import thermal_broadening_kms

from cup1d.models.igm.mean_flux_measurements import get_mean_flux_measurements
from cup1d.models.igm.thermal_class import Thermal

# %% [markdown]
# Retrieve the measurements. The tabulated $T_0$ values and their uncertainties
# are in K; the model converts them to the thermal-broadening scale internally.

# %%
gaikwad2021, _ = get_mean_flux_measurements()

# %% [markdown]
# Build a thermal model around the Gaikwad et al. fiducial history. The scaling
# factors are constant at each node before being interpolated in redshift.

# %%
def make_thermal_model(sigT_node_values, gamma_node_values, redshift_nodes):
    """Build a node-based cup1d thermal model around the Gaikwad history."""
    fiducial_igm = {
        "sigT_kms_z": gaikwad2021["z"],
        "sigT_kms": thermal_broadening_kms(gaikwad2021["T0"]),
        "gamma_z": gaikwad2021["z"],
        "gamma": gaikwad2021["gamma"],
    }
    properties = {
        "sigT_kms_otype": "const",
        "sigT_kms_ztype": "interp_spl",
        "sigT_kms_znodes": redshift_nodes,
        "gamma_otype": "const",
        "gamma_ztype": "interp_spl",
        "gamma_znodes": redshift_nodes,
    }
    return Thermal(
        coeffs={"sigT_kms": sigT_node_values, "gamma": gamma_node_values},
        prop_coeffs=properties,
        fid_igm=fiducial_igm,
        fid_vals={},
    )


# %% [markdown]
# Construct the fiducial model and an illustrative three-node variation. Since
# $T_0$ is proportional to $\sigma_\mathrm{T}^2$, a change in the broadening
# scale produces a larger fractional change in the plotted temperature.

# %%
redshift = np.linspace(2.0, 4.5, 100)
redshift_nodes = np.array([2.0, 3.25, 4.5])
fiducial_model = make_thermal_model(
    np.ones_like(redshift_nodes), np.ones_like(redshift_nodes), redshift_nodes
)
perturbed_model = make_thermal_model(
    np.array([1.1, 1.0, 0.95]),
    np.array([0.95, 1.0, 1.05]),
    redshift_nodes,
)

# %% [markdown]
# Compare $T_0$ and $\gamma$ with the Gaikwad et al. (2021) measurements.
# Vertical lines identify the redshift nodes where the thermal nuisance factors
# are defined.

# %%
fiducial_T0 = fiducial_model.get_T0(redshift)
perturbed_T0 = perturbed_model.get_T0(redshift)
fiducial_gamma = fiducial_model.get_gamma(redshift)
perturbed_gamma = perturbed_model.get_gamma(redshift)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(redshift, fiducial_T0, label="fiducial history")
axes[0].plot(redshift, perturbed_T0, label="three-node variation")
axes[0].errorbar(
    gaikwad2021["z"],
    gaikwad2021["T0"],
    yerr=gaikwad2021["T0_err"],
    fmt="o",
    ms=3,
    label="Gaikwad et al. (2021)",
)
axes[0].set(xlabel="z", ylabel=r"$T_0$ [K]", title="Temperature at mean density")

axes[1].plot(redshift, fiducial_gamma, label="fiducial history")
axes[1].plot(redshift, perturbed_gamma, label="three-node variation")
axes[1].errorbar(
    gaikwad2021["z"],
    gaikwad2021["gamma"],
    yerr=gaikwad2021["gamma_err"],
    fmt="o",
    ms=3,
    label="Gaikwad et al. (2021)",
)
axes[1].set(xlabel="z", ylabel=r"$\gamma$", title="Temperature--density slope")

for ax in axes:
    for node in redshift_nodes:
        ax.axvline(node, color="0.7", linestyle=":", zorder=0)
    ax.legend()
fig.tight_layout()

# %% [markdown]
# The final figure shows the node-interpolated factors applied to the thermal
# broadening scale and the temperature--density slope.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(
    redshift,
    perturbed_model.get_sigT_kms(redshift) / fiducial_model.get_sigT_kms(redshift),
)
axes[0].set(
    xlabel="z",
    ylabel=r"$\sigma_\mathrm{T}/\sigma_\mathrm{T}^\mathrm{fid}$",
    title="Thermal-broadening ratio",
)
axes[1].plot(redshift, perturbed_gamma / fiducial_gamma)
axes[1].set(
    xlabel="z",
    ylabel=r"$\gamma/\gamma_\mathrm{fid}$",
    title="Slope ratio",
)
for ax in axes:
    ax.axhline(1, color="k", linestyle=":")
    for node in redshift_nodes:
        ax.axvline(node, color="0.7", linestyle=":", zorder=0)
fig.tight_layout()

# %%

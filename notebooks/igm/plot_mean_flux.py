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
# # Mean-flux parameterization
#
# The mean transmitted flux fraction, $\bar F$, is a key nuisance quantity in
# Ly-alpha forest analyses. It is conventionally expressed through the
# effective optical depth, $\tau_\mathrm{eff}=-\ln\bar F$. This notebook
# demonstrates the flexible mean-flux model used by cup1d.

# %% [markdown]
# cup1d starts from a fiducial history, $\tau_\mathrm{eff}^\mathrm{fid}(z)$,
# and rescales it as
#
# $$\tau_\mathrm{eff}(z)=\tau_\mathrm{eff}^\mathrm{fid}(z)
# \exp\left[\delta_\tau(z)\right].$$
#
# The values of $\delta_\tau$ are specified at redshift nodes and linearly
# interpolated between them. This retains a smooth history while allowing the
# data to change it independently over broad redshift intervals. A power-law
# history is included below solely as an illustrative fiducial curve.

# %% [markdown]
# Import the current mean-flux model. The older `MeanFluxModel` interface has
# been replaced by `MeanFlux` under `cup1d.models.igm`.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from cup1d.models.igm.mean_flux_class import MeanFlux
from cup1d.models.igm.mean_flux_measurements import get_mean_flux_measurements

# %% [markdown]
# We use the power law quoted for Kamble et al. (2019) as a simple fiducial
# history. It is convenient for this demonstration because it is smooth and
# can be evaluated at any redshift; production analyses instead obtain the
# fiducial history from the configured simulation or measurement.

# %%
def tau_eff_kamble2019(redshift):
    """Illustrative effective-optical-depth power law from Kamble et al. (2019)."""
    return 0.0055 * (1 + np.asarray(redshift)) ** 3.18


def make_mean_flux_model(node_values, redshift_nodes, fiducial_redshift):
    """Build a cup1d mean-flux model around the illustrative fiducial history."""
    fiducial_igm = {
        "tau_eff_z": fiducial_redshift,
        "tau_eff": tau_eff_kamble2019(fiducial_redshift),
    }
    properties = {
        "tau_eff_otype": "exp",
        "tau_eff_ztype": "interp_spl",
        "tau_eff_znodes": redshift_nodes,
    }
    return MeanFlux(
        coeffs={"tau_eff": node_values},
        prop_coeffs=properties,
        fid_igm=fiducial_igm,
        fid_vals={},
    )


# %% [markdown]
# cup1d also stores the tabulated mean-flux measurements used for comparison in
# Gaikwad et al. (2021) and Turner et al. (2024).  For the optical-depth panel,
# we use first-order propagation, $\sigma_\tau=\sigma_{\bar F}/\bar F$.

# %%
gaikwad2021, turner2024 = get_mean_flux_measurements()
for measurement in (gaikwad2021, turner2024):
    measurement["tau_eff"] = -np.log(measurement["mF"])
    measurement["tau_eff_err"] = measurement["mF_err"] / measurement["mF"]


# %% [markdown]
# Construct a fiducial model and a three-node perturbation. The perturbation
# raises the optical depth near the lowest node by a factor $e^{0.4}$ while
# leaving the other nodes unchanged.

# %%
redshift = np.linspace(2.0, 4.5, 100)
redshift_nodes = np.array([2.0, 3.25, 4.5])
fiducial_model = make_mean_flux_model(
    np.zeros_like(redshift_nodes), redshift_nodes, redshift
)
perturbed_model = make_mean_flux_model(
    np.array([-0.25, -0.2, 0.0]), redshift_nodes, redshift
)

# %% [markdown]
# Compare the fiducial and perturbed histories in both mean flux and effective
# optical depth. Markers identify the redshift nodes at which the nuisance
# coefficients are defined.

# %%
fiducial_mean_flux = fiducial_model.get_mean_flux(redshift)
perturbed_mean_flux = perturbed_model.get_mean_flux(redshift)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(redshift, fiducial_mean_flux, label="fiducial history")
axes[0].plot(redshift, perturbed_mean_flux, label="three-node perturbation")
axes[0].errorbar(
    gaikwad2021["z"],
    gaikwad2021["mF"],
    yerr=gaikwad2021["mF_err"],
    fmt="o",
    ms=3,
    label="Gaikwad et al. (2021)",
)
axes[0].errorbar(
    turner2024["z"],
    turner2024["mF"],
    yerr=turner2024["mF_err"],
    fmt="o",
    ms=3,
    label="Turner et al. (2024)",
)
axes[0].set(xlabel="z", ylabel=r"$\bar F(z)$", title="Mean transmitted flux")

axes[1].plot(redshift, fiducial_model.get_tau_eff(redshift), label="fiducial history")
axes[1].plot(redshift, perturbed_model.get_tau_eff(redshift), label="three-node perturbation")
axes[1].errorbar(
    gaikwad2021["z"],
    gaikwad2021["tau_eff"],
    yerr=gaikwad2021["tau_eff_err"],
    fmt="o",
    ms=3,
    label="Gaikwad et al. (2021)",
)
axes[1].errorbar(
    turner2024["z"],
    turner2024["tau_eff"],
    yerr=turner2024["tau_eff_err"],
    fmt="o",
    ms=3,
    label="Turner et al. (2024)",
)
axes[1].set(xlabel="z", ylabel=r"$\tau_\mathrm{eff}(z)$", title="Effective optical depth")
for ax in axes:
    for node in redshift_nodes:
        ax.axvline(node, color="0.7", linestyle=":", zorder=0)
    ax.legend()
fig.tight_layout()

# %% [markdown]
# The final plot makes the multiplicative parameterization explicit. The
# optical-depth ratio is $\exp[\delta_\tau(z)]$, whereas the mean-flux ratio
# responds nonlinearly because $\bar F=\exp(-\tau_\mathrm{eff})$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(redshift, perturbed_mean_flux / fiducial_mean_flux)
axes[0].set(xlabel="z", ylabel=r"$\bar F / \bar F_\mathrm{fid}$", title="Mean-flux ratio")

axes[1].plot(
    redshift,
    perturbed_model.get_tau_eff(redshift) / fiducial_model.get_tau_eff(redshift),
)
axes[1].set(
    xlabel="z",
    ylabel=r"$\tau_\mathrm{eff}/\tau_\mathrm{eff}^\mathrm{fid}$",
    title="Optical-depth ratio",
)
for ax in axes:
    ax.axhline(1, color="k", linestyle=":")
    for node in redshift_nodes:
        ax.axvline(node, color="0.7", linestyle=":", zorder=0)
fig.tight_layout()

# %%

# %%

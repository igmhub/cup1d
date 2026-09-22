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
# # Optimising the star-parameter pivot
#
# This notebook studies how the compressed linear-power parameters
# $\Delta^2_\star$, $n_\star$, and $\alpha_\star$ vary with redshift
# $z_\star$ and velocity-space pivot $k_\star$. Posterior samples vary $A_s$
# and $n_s$ while holding the background and $n_\mathrm{run}=0$ fixed.
#
# The calculation uses LaCE's ``Cosmology`` and ``RescaledCosmology`` classes.
# ``_star`` parameters are dimensionless and use a pivot in s/km; ``_p``
# parameters instead use a pivot in 1/Mpc.

# %% [markdown]
# ## Imports and posterior input
#
# The posterior sample comes from the DR1 fit used in the original notebook.
# It supplies the distribution of $A_s$ and $n_s$ whose propagation into star
# parameters is studied below. No cup1d Python module is imported.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lace.cosmo.cosmology import Cosmology
from lace.cosmo.rescale_cosmology import RescaledCosmology

posterior_directory = Path(
    "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
    "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7"
)
chain_path = posterior_directory / "chain.npy"
results_path = posterior_directory / "fitter_results.npy"

# %% [markdown]
# ## Draw primordial-power samples
#
# The saved chain stores parameters normalized to their sampling bounds. We
# select a reproducible subset and convert its first two free parameters,
# ``As`` and ``ns``, back to physical values. A subset keeps the notebook fast
# while retaining the posterior shape relevant for this pivot study.

# %%
n_posterior_samples = 1_000
rng = np.random.default_rng(0)

fit_results = np.load(results_path, allow_pickle=True).item()
free_parameter_names = fit_results["like"]["free_param_names"]
assert free_parameter_names[:2] == ["As", "ns"]

normalized_chain = np.load(chain_path, mmap_mode="r")
sample_indices = rng.choice(
    np.prod(normalized_chain.shape[:2]), size=n_posterior_samples, replace=False
)
walkers, steps = np.unravel_index(sample_indices, normalized_chain.shape[:2])
normalized_As_ns = normalized_chain[walkers, steps, :2]

As_ns_samples = np.empty_like(normalized_As_ns)
for column, name in enumerate(free_parameter_names[:2]):
    parameter = fit_results["like"]["free_params"][name]
    As_ns_samples[:, column] = (
        normalized_As_ns[:, column]
        * (parameter["max_value"] - parameter["min_value"])
        + parameter["min_value"]
    )

print("Posterior samples:", As_ns_samples.shape)
print("As range:", np.min(As_ns_samples[:, 0]), np.max(As_ns_samples[:, 0]))
print("ns range:", np.min(As_ns_samples[:, 1]), np.max(As_ns_samples[:, 1]))

# %% [markdown]
# ## Fiducial cosmology and grids
#
# These explicit values are the fiducial cosmology used in the original
# calculation. ``Cosmology`` evaluates it with CAMB once and caches the
# resulting linear spectrum. ``RescaledCosmology`` then changes only $A_s$ and
# $n_s$ for each posterior sample, keeping the background fixed.
#
# The active grids are retained from the saved notebook: $z_\star=3$ and
# $k_\star=0.02\ldots0.03\,{\rm s/km}$. The standard cup1d choice,
# $z_\star=3$ and $k_\star=0.009\,{\rm s/km}$, is shown where it lies inside
# a plot range; the present pivot scan is deliberately at higher $k_\star$.

# %%
fiducial_parameters = {
    "H0": 67.66,
    "mnu": 0.0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0.0,
    "As": 2.105e-9,
    "ns": 0.9665,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
fiducial_cosmology = Cosmology(cosmo_params_dict=fiducial_parameters)

z_star_grid = np.array([3.0])
k_star_grid_kms = np.linspace(0.005, 0.025, 10)
standard_z_star = 3.0
standard_k_star_kms = 0.009

# %% [markdown]
# ## Compute star parameters on the grid
#
# ``get_linP_kms_params`` converts each velocity-space pivot internally to a
# comoving pivot using $H(z)/(1+z)$, evaluates the linear spectrum, and fits
# the three compressed parameters. This cell delegates that work to LaCE rather
# than reproducing the old analytic rescaling formula.
#
# For every posterior sample, the same cached fiducial CAMB result is reused.
# Since $n_\mathrm{run}$ is fixed, $\alpha_\star$ should be constant across
# these $A_s$ and $n_s$ samples; it is retained as a consistency check.

# %%
parameter_names = ("Delta2_star", "n_star", "alpha_star")
star_parameters = np.empty(
    (
        n_posterior_samples,
        len(z_star_grid),
        len(k_star_grid_kms),
        len(parameter_names),
    )
)

for sample_index, (As, ns) in enumerate(As_ns_samples):
    rescaled_cosmology = RescaledCosmology(
        fiducial_cosmology,
        new_params_dict={"As": As, "ns": ns, "nrun": 0.0},
    )
    for redshift_index, z_star in enumerate(z_star_grid):
        for pivot_index, k_star_kms in enumerate(k_star_grid_kms):
            parameters = rescaled_cosmology.get_linP_kms_params(
                z=z_star, kp_kms=k_star_kms
            )
            star_parameters[sample_index, redshift_index, pivot_index] = [
                parameters[name] for name in parameter_names
            ]

assert star_parameters.shape == (
    n_posterior_samples,
    len(z_star_grid),
    len(k_star_grid_kms),
    len(parameter_names),
)
assert np.all(np.isfinite(star_parameters))
print("Star-parameter array shape:", star_parameters.shape)

# %% [markdown]
# ## Posterior correlation and uncertainty versus $k_\star$
#
# At each $z_\star$ and $k_\star$, this computes the correlation between
# $\Delta^2_\star$ and $n_\star$, their fractional 68% posterior widths, and
# the area of their correlation ellipse. Smaller area gives a less correlated,
# more compact two-parameter description for this posterior. The third star
# parameter is reported separately because it does not vary when
# $n_\mathrm{run}$ is held fixed.

# %%
correlation = np.empty((len(z_star_grid), len(k_star_grid_kms)))
relative_width = np.empty((len(z_star_grid), len(k_star_grid_kms), 2))
ellipse_area = np.empty((len(z_star_grid), len(k_star_grid_kms)))
alpha_star_width = np.empty((len(z_star_grid), len(k_star_grid_kms)))

for redshift_index, z_star in enumerate(z_star_grid):
    for pivot_index, k_star_kms in enumerate(k_star_grid_kms):
        samples = star_parameters[:, redshift_index, pivot_index]
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        widths = 0.5 * (percentiles[2] - percentiles[0])
        relative_width[redshift_index, pivot_index] = (
            widths[:2] / np.abs(percentiles[1, :2])
        )
        alpha_star_width[redshift_index, pivot_index] = widths[2]
        correlation[redshift_index, pivot_index] = np.corrcoef(
            samples[:, 0], samples[:, 1]
        )[0, 1]
        ellipse_area[redshift_index, pivot_index] = (
            np.pi
            * relative_width[redshift_index, pivot_index, 0]
            * relative_width[redshift_index, pivot_index, 1]
            * np.sqrt(1.0 - correlation[redshift_index, pivot_index] ** 2)
        )

fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
for redshift_index, z_star in enumerate(z_star_grid):
    label = rf"$z_\star={z_star:g}$"
    axes[0].plot(k_star_grid_kms, correlation[redshift_index], ".-", label=label)
    axes[1].plot(
        k_star_grid_kms,
        relative_width[redshift_index, :, 0],
        ".-",
        label=rf"$\Delta^2_\star$, {label}",
    )
    axes[1].plot(
        k_star_grid_kms,
        relative_width[redshift_index, :, 1],
        "--",
        label=rf"$n_\star$, {label}",
    )
    axes[2].plot(k_star_grid_kms, ellipse_area[redshift_index], ".-", label=label)

for axis in axes:
    if axis.get_xlim()[0] <= standard_k_star_kms <= axis.get_xlim()[1]:
        axis.axvline(
            standard_k_star_kms, color="black", ls=":", label=r"standard $k_\star$"
        )
axes[0].axhline(0.0, color="black", lw=0.8)
axes[0].set_ylabel(r"$\rho(\Delta^2_\star,n_\star)$")
axes[1].set_ylabel("fractional 68% width")
axes[2].set_ylabel("ellipse area")
axes[2].set_xlabel(r"$k_\star\ [{\rm s/km}]$")
axes[0].legend()
axes[1].legend()
axes[2].legend()
fig.tight_layout()

# %% [markdown]
# ## Interpret the scan
#
# The table identifies the pivot that minimizes the two-parameter ellipse area
# at each redshift. It also checks the expected zero posterior width of
# $\alpha_\star$, because $n_\mathrm{run}=0$ for all posterior points. To
# study $\alpha_\star$ uncertainty, extend the input posterior sample to vary
# $n_\mathrm{run}$.

# %%
for redshift_index, z_star in enumerate(z_star_grid):
    best_index = np.argmin(ellipse_area[redshift_index])
    print(
        f"z_star={z_star:g}: best k_star={k_star_grid_kms[best_index]:.5f} s/km, "
        f"correlation={correlation[redshift_index, best_index]:.4f}, "
        f"ellipse area={ellipse_area[redshift_index, best_index]:.5f}"
    )
    print(
        "max alpha_star 68% width:",
        f"{np.max(alpha_star_width[redshift_index]):.3e}",
    )

# %% [markdown]
# ## Minimise the parameter correlation
#
# The ellipse-area criterion also includes the fractional parameter widths. If
# the only goal is to make $\Delta^2_\star$ and $n_\star$ as uncorrelated as
# possible, select the pivot that minimizes the absolute correlation instead.

# %%
for redshift_index, z_star in enumerate(z_star_grid):
    best_index = np.argmin(np.abs(correlation[redshift_index]))
    print(
        f"z_star={z_star:g}: least-correlated k_star="
        f"{k_star_grid_kms[best_index]:.5f} s/km, "
        f"correlation={correlation[redshift_index, best_index]:.4f}"
    )

# %%

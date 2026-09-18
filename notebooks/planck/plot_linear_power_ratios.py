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
# # Linear-power ratios: Planck and DESI P1D
#
# This notebook creates Figure 22: the ratio of linear matter-power spectra
# for several Planck cosmological extensions to the Planck LCDM reference, with
# the DESI P1D amplitude-and-slope constraint at the common pivot.

# %% [markdown]
# The plotter loads Planck chains, DESI samples, blinding offsets, and cached
# power spectra from the paths below. Missing ``P_kms_*.npy`` cache files are
# computed from the corresponding CMB chain and stored automatically.

# %%
from pathlib import Path

from cup1d.postprocessing.power_ratios import PowerRatioPlotter
from cup1d.utils.utils import get_path_repo

repository_path = Path(get_path_repo("cup1d"))
planck_root_dir = repository_path / "data" / "planck_linP_chains"
desi_chain_directory = (
    repository_path.parent
    / "data"
    / "out_DESI_DR1"
    / "DESIY1_QMLE3"
    / "global_opt"
    / "CH24_mpgcen_gpr"
    / "chain_7"
)
blinding_path = repository_path / "notebooks" / "tutorials" / "blinding.npy"
power_file_paths = [
    repository_path / "notebooks" / "planck" / "figs" / f"P_kms_{index}.npy"
    for index in range(5)
]

chain_specs = [
    {"model": "base", "data": "plikHM_TTTEEE_lowl_lowE_linP", "label": r"$\mathit{Planck}$ T&E: $\Lambda$CDM"},
    {"model": "base_mnu", "data": "plikHM_TTTEEE_lowl_lowE_linP", "label": r"$\mathit{Planck}$ T&E: $\sum m_\nu$"},
    {"model": "base_nnu", "data": "plikHM_TTTEEE_lowl_lowE_linP", "label": r"$\mathit{Planck}$ T&E: $N_\mathrm{eff}$"},
    {"model": "base_nrun", "data": "plikHM_TTTEEE_lowl_lowE_linP", "label": r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}$"},
    {"model": "base_nrun_nrunrun", "data": "plikHM_TTTEEE_lowl_lowE_linP", "label": r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}, \,\beta_\mathrm{s}$"},
]

# %% [markdown]
# Load the inputs and create the figure. ``figure_data`` contains the median
# and 68% bands of every plotted constraint.

# %%
power_plotter = PowerRatioPlotter(pivot_kms=0.009, random_seed=0).load_data(
    chain_specs=chain_specs,
    desi_blobs_path=desi_chain_directory / "blobs.npy",
    blinding_path=blinding_path,
    power_file_paths=power_file_paths,
    planck_root_dir=planck_root_dir,
)
fig, axes, figure_data = power_plotter.plot()

# %% [markdown]
# Optionally save the figure or its publication-data dictionary.

# %%
# fig.savefig("figs/Plin_extra.pdf", bbox_inches="tight")
# PowerRatioPlotter.save_data_to_zenodo(figure_data)

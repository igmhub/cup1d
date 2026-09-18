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
# # Linear-power amplitude--slope contours
#
# This notebook creates the cumulative DESI P1D and Planck contour figures in
# the $(\Delta_\star^2,n_\star)$ plane.

# %% [markdown]
# The plotter loads the six Planck chains and the DESI contour polygons itself.

# %%
from pathlib import Path

from cup1d.postprocessing.star_contours import StarContourPlotter
from cup1d.utils.utils import get_path_repo

repository_path = Path(get_path_repo("cup1d"))
planck_root_dir = repository_path / "data" / "planck_linP_chains"
desi_contours_path = (
    repository_path.parent
    / "data"
    / "out_DESI_DR1"
    / "DESIY1_QMLE3"
    / "global_opt"
    / "CH24_mpgcen_gpr"
    / "chain_7"
    / "line_sigmas.npy"
)

standard_planck_data = "plikHM_TTTEEE_lowl_lowE_linP"
standard_models = [
    ("base", r"$\mathit{Planck}$ T&E: $\Lambda$CDM"),
    ("base_mnu", r"$\mathit{Planck}$ T&E: $\sum m_\nu$"),
    ("base_nnu", r"$\mathit{Planck}$ T&E: $N_\mathrm{eff}$"),
    ("base_nrun", r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}$"),
    ("base_nrun_nrunrun", r"$\mathit{Planck}$ T&E: $\alpha_\mathrm{s}, \,\beta_\mathrm{s}$"),
]
chain_specs = [
    {"model": model, "data": standard_planck_data, "label": label}
    for model, label in standard_models
]
chain_specs.append(
    {
        "model": "base_w_wa",
        "data": "plikHM_TTTEEE_lowl_lowE_BAO_linP",
        "label": r"$\mathit{Planck}$ T&E + BAO:" + "\n" + r"$\omega_0\omega_a$CDM",
    }
)

# %% [markdown]
# Load the inputs and create the progressive contour sequence.

# %%
star_contour_plotter = StarContourPlotter().load_data(
    chain_specs=chain_specs,
    desi_contours_path=desi_contours_path,
    planck_root_dir=planck_root_dir,
)
contour_figures = star_contour_plotter.plot_progressive()

# %% [markdown]
# Optionally write the figures and the Figure-21 publication-data dictionary.

# %%
# contour_figures = star_contour_plotter.plot_progressive(save_directory="figs")
# star_contour_plotter.save_data_to_zenodo()

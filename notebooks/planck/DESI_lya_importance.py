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
# # Importance sampling of CMB chains with DESI Ly$\alpha$ P1D
#
# This notebook compares CMB constraints before and after importance sampling
# with the DESI DR1 compressed linear-power constraint. Each fiducial plot is
# followed by a forecast using the same DESI central value and correlation but
# half of the DESI DR1 marginal errors.

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
from cup1d.postprocessing.chains import planck as planck_chains
from cup1d.postprocessing.importance_sampling import plot_importance_sampling
from cup1d.utils.utils import get_path_repo

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### Load the CMB chains used in this notebook
#
# The model and data-combination specifications live in one list, which is
# passed to the shared Planck-chain loader. This avoids re-reading the same
# chains in later plotting cells and uses the same loading path as the
# publication-figure plotters.

# %%

root_dir = os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")
planck_data = "plikHM_TTTEEE_lowl_lowE_linP"
planck_specs = [
    {"name": "cmb", "model": "base", "data": planck_data},
    {"name": "cmb_nrun", "model": "base_nrun", "data": planck_data},
    {"name": "cmb_nrunrun", "model": "base_nrun_nrunrun", "data": planck_data},
    {"name": "cmb_mnu", "model": "base_mnu", "data": planck_data},
    {"name": "cmb_nnu", "model": "base_nnu", "data": planck_data},
    {"name": "cmb_tau", "model": "base", "data": "plikHM_TTTEEE_lowl_linP"},
    {"name": "cmb_omega_k", "model": "base_omegak", "data": planck_data},
    {"name": "cmb_r", "model": "base_r", "data": planck_data},
    {
        "name": "cmb_nrun_nnu_w_mnu",
        "model": "base_nrun_nnu_w_mnu",
        "data": "plikHM_TTTEEE_lowl_lowE_BAO_Riess18_Pantheon18_lensing_linP",
    },
    {
        "name": "cmb_w_wa",
        "model": "base_w_wa",
        "data": "plikHM_TTTEEE_lowl_lowE_BAO_linP",
    },
]
planck_results = planck_chains.load_planck_2018_chains(planck_specs, root_dir=root_dir)
cmb = planck_results["cmb"]
cmb_nrun = planck_results["cmb_nrun"]
cmb_nrunrun = planck_results["cmb_nrunrun"]
cmb_mnu = planck_results["cmb_mnu"]
cmb_nnu = planck_results["cmb_nnu"]
cmb_tau = planck_results["cmb_tau"]
cmb_omega_k = planck_results["cmb_omega_k"]
cmb_r = planck_results["cmb_r"]
cmb_nrun_nnu_w_mnu = planck_results["cmb_nrun_nnu_w_mnu"]
cmb_w_wa = planck_results["cmb_w_wa"]

spa_specs = [
    {"name": "cmbspa_mnu", "model": "base_mnu", "data": "DESI_CMB-SPA"},
    {"name": "cmbspa_nnu", "model": "base_nnu", "data": "DESI_CMB-SPA"},
    {"name": "cmbspa_nrun", "model": "base_nrun", "data": "DESI_CMB-SPA"},
    {"name": "cmbspa_nrunrun", "model": "base_nrunrun", "data": "DESI_CMB-SPA"},
]
spa_results = planck_chains.load_spa_chains(spa_specs)
cmbspa_mnu = spa_results["cmbspa_mnu"]
cmbspa_nnu = spa_results["cmbspa_nnu"]
cmbspa_nrun = spa_results["cmbspa_nrun"]
cmbspa_nrunrun = spa_results["cmbspa_nrunrun"]


# %% [markdown]
# ### Load and unblind the DESI DR1 linear-power samples
#
# This cell reads the DESI contour polygons, summary statistics, and posterior
# samples.  The stored amplitude and slope are blinded, so we subtract the
# corresponding offsets before using them to form the DESI compressed
# constraint or reweight the CMB chains.

# %%
base_notebook = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/"
desi_chain_directory = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"

summary_mpg = np.load(desi_chain_directory + "summary.npy", allow_pickle=True).item()
blobs = np.load(desi_chain_directory + "blobs.npy")
blinding = np.load(base_notebook + "blinding.npy", allow_pickle=True).item()

delta2_star_samples = blobs["Delta2_star"].reshape(-1) - blinding["Delta2_star"]
n_star_samples = blobs["n_star"].reshape(-1) - blinding["n_star"]
correlation = np.corrcoef(delta2_star_samples, n_star_samples)[0, 1]
desi_dr1 = {
    "Delta2_star": summary_mpg["delta2_star_16_50_84"][1] - blinding["Delta2_star"],
    "n_star": summary_mpg["n_star_16_50_84"][1] - blinding["n_star"],
    "r": correlation,
    "Delta2_star_err": summary_mpg["delta2_star_err"],
    "n_star_err": summary_mpg["n_star_err"],
}
desi_dr1_half = desi_dr1 | {
    "Delta2_star_err": desi_dr1["Delta2_star_err"] / 2,
    "n_star_err": desi_dr1["n_star_err"] / 2,
}
# %% [markdown]
# #### Neutrino mass: $\sum m_\nu$

# %%
chains = [cmb_mnu, cmbspa_mnu]
plot_importance_sampling(chains, desi_dr1, "mnu", include_original_second_chain=True)

# %%
plot_importance_sampling(
    chains,
    desi_dr1_half,
    "mnu",
    include_original_second_chain=True,
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)


# %% [markdown]
# #### Running of the scalar spectral index: $\alpha_\mathrm{s}$

# %%
chains = [cmb_nrun, cmbspa_nrun]
plot_importance_sampling(
    chains, desi_dr1, "nrun", include_original_second_chain=True, use_as_ns=True
)

# %%
plot_importance_sampling(
    chains,
    desi_dr1_half,
    "nrun",
    include_original_second_chain=True,
    use_as_ns=True,
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Running and running of the scalar spectral index: $\alpha_\mathrm{s}$ and $\beta_\mathrm{s}$

# %%
plot_importance_sampling(
    [cmb_nrunrun, cmbspa_nrunrun],
    desi_dr1,
    "nrunrun",
    include_original_second_chain=True,
    use_as_ns=True,
)

# %%
plot_importance_sampling(
    [cmb_nrunrun, cmbspa_nrunrun],
    desi_dr1_half,
    "nrunrun",
    include_original_second_chain=True,
    use_as_ns=True,
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Effective number of relativistic species: $N_\mathrm{eff}$

# %%
plot_importance_sampling(
    [cmb_nnu, cmbspa_nnu], desi_dr1, "nnu", include_original_second_chain=True
)

# %%
plot_importance_sampling(
    [cmb_nnu, cmbspa_nnu],
    desi_dr1_half,
    "nnu",
    include_original_second_chain=True,
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Scalar amplitude and tilt: $A_\mathrm{s}$ and $n_\mathrm{s}$

# %%
plot_importance_sampling([cmb], desi_dr1, use_as_ns=True)

# %%
plot_importance_sampling(
    [cmb],
    desi_dr1_half,
    use_as_ns=True,
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Optical depth to reionization: $\tau$

# %%
plot_importance_sampling([cmb_tau], desi_dr1, "tau")

# %%
plot_importance_sampling(
    [cmb_tau],
    desi_dr1_half,
    "tau",
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Evolving dark energy: $w_0w_a$CDM

# %%
plot_importance_sampling(
    [cmb_w_wa], desi_dr1, additional_parameters=("w", "wa")
)

# %%
plot_importance_sampling(
    [cmb_w_wa],
    desi_dr1_half,
    additional_parameters=("w", "wa"),
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Spatial curvature: $\Omega_k$

# %%
plot_importance_sampling([cmb_omega_k], desi_dr1, "omegak")

# %%
plot_importance_sampling(
    [cmb_omega_k],
    desi_dr1_half,
    "omegak",
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Extended cosmology: $\alpha_\mathrm{s}$, $N_\mathrm{eff}$, $w_0$, and $\sum m_\nu$

# %%
plot_importance_sampling(
    [cmb_nrun_nnu_w_mnu],
    desi_dr1,
    "nrun",
    additional_parameters=("nnu", "w", "mnu"),
)

# %%
plot_importance_sampling(
    [cmb_nrun_nnu_w_mnu],
    desi_dr1_half,
    "nrun",
    additional_parameters=("nnu", "w", "mnu"),
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %% [markdown]
# #### Tensor-to-scalar ratio: $r$

# %%
plot_importance_sampling([cmb_r], desi_dr1, "r")

# %%
plot_importance_sampling(
    [cmb_r],
    desi_dr1_half,
    "r",
    desi_label=r"DESI $P_\mathrm{1D}$ (half errors)",
)

# %%

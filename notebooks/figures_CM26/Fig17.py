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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fig. 17
#
# Most correlated cosmological and nuisance parameters

# %%
import numpy as np

from cup1d.postprocessing.io import save_figure_data
from cup1d.postprocessing.literature import dr1_chain_directory, load_blinding_dr1
from cup1d.postprocessing.posterior import PosteriorSamples
from cup1d.postprocessing.posterior.parameters import (
    amplitude_transforms, parameters_vs_star_data, plot_parameters_vs_star,
)

# my local machine
folder = dr1_chain_directory()
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"

# unblinded star parameters; every second step of the chain
samples = PosteriorSamples.from_cup1d_folder(folder, blinding=load_blinding_dr1(), thin=2)

# %%
# the most correlated IGM and HCD parameters, HCD amplitudes as amplitudes
names = ["tau_eff_1", "HCD_damp1_0", "HCD_damp1_1"]
transforms = amplitude_transforms(names)
fig = plot_parameters_vs_star(samples, names, fontsize=20, show_correlation=False,
                              transforms=transforms)
# fig.savefig("figs/corr_compressed_mix.pdf", bbox_inches="tight")

# %%
save_figure_data(parameters_vs_star_data(samples, names, transforms), "fig_17.npy")

# %% [markdown]
# ## Other variations, check out local machine

# %%
import os

from cup1d.postprocessing.chains.products import write_chain_products
from cup1d.postprocessing.literature import dr1_output_root

base = str(dr1_output_root())

variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "zmax": ["Data: $z \\leq 3.4$", "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"],
    "zmin": ["Data: $z \\geq 2.6$", "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE_mpg": ["Data: w/ low SNR", "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "DESIY1_FFT3_dir_mpg": ["Data: FFT", "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    
    "no_emu_cov": ["Cov: w/o emu err", "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate": ["Cov: w/o 5\% err", "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate_no_emu_cov": ["Cov: w/o emu, 5\% err", "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE3_nyx": ["Model: lace-lyssa", "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/"],
    
    "cosmo": ["Model: $\omega_0\omega_a$CDM", "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_high": ["Model: high $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_low": ["Model: low $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"],
    
    "more_igm": ["Model: IGM $n_z=8$", "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"],
    "less_igm": ["Model: IGM $n_z=4$", "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"],
    "Turner24": ["Model: $\\bar{F}\\, n_z=1$", "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"],
    
    "hcd_z": ["Model: HCD $n_z=2$", "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"],
    "dlas": ["Model: only DLAs", "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"],
    
    "metals_z": ["Model: metals $n_z=2$", "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"],
    "metal_trad": ["Model: trad metal", "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"],
    "metal_thin": ["Model: metal thin", "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"],
    "metal_deco": ["Model: no metal decorr", "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"],
    "metal_si2": ["Model: no SiII-SiII", "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"],
    
    "no_res": ["Model: no resolution", "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"],
    
    # "sim_mpg_central": ["Val: mpg-central simulation", "DESIY1_QMLE3/sim_mpg_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm": ["Val: mpg-central simulation only IGM", "DESIY1_QMLE3/sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm0": ["Val: mpg-central simulation only cosmo", "DESIY1_QMLE3/sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"],
    # "sim_nyx_central": ["Val: nyx-central simulation", "DESIY1_QMLE3/sim_nyx_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_sherwood": ["Val: sherwood simulation", "DESIY1_QMLE3/sim_sherwood/CH24_mpgcen_gpr/chain_1/"],
}

# contours and summaries of every variation (read by the variation tables and figures)
for ii, var in enumerate(variations):
    folder = os.path.join(base, variations[var][1])
    if not os.path.isfile(os.path.join(folder, "chain.npy")):
        print("missing:", folder)
        continue
    write_chain_products(folder)

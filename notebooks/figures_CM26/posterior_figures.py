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
# # Posterior figures of the DESI DR1 analysis
#
# Check of `cup1d.postprocessing.posterior` (the sampler-independent figures
# also used for the lya_interface Cobaya runs) on the DESI DR1 baseline chain
# (`chain_7`). Each figure has a counterpart in the paper: best fit (Fig. 9),
# star parameters (Fig. 18), parameters against the star parameters (Fig. 17),
# IGM histories (Fig. 19), contaminants (Fig. 20) and resolution (Fig. 20d).

# %%
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import matplotlib.pyplot as plt

from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.postprocessing.p1d import P1DPlotter
from cup1d.postprocessing.posterior import PosteriorSamples, PosteriorPlotter
from cup1d.postprocessing.posterior import contaminants, histories, parameters, star
from cup1d.postprocessing.posterior.p1d import plot_best_fit
from cup1d.utils.utils import get_path_repo

# %% [markdown]
# ## Posterior samples
#
# The emcee chain is memory mapped; `thin=2` keeps every second step of every
# walker (~1.4 million samples, ~0.6 GB). The blinding offsets are removed
# from the stored star parameters.

# %%
# my local machine
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"

fname = os.path.join(get_path_repo("cup1d"), "data", "blinding_dr1.npy")
real_blinding = np.load(fname, allow_pickle=True).item()

samples = PosteriorSamples.from_cup1d_folder(folder, blinding=real_blinding, label="DESI DR1", thin=2)
print(len(samples), "samples")
print(samples.names)

# %% [markdown]
# Compare with Table 5 of the paper ($\Delta^2_\star = 0.379 \pm 0.032$,
# $n_\star = -2.309 \pm 0.019$):

# %%
for name in ["Delta2_star", "n_star"]:
    print(name, samples.quantiles(name))

# %% [markdown]
# ## Models
#
# The IGM and contaminant models of the baseline likelihood; nothing is
# emulated by the posterior figures.

# %%
args = Args(pre_defined="CM2026", system="local")
pip = Analysis(args, out_folder=None)
like = pip.fitter.like

data = next(iter(like.data.values()))
iz = np.argmin(np.abs(data.z - 3))
k_kms = np.linspace(data.k_kms[iz].min(), data.k_kms[iz].max(), 300)
z_grid = np.linspace(data.z.min(), data.z.max(), 60)

# %% [markdown]
# ## Best fit
#
# The saved minimum is stored in unit-cube coordinates of the chain's priors.
# Convert it with those priors (`samples.bounds`), not the current ones: the
# `ns` prior of CM2026 changed since `chain_7` (0.69-1.27 then, 0.91-1.03
# now), so the same cube point is a different physical point today.

# %%
fit = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = fit["fitter"]["mle_cube"]
best = {
    name: samples.bounds[name][0] + (samples.bounds[name][1] - samples.bounds[name][0]) * cube
    for name, cube in zip(samples.sampled, p0)
}  # name -> physical value

print("chi2 saved with the chain (paper):", -2 * fit["fitter"]["lnprob_mle"])
print("chi2 of that point, current likelihood:", like.get_chi2(best))
for name, (low, high) in samples.bounds.items():
    current = like.free_params[name]
    if not np.allclose((low, high), (current["min_value"], current["max_value"])):
        print(f"prior of {name} changed: chain {low:.4g}-{high:.4g}, now "
              f"{current['min_value']:.4g}-{current['max_value']:.4g}; best fit {best[name]:.4g}")

# %% [markdown]
# Optionally minimize again with the current likelihood, starting from the
# saved best fit (cup1d leaves two cores free). The new minimum replaces
# `best` in the figures below.

# %%
run_minimizer = True
if run_minimizer:
    # the objective must be given explicitly (Analysis.run_minimizer does it,
    # but also writes a result file); nothing is saved here
    pip.fitter.run_minimizer(log_func_minimize=pip.fitter.minus_log_prob, p0=best, restart=True)
    print("chi2 after minimization:", pip.fitter.mle_chi2)
    best = pip.fitter.parameters_from_sampling_point(pip.fitter.mle_cube)
    print("chi2 check:", like.get_chi2(best))

# %%
p1d_plotter = P1DPlotter(like)
bins, chi2, dof = p1d_plotter.prepare(best)
fig = plot_best_fit(bins, chi2, dof, ndata=p1d_plotter.ndata)
plt.show()

# %% [markdown]
# ## Star parameters

# %%
fig = star.plot_star_posterior(samples)
plt.show()

# %% [markdown]
# The DR1 Gaussian (black) is drawn on top of the full DR1 posterior: a check
# of the Gaussian approximation.

# %%
fig = star.plot_star_literature(samples)
plt.show()

# %% [markdown]
# ## Parameters against the star parameters
#
# Sampled values, i.e. logarithms for the metal and HCD amplitudes.

# %%
groups = parameters.group_parameters(samples)
for group, names in groups.items():
    print(group, names)

# %%
for group, names in groups.items():
    for i in range(0, len(names), 6):
        fig = parameters.plot_parameters_vs_star(samples, names[i:i + 6], title=group)
        plt.show()

# %%
fig = parameters.plot_correlation_matrix(samples)
plt.show()

# %% [markdown]
# ## IGM histories
#
# Without the DR1 IGM of ForestFlow's priors, which comes from this same chain.

# %%
draws = samples.subsample(4000, seed=0)
bands = histories.igm_histories(like.theory.model_igm, draws.columns(), z_grid)
best_igm = histories.igm_histories(like.theory.model_igm, {k: [v] for k, v in best.items()}, z_grid)
fig = histories.plot_igm_histories(bands, z_grid, best_fit=best_igm, label="DESI DR1")
plt.show()

# %% [markdown]
# ## Contaminants at z = 3

# %%
metals = contaminants.metal_contamination(like.theory.model_cont, like.theory.model_igm,
                                          draws.columns(), 3.0, k_kms)
ylabel = {label: r"$C_\mathrm{metal}$" for label in metals}
ylabel["SiII-SiII (additive)"] = r"$P_\mathrm{SiII-SiII}\,[\mathrm{km\,s^{-1}}]$"
fig = contaminants.plot_contamination(metals, k_kms, ylabel=ylabel, stacked=True, title=r"$z=3$")
plt.show()

# %%
hcd = contaminants.hcd_contamination(like.theory.model_cont, draws.columns(), 3.0, k_kms)
fig = contaminants.plot_contamination(hcd, k_kms, ylabel=r"$C_\mathrm{HCD}$", title=r"$z=3$")
plt.show()

# %% [markdown]
# ## Resolution against redshift

# %%
fig = histories.plot_parameters_vs_z(samples, {"R_coeff": args.fid_syst["R_coeff_znodes"]}, best_fit=best)
plt.show()

# %% [markdown]
# ## Everything to files
#
# What `lya_interface/scripts/chain_plots.py` does for Cobaya runs (without
# the best-fit P1D, done above).

# %%
save = False
if save:
    plotter = PosteriorPlotter(
        samples, "figs/posterior", igm=like.theory.model_igm, contaminants=like.theory.model_cont,
        syst_znodes={"R_coeff": args.fid_syst["R_coeff_znodes"]},
        z_range=(data.z.min(), data.z.max()), k_range=(k_kms[0], k_kms[-1]), best_fit=best,
    )
    plotter.plot_all()

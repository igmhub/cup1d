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
# # DESI DR1 $\chi^2$: what changed since the paper
#
# The paper best fit (`chain_7`) had $\chi^2 = 655.0$; the current likelihood
# gives 683.4 at the same point. Model, data vector and data covariance are
# unchanged; what changed is the **emulator covariance**,
# $C_\mathrm{emu} = R \,\odot\, P_s P_s^T$ ($R$ the relative emulator
# covariance of the emulator grid):
#
# * $P_s$, the P1D that scales $R$: measured P1D (paper) → `PSMOOTH` of the
#   file → our spline (`cup1d.p1ds.smooth_p1d`, current baseline);
# * the mapping of $R$ to the data redshifts: nearest emulator redshift
#   (paper) → linear interpolation between the two bracketing redshifts
#   (current).
#
# Here every ingredient is changed **one at a time** with respect to the
# current baseline, at fixed parameters (no minimization), and the $\chi^2$
# compared. The last rows combine changes (paper setup).

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
from astropy.io import fits

import cup1d.likelihood.likelihood as likelihood_module
from cup1d.configuration.args import Args
from cup1d.inference.analysis import Analysis
from cup1d.p1ds.observations.data_DESIY1 import set_p1d_filename
from cup1d.p1ds.smooth_p1d import read_cache, smooth_rows
from cup1d.postprocessing.chains.products import load_parameter_bounds

# %% [markdown]
# ## Likelihood and points
#
# The paper best fit is stored in unit-cube coordinates of the chain's priors
# (the `ns` prior changed since), so it is converted with the chain's bounds.
# Add other points to `points` (e.g. a new minimum) to repeat the comparison
# there.

# %%
args = Args(pre_defined="CM2026", system="local")
pip = Analysis(args, out_folder=None)
like = pip.fitter.like
key = next(iter(like.data))
data = like.data[key]

folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
fit = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
names, bounds = load_parameter_bounds(folder)
paper_best = {name: bounds[name][0] + (bounds[name][1] - bounds[name][0]) * cube
              for name, cube in zip(names, fit["fitter"]["mle_cube"])}
print("chi2 saved with the chain (paper):", -2 * fit["fitter"]["lnprob_mle"])
print("chi2 of that point, current likelihood:", like.get_chi2(paper_best))

points = {"paper best fit": paper_best}
# points["new minimum"] = {...}  # e.g. pip.fitter.parameters_from_sampling_point(pip.fitter.mle_cube)

# %% [markdown]
# ## Ingredients
#
# Full measurement (before scale cuts) from the cache of the baseline smooth
# P1D, to recompute spline variants; rows of the likelihood data in it.

# %%
fname = set_p1d_filename("QMLE3")
baseline_smooth = [np.array(p) for p in data.Pksmooth_kms]
full = cut_rows = None
for entry in read_cache(fname).values():
    rows = [np.flatnonzero((entry["z"] == z) & np.isin(entry["k"], k)) for z, k in zip(data.z, data.k_kms)]
    if all(len(r) == len(k) for r, k in zip(rows, data.k_kms)) and all(
            np.allclose(entry["smooth"][r], p) for r, p in zip(rows, baseline_smooth)):
        full, cut_rows = entry, rows
assert full is not None, "no cache entry matches the likelihood data"


def spline_variant(**options):
    """Spline on the full measurement with other options, at the likelihood rows."""
    smooth = smooth_rows(full["z"], full["k"], full["pk"], full["var"], **options)
    return [smooth[r] for r in cut_rows]


def spline_after_cuts(**options):
    """Spline fitted to the data after the scale cuts."""
    z = np.concatenate([np.full(len(k), zz) for zz, k in zip(data.z, data.k_kms)])
    var = np.concatenate([np.diag(c) for c in data.cov_Pk_kms])
    smooth = smooth_rows(z, np.concatenate(data.k_kms), np.concatenate(data.Pk_kms), var, **options)
    return np.split(smooth, np.cumsum([len(k) for k in data.k_kms])[:-1])


with fits.open(fname) as hdu:
    psmooth_file = [np.asarray(hdu["P1D_BLIND"].data["PSMOOTH"])[r] for r in cut_rows]
measured = [np.array(p) for p in data.Pk_kms]


def nearest_z_weights(z, k_Mpc, emu_z, emu_k_Mpc):
    """Paper mapping: nearest emulator redshift, nearest wavenumber there."""
    z, k_Mpc = np.atleast_1d(z), np.atleast_1d(k_Mpc)
    emu_z, emu_k_Mpc = np.asarray(emu_z), np.asarray(emu_k_Mpc)
    weights = np.zeros((z.size, emu_z.size))
    for i, (z_point, k_point) in enumerate(zip(z, k_Mpc)):
        candidates = np.flatnonzero(emu_z == emu_z[np.argmin(np.abs(emu_z - z_point))])
        weights[i, candidates[np.argmin(np.abs(emu_k_Mpc[candidates] - k_point))]] = 1.0
    return weights


# %% [markdown]
# ## Evaluation
#
# `evaluate` swaps one ingredient and rebuilds the covariance
# (`like.set_icov()`, a few seconds); the baseline ingredients are restored
# afterwards, and its covariance at the end of the loop.

# %%
interpolation_weights = likelihood_module.emulator_interpolation_weights
baseline_emu_cov_type = like.emu_cov_type


def evaluate(point, smooth=None, weights=None, emu_cov_type=None):
    """chi2 (total, per-redshift blocks) with some ingredients swapped."""
    data.Pksmooth_kms = smooth if smooth is not None else baseline_smooth
    likelihood_module.emulator_interpolation_weights = weights or interpolation_weights
    like.emu_cov_type = emu_cov_type or baseline_emu_cov_type
    try:
        like.set_icov()
        chi2, each = like.get_chi2(point, return_all=True)
    finally:
        data.Pksmooth_kms = baseline_smooth
        likelihood_module.emulator_interpolation_weights = interpolation_weights
        like.emu_cov_type = baseline_emu_cov_type
    return chi2, np.asarray(each[key])


def variations(point):
    """Every variation: name -> keyword arguments of evaluate."""
    model = like.get_p1d_kms(point)[0][key]
    return {
        "baseline (spline, z interpolation)": {},
        # scaling P1D of the emulator covariance
        "P_s = measured P1D": dict(smooth=measured),
        "P_s = PSMOOTH (file)": dict(smooth=psmooth_file),
        "P_s = best-fit model": dict(smooth=[np.array(m) for m in model]),
        "spline: plain GCV (x1)": dict(smooth=spline_variant(lam_factor=1.0)),
        "spline: no running median in z": dict(smooth=spline_variant(z_median=False)),
        "spline: no weight iteration": dict(smooth=spline_variant(n_iterations=1)),
        "spline: fitted after scale cuts": dict(smooth=spline_after_cuts()),
        # mapping and structure of the emulator covariance
        "nearest emulator z": dict(weights=nearest_z_weights),
        "emulator cov block-diagonal in z": dict(emu_cov_type="block"),
        "no emulator covariance": dict(smooth=[np.zeros_like(p) for p in measured]),
        # combinations
        "paper: measured P1D + nearest z": dict(smooth=measured, weights=nearest_z_weights),
        "PSMOOTH + nearest z": dict(smooth=psmooth_file, weights=nearest_z_weights),
    }


# %%
results = {}
for label, point in points.items():
    results[label] = {}
    for name, kwargs in variations(point).items():
        results[label][name] = evaluate(point, **kwargs)
        print(f"{label:>16s} | {name:38s} chi2 = {results[label][name][0]:8.2f}", flush=True)
like.set_icov()  # back to the baseline covariance

# %% [markdown]
# ## Summary
#
# $\Delta\chi^2$ with respect to the baseline, total and per-redshift blocks
# (with the full covariance the blocks do not add up to the total).

# %%
for label, res in results.items():
    base_total, base_each = res["baseline (spline, z interpolation)"]
    print(f"\n{label}: baseline chi2 = {base_total:.2f}")
    print(f"{'variation':38s} {'chi2':>8s} {'dchi2':>7s}  " + " ".join(f"{z:6.1f}" for z in data.z))
    for name, (total, each) in res.items():
        print(f"{name:38s} {total:8.2f} {total - base_total:+7.2f}  "
              + " ".join(f"{d:+6.1f}" for d in each - base_each))

# %%

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
# # Convergence of the P1D theory settings with ForestFlow emulators
#
# Three numerical settings control both the accuracy and the cost of a
# likelihood call with `forest_mpg_cinn` / `forest_mpg_gp`:
#
# 1. `k_rebin_factor`: the theory is evaluated on a fine k grid with this many
#    points per data bin and averaged over the bin edges
#    ([cup1d/utils/rebinning.py](../../cup1d/utils/rebinning.py)).
# 2. The P3D to P1D quadrature of ForestFlow,
#    `forestflow.statistics.p1d._DEFAULT_P1D_K_MU_INTEGRATOR`: Gauss-Legendre in
#    `ln k` at fixed `k_par`, split at `k = 0.4/Mpc` (`P1DIntegratorKMu`), over
#    $k_\perp\in[k_{\perp,\min}, k_{\perp,\max}]$. It is compared with the
#    Simpson rule in `ln k_perp` (`P1DIntegrator`).
# 3. `n_k_emu`: the emulator is evaluated on this many log-spaced k per
#    redshift and its P1D interpolated to the fine grid, instead of being
#    evaluated at every fine point (`null`).
#
# Requirement: $|\Delta\chi^2| < 0.05$ on DESI DR1, at the initial point of the
# fitter and at points displaced from it (the largest shift over the test
# points is reported), together with the time per $\chi^2$ call.
#
# The reference of section 2 must not share the truncation of the candidates:
# it uses a much wider $k_\perp$ range ($10^{-5}$ to $190$/Mpc, the linear power
# reaches 200/Mpc) and many more nodes, and its own convergence is printed.
# (An earlier version of this notebook used a reference on the default range
# $[10^{-3}, 100]$/Mpc, which hid a $\Delta\chi^2\simeq0.25$ bias from
# $k_{\perp,\min}=10^{-3}$; and too few high-k nodes in the default, 20.)

# %%
# %load_ext autoreload
# %autoreload 2

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import forestflow.statistics.p1d as ff_p1d
from cup1d import Analysis, Args
from cup1d.utils.utils import get_path_repo

# %% [markdown]
# ## Settings

# %%
EMULATOR_LABEL = "forest_mpg_gp"  # or "forest_mpg_cinn"
config_names = {
    "forest_mpg_cinn": "cm2026_base_forestmpgcinn.yaml",
    "forest_mpg_gp": "cm2026_base_forestmpggp.yaml",
}
config_file = os.path.join(
    get_path_repo("cup1d"), "configs", "desi_dr1_tests", config_names[EMULATOR_LABEL]
)

DCHI2_TARGET = 0.05
# Test points: the initial point and points displaced from it by these amounts
# in sampling-cube units. 0.01 and 0.03 stay within ~150 in chi2 of the initial
# point (what a fit or a chain explores); 0.1 gives chi2 of 10^3 - 10^4, where
# any projection error is amplified by the residuals.
CUBE_SHIFTS = (0.01, 0.03, 0.1)
n_timing = 1        # calls used to time each configuration

rebin_factors = [8, 16, 32, 64, 128]
rebin_reference = 128
n_k_emu_values = [24, 32, 48, 64, 96]

KMu = ff_p1d.P1DIntegratorKMu
REFERENCE = dict(k_perp_min_iMpc=1e-5, k_perp_max_iMpc=190.0, n_low=150, n_high=150)
REFERENCE_CHECK = dict(k_perp_min_iMpc=1e-5, k_perp_max_iMpc=150.0, n_low=300, n_high=300)
QUADRATURES = {
    "old default (1e-3, 20+20)": KMu(k_perp_min_iMpc=1e-3, n_low=20, n_high=20),
    "current default": KMu(),
    "k_perp_min=1e-3 (32+32)": KMu(k_perp_min_iMpc=1e-3, n_low=32, n_high=32),
    "k_perp_min=1e-5 (32+32)": KMu(k_perp_min_iMpc=1e-5, n_low=32, n_high=32),
    "n_high=16": KMu(n_low=32, n_high=16),
    "n_high=24": KMu(n_low=32, n_high=24),
    "n_high=48": KMu(n_low=32, n_high=48),
    "n_low=16": KMu(n_low=16, n_high=32),
    "n_low=24": KMu(n_low=24, n_high=32),
    "k_perp_max=150": KMu(k_perp_max_iMpc=150.0),
    "Simpson 1e-4..190, 96": ff_p1d.P1DIntegrator(k_perp_min_iMpc=1e-4, k_perp_max_iMpc=190.0, n_k_perp=96),
    "Simpson 1e-4..190, 200": ff_p1d.P1DIntegrator(k_perp_min_iMpc=1e-4, k_perp_max_iMpc=190.0, n_k_perp=200),
    "Simpson 1e-4..190, 300": ff_p1d.P1DIntegrator(k_perp_min_iMpc=1e-4, k_perp_max_iMpc=190.0, n_k_perp=300),
}

# %% [markdown]
# ## Helpers

# %%
FROM_YAML = object()
default_integrator = ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR


def build_analysis(k_rebin_factor=None, n_k_emu=FROM_YAML):
    args = Args.from_yaml(config_file, verbose=False)
    if k_rebin_factor is not None:
        args.k_rebin_factor = k_rebin_factor
    if n_k_emu is not FROM_YAML:
        args.n_k_emu = n_k_emu
    return Analysis(args)


def test_points(analysis):
    """Initial point of the fitter and points displaced from it in the cube."""
    x0 = analysis.fitter.sampling_point_from_parameters().copy()
    rng = np.random.default_rng(0)
    cubes = [x0] + [np.clip(x0 + shift * rng.standard_normal(x0.size), 0.01, 0.99)
                    for shift in CUBE_SHIFTS]
    return [analysis.fitter.parameters_from_sampling_point(cube) for cube in cubes]


def evaluate(analysis, points, integrator=None):
    """chi2 at every point, prediction at the first one, seconds per call."""
    try:
        if integrator is not None:
            ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = integrator
        chi2 = np.array([analysis.like.get_chi2(p) for p in points])
        t0 = time.time()
        for _ in range(n_timing):
            analysis.like.get_chi2(points[0])
        dt = (time.time() - t0) / n_timing
        pred, _ = analysis.like.get_p1d_kms(points[0])
    finally:
        ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = default_integrator
    flat = np.concatenate([np.concatenate(pred[key]) for key in pred])
    return chi2, flat, dt


def summary_table(results, reference, label):
    print(f"{label:>28s}  max|dchi2|   dchi2 per point          max|dP/P|   time [s]")
    for key, (chi2, pred, dt) in results.items():
        d = chi2 - reference[0]
        flag = "" if np.max(np.abs(d)) < DCHI2_TARGET else "  > target"
        print(f"{key!s:>28s}  {np.max(np.abs(d)):9.4f}   {np.array2string(d, precision=4):22s}"
              f"  {np.abs(pred / reference[1] - 1).max():.2e}  {dt:8.3f}{flag}")


# %% [markdown]
# ## 1. `k_rebin_factor`
#
# Reference: the largest factor, with the emulator at every fine point. The
# largest shift is reported separately from the near-fit points.

# %%
analysis = build_analysis(rebin_factors[0], n_k_emu=None)
points = test_points(analysis)

rebin = {}
for factor in rebin_factors:
    rebin[factor] = evaluate(build_analysis(factor, n_k_emu=None), points)
    print(f"k_rebin_factor={factor:3d}  chi2={rebin[factor][0]}  {rebin[factor][2]:.3f} s/call")

# %%
summary_table(rebin, rebin[rebin_reference], "k_rebin_factor")

# %%
chi2_ref, p_ref, _ = rebin[rebin_reference]
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
f = [x for x in rebin if x != rebin_reference]
axs[0].semilogx(f, [np.max(np.abs(rebin[x][0] - chi2_ref)) for x in f], "o-")
axs[0].axhline(DCHI2_TARGET, color="k", ls="--")
axs[0].set_ylabel(r"max $|\Delta\chi^2|$")
axs[1].loglog(f, [np.abs(rebin[x][1] / p_ref - 1).max() for x in f], "o-")
axs[1].set_ylabel("max |P1D / P1D_ref - 1|")
axs[2].semilogx(list(rebin), [rebin[x][2] for x in rebin], "o-")
axs[2].set_ylabel("time per chi2 call [s]")
for ax in axs:
    ax.set_xlabel("k_rebin_factor")
plt.tight_layout()

# %% [markdown]
# ## 2. P3D to P1D quadrature at the production `k_rebin_factor`
#
# `ArinyoModel.P1D_Mpc` looks up `_DEFAULT_P1D_K_MU_INTEGRATOR` at call time
# and dispatches on its type, so it is replaced temporarily. Each scan changes
# one setting of the current default (split at 0.4/Mpc, 32 + 32 nodes,
# $k_\perp\in[10^{-4}, 100]$/Mpc).

# %%
analysis = build_analysis(n_k_emu=None)
points = test_points(analysis)
ref = evaluate(analysis, points, KMu(**REFERENCE))
check = evaluate(analysis, points, KMu(**REFERENCE_CHECK))
print(f"reference chi2 {ref[0]}; self-convergence {check[0] - ref[0]}")
quadrature = {label: evaluate(analysis, points, integrator)
              for label, integrator in QUADRATURES.items()}

# %%
summary_table(quadrature, ref, "quadrature")

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, (chi2, _, dt) in quadrature.items():
    ax.semilogy(dt, max(np.max(np.abs(chi2 - ref[0])), 1e-5), "o")
    ax.annotate(label, (dt, max(np.max(np.abs(chi2 - ref[0])), 1e-5)), fontsize=7)
ax.axhline(DCHI2_TARGET, color="k", ls="--")
ax.set(xlabel="time per chi2 call [s]", ylabel=r"max $|\Delta\chi^2|$ over test points")
plt.tight_layout()

# %% [markdown]
# ## 3. `n_k_emu`: emulator on a coarse log-k grid
#
# Reference: the emulator evaluated at every fine point (`n_k_emu=None`), with
# the default quadrature and the production `k_rebin_factor`. The interpolation
# is a cubic spline in `(ln k, ln P)`. Log spacing is required: the data bins
# are linear in k and too coarse in `ln k` at low k.

# %%
analysis = build_analysis(n_k_emu=None)
points = test_points(analysis)
ref_emu = evaluate(analysis, points)
n_fine = len(analysis.like.Rebin_data.k_kms[next(iter(analysis.like.Rebin_data.k_kms))][0])
print(f"fine grid: {n_fine} k per redshift; n_k_emu=None: {ref_emu[2]:.3f} s/call")
interp = {n: evaluate(build_analysis(n_k_emu=n), points) for n in n_k_emu_values}

# %%
summary_table(interp, ref_emu, "n_k_emu")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
n = list(interp)
axs[0].plot(n, [np.max(np.abs(interp[i][0] - ref_emu[0])) for i in n], "o-")
axs[0].axhline(DCHI2_TARGET, color="k", ls="--")
axs[0].set_ylabel(r"max $|\Delta\chi^2|$")
axs[1].plot(n, [interp[i][2] for i in n], "o-")
axs[1].axhline(ref_emu[2], color="k", ls="--", label="every fine point")
axs[1].set_ylabel("time per chi2 call [s]")
axs[1].legend()
for ax in axs:
    ax.set_xlabel("n_k_emu")
plt.tight_layout()

# %% [markdown]
# ## 4. Baseline versus converged settings, all effects combined
#
# Baseline: the YAML settings (`k_rebin_factor=16`, `n_k_emu=64`) with the
# default quadrature. Converged: `k_rebin_factor=128`, `n_k_emu=128` and the
# reference quadrature, for both emulators.

# %%
converged = {"k_rebin_factor": 128, "n_k_emu": 128}
combined = {}
for label in ("forest_mpg_gp", "forest_mpg_cinn"):
    config_file = os.path.join(
        get_path_repo("cup1d"), "configs", "desi_dr1_tests", config_names[label]
    )
    baseline = build_analysis()
    points = test_points(baseline)
    chi2_base, _, t_base = evaluate(baseline, points)
    reference_analysis = build_analysis(converged["k_rebin_factor"], n_k_emu=converged["n_k_emu"])
    chi2_conv, _, t_conv = evaluate(reference_analysis, points, KMu(**REFERENCE))
    combined[label] = (chi2_base, chi2_conv, t_base, t_conv)

for label, (chi2_base, chi2_conv, t_base, t_conv) in combined.items():
    print(label)
    print(f"  time per call: baseline {t_base:.3f} s, converged {t_conv:.3f} s")
    for i, (b, c) in enumerate(zip(chi2_base, chi2_conv)):
        print(f"  point {i}: chi2 baseline {b:.4f}  converged {c:.4f}  dchi2 {b - c:+.4f}")

# %% [markdown]
# ## Take-away
#
# Results of the last run (forest_mpg_gp, DESI DR1, initial point and two
# displaced points with chi2 ~ 675, 1817 and 24844):
#
# - **Quadrature.** The lower limit must be $k_{\perp,\min}\le10^{-4}$/Mpc: with
#   $10^{-3}$ the projection misses ~$10^{-4}$ of P1D at the lowest k, coherently
#   over redshift ($|\Delta\chi^2|$ up to 0.78). The high-k segment needs >= 24
#   nodes and the low-k one >= 24. The default is now $[10^{-4},100]$/Mpc with
#   32 + 32 nodes: max $|\Delta\chi^2|=0.008$, 0.125 s per call (the previous
#   default, $10^{-3}$ and 20 + 20, gave 0.86). Simpson in $\ln k_\perp$ needs
#   ~300 nodes (0.56 s) for the same accuracy.
# - **`n_k_emu`.** 48 or more is converged (64 in the YAML: 0.003).
# - **`k_rebin_factor`.** Against 128 (itself converged to <0.01 against 192),
#   16 gives up to 0.09 (gp) and 0.11 (cinn) within ~150 in chi2 of the initial
#   point; 32 gives at most 0.009 (gp) and 0.011 (cinn), for ~15% more time
#   (gp) and none (cinn). The ForestFlow test configs now use 32. At points with
#   chi2 ~ 10^3 - 10^4 the errors grow (32: up to 0.28; 64: up to 0.07), which
#   matters only for the path of a minimizer started very far from the fit.
# - Choose the cheapest settings with max $|\Delta\chi^2|<0.05$ over the test
#   points, and check that the minimizer result is stable by repeating
#   `analysis.run_minimizer` at the candidate and the reference settings.

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
# 2. The P3D to P1D quadrature of ForestFlow. The default is
#    `P1DIntegratorKMu`: Gauss-Legendre in `ln k` at fixed `k_par`, split at
#    `k = 0.4/Mpc` (20 + 20 nodes). It is compared with the same rule without
#    the split and with the previous Simpson rule in `ln k_perp`
#    (`P1DIntegrator`). See also
#    `ForestFlow/notebooks/emulator/convergence_p1d_quadrature.py` for the
#    model-level test of the quadrature on its own.
# 3. `n_k_emu`: the emulator is evaluated on this many log-spaced k per
#    redshift and its P1D interpolated to the fine grid, instead of being
#    evaluated at every fine point (`null`).
#
# Sections 1 and 2 evaluate the emulator at every fine point (`n_k_emu=None`)
# to isolate each effect. For each setting we report the change in the
# predicted P1D and in chi2 relative to a converged reference, together with
# the time per chi2 call. We suggest |dchi2| < 0.1.

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

rebin_factors = [1, 2, 4, 8, 16, 32]
rebin_reference = 32

# Simpson in ln k_perp (P1DIntegrator): number of nodes
# nodes_simpson = [30, 40, 50, 60, 70, 80, 99]
nodes_simpson = [99]
# Gauss-Legendre in ln k without split (P1DIntegratorKMu, k_split_iMpc=None):
# total number of nodes
# nodes_gl_nosplit = [16, 24, 32, 40, 48, 64]
nodes_gl_nosplit = [99]
# Gauss-Legendre in ln k split at k_split: nodes per segment
nodes_gl_split = [12, 16, 20, 24, 32]
k_split = 0.4
# reference: Gauss-Legendre in ln k_perp, independent of the rules above
nodes_reference = 2048

n_k_emu_values = [24, 32, 48, 64, 96]

n_timing = 1  # calls used to time each configuration


# %% [markdown]
# ## Helpers

# %%
FROM_YAML = object()


def build_analysis(k_rebin_factor=None, n_k_emu=FROM_YAML):
    args = Args.from_yaml(config_file, verbose=False)
    if k_rebin_factor is not None:
        args.k_rebin_factor = k_rebin_factor
    if n_k_emu is not FROM_YAML:
        args.n_k_emu = n_k_emu
    return Analysis(args)


def evaluate(analysis, point):
    """chi2, rebinned prediction (flat vector), seconds per call."""
    chi2 = analysis.fitter.get_chi2(point)
    t0 = time.time()
    for _ in range(n_timing):
        analysis.fitter.get_chi2(point)
    dt = (time.time() - t0) / n_timing
    pred, _ = analysis.like.get_p1d_kms(point)
    flat = np.concatenate(
        [np.concatenate(pred[key]) for key in pred]
    )
    return chi2, flat, dt


def summary_table(results, reference, label):
    print(f"{label:>12s}    dchi2    max|dP/P|   time [s]")
    for key, (chi2, pred, dt) in results.items():
        print(f"{key!s:>12s} {chi2 - reference[0]:9.4f}  "
              f"{np.abs(pred / reference[1] - 1).max():.2e}  {dt:8.3f}")


# %% [markdown]
# ## 1. `k_rebin_factor`
#
# Reference: the largest factor. The test point is the initial point of the
# fitter (close to the best fit); repeat with other points for a stronger test.

# %%
analysis = build_analysis(rebin_factors[0], n_k_emu=None)
point = analysis.fitter.initial_parameters()

rebin = {}
for factor in rebin_factors:
    analysis = build_analysis(factor, n_k_emu=None)
    rebin[factor] = evaluate(analysis, point)
    print(f"k_rebin_factor={factor:3d}  chi2={rebin[factor][0]:.4f}  "
          f"{rebin[factor][2]:.3f} s/call")

# %%
summary_table(rebin, rebin[rebin_reference], "factor")

# %%
chi2_ref, p_ref, _ = rebin[rebin_reference]
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
f = [x for x in rebin if x != rebin_reference]
axs[0].semilogx(f, [rebin[x][0] - chi2_ref for x in f], "o-")
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_ylabel(r"$\chi^2 - \chi^2_{\rm ref}$")
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
# `ArinyoModel.P1D_Mpc` uses
# `forestflow.statistics.p1d._DEFAULT_P1D_K_MU_INTEGRATOR` when no integrator
# is passed, so we replace it temporarily. It dispatches on the integrator
# type: a `P1DIntegrator` integrates `P3D(k_par, k_perp)` over `ln k_perp`,
# a `P1DIntegratorKMu` integrates `P3D(k, mu)` over `ln k`.

# %%
analysis = build_analysis(n_k_emu=None)
point = analysis.fitter.initial_parameters()
default_integrator = ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR

schemes = {
    "Simpson ln k_perp": {
        n: ff_p1d.P1DIntegrator(n_k_perp=n, method="simpson") for n in nodes_simpson
    },
    "GL ln k, no split": {
        n: ff_p1d.P1DIntegratorKMu(n_low=n // 2, n_high=n - n // 2, k_split_iMpc=None)
        for n in nodes_gl_nosplit
    },
    f"GL ln k, split {k_split}": {
        2 * n: ff_p1d.P1DIntegratorKMu(n_low=n, n_high=n, k_split_iMpc=k_split)
        for n in nodes_gl_split
    },
}

quadrature = {}
try:
    print(f"Evaluating reference (GL ln k_perp, {nodes_reference} nodes) ...")
    ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = ff_p1d.P1DIntegrator(
        n_k_perp=nodes_reference, method="gauss_legendre"
    )
    ref = evaluate(analysis, point)
    print(f"reference: chi2={ref[0]:.4f}  {ref[2]:.3f} s/call")
    for label, integrators in schemes.items():
        quadrature[label] = {}
        for n, integrator in integrators.items():
            ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = integrator
            quadrature[label][n] = evaluate(analysis, point)
            print(f"{label:22s} n={n:4d}  chi2={quadrature[label][n][0]:.4f}  "
                  f"{quadrature[label][n][2]:.3f} s/call")
finally:
    ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = default_integrator

# %%
for label, res in quadrature.items():
    print(label)
    summary_table(res, ref, "total nodes")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for (label, res), marker in zip(quadrature.items(), "os^"):
    n = list(res)
    axs[0].plot(n, [res[i][0] - ref[0] for i in n], marker + "-", label=label)
    axs[1].semilogy(
        n, [np.abs(res[i][1] / ref[1] - 1).max() for i in n], marker + "-", label=label
    )
    axs[2].semilogy([res[i][2] for i in n], [abs(res[i][0] - ref[0]) for i in n],
                    marker + "-", label=label)
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_ylabel(r"$\chi^2 - \chi^2_{\rm ref}$")
axs[1].set_ylabel("max |P1D / P1D_ref - 1|")
for ax in axs[:2]:
    ax.set_xlabel("total number of nodes")
axs[2].set_xlabel("time per chi2 call [s]")
axs[2].set_ylabel(r"$|\Delta\chi^2|$")
axs[2].axhline(0.1, color="k", ls="--")
axs[0].legend()
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
point = analysis.fitter.initial_parameters()
ref_emu = evaluate(analysis, point)
print(f"n_k_emu=None: chi2={ref_emu[0]:.4f}  {ref_emu[2]:.3f} s/call")
n_fine = len(analysis.like.Rebin_data.k_kms[next(iter(analysis.like.Rebin_data.k_kms))][0])
print(f"fine grid: {n_fine} k per redshift")

interp = {}
for n in n_k_emu_values:
    interp[n] = evaluate(build_analysis(n_k_emu=n), point)
    print(f"n_k_emu={n:4d}  chi2={interp[n][0]:.4f}  {interp[n][2]:.3f} s/call")

# %%
summary_table(interp, ref_emu, "n_k_emu")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
n = list(interp)
axs[0].plot(n, [interp[i][0] - ref_emu[0] for i in n], "o-")
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_ylabel(r"$\chi^2 - \chi^2_{\rm ref}$")
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
# default quadrature (Gauss-Legendre split at 0.4/Mpc, 20 + 20 nodes).
# Converged: `k_rebin_factor=128`, `n_k_emu=128`, 64 + 64 nodes. Both
# emulators are tested at the initial point and at two points displaced from
# it in the sampling cube.

# %%
converged = {"k_rebin_factor": 128, "n_k_emu": 128, "nodes": 64}
cube_shift = 0.1  # displacement of the extra test points in cube units


def evaluate_points(analysis, cubes):
    chi2 = []
    for cube in cubes:
        p = analysis.fitter.parameters_from_sampling_point(cube)
        chi2.append(analysis.like.get_chi2(p))
    return np.array(chi2)


combined = {}
for label in ("forest_mpg_gp", "forest_mpg_cinn"):
    config_file = os.path.join(
        get_path_repo("cup1d"), "configs", "desi_dr1_tests", config_names[label]
    )
    baseline = build_analysis()
    x0 = baseline.fitter.sampling_point_from_parameters().copy()
    rng = np.random.default_rng(0)
    cubes = [x0] + [
        np.clip(x0 + cube_shift * rng.standard_normal(x0.size), 0.01, 0.99)
        for _ in range(2)
    ]
    t0 = time.time()
    chi2_base = evaluate_points(baseline, cubes)
    t_base = (time.time() - t0) / len(cubes)

    reference_analysis = build_analysis(
        converged["k_rebin_factor"], n_k_emu=converged["n_k_emu"]
    )
    try:
        ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = ff_p1d.P1DIntegratorKMu(
            n_low=converged["nodes"], n_high=converged["nodes"]
        )
        t0 = time.time()
        chi2_conv = evaluate_points(reference_analysis, cubes)
        t_conv = (time.time() - t0) / len(cubes)
    finally:
        ff_p1d._DEFAULT_P1D_K_MU_INTEGRATOR = default_integrator
    combined[label] = (chi2_base, chi2_conv, t_base, t_conv)

for label, (chi2_base, chi2_conv, t_base, t_conv) in combined.items():
    print(label)
    print(f"  time per call: baseline {t_base:.3f} s, converged {t_conv:.3f} s")
    for i, (b, c) in enumerate(zip(chi2_base, chi2_conv)):
        print(f"  point {i}: chi2 baseline {b:.4f}  converged {c:.4f}  dchi2 {b - c:+.4f}")

# %% [markdown]
# ## Take-away
#
# Choose the cheapest settings with |dchi2| < 0.1 at the reference point, and
# confirm at a few other points (best fit, edges of the prior). Because chi2
# shifts from quadrature noise act like a small random offset, also check that
# the *minimizer* result is stable by repeating `analysis.run_minimizer` at the
# candidate and the reference settings.

# %% [markdown]
#

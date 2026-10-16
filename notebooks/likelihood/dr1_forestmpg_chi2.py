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
# # DESI DR1 Forest-MPG: χ² versus number of ForestFlow realizations
#
# This notebook builds only the `dr1_forestmpg` likelihood and evaluates its
# initial point with either ForestFlow emulator: `forest_mpg_cinn` or
# `forest_mpg_gp` (choose `EMULATOR_LABEL` below).  The cINN averages latent
# realizations before returning Arinyo coefficients: change `N_REALIZATIONS` and
# rerun the evaluation cell to quantify the resulting change in χ².  The cINN uses
# a fixed seed, so a given realization count is reproducible.  The GP is
# deterministic, so the realization settings and scan do not apply to it.

# %%
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from cup1d import Analysis, Args
from cup1d.emulator.factory import set_emulator
from cup1d.utils.utils import get_path_repo


# %% [markdown]
# ## Configure the ForestFlow averaging

# %%
# EMULATOR_LABEL = "forest_mpg_cinn"
EMULATOR_LABEL = "forest_mpg_gp"
SAMPLING = EMULATOR_LABEL == "forest_mpg_cinn"  # only the cINN has latent draws
CONFIG_NAMES = {
    "forest_mpg_cinn": "cm2026_base_forestmpgcinn.yaml",
    "forest_mpg_gp": "cm2026_base_forestmpggp.yaml",
}

# Set this to the number of latent ForestFlow realizations to average for every
# emulator prediction.  Typical comparisons are 100, 500, 1000, and 3000.
N_REALIZATIONS = 2048
SEEDS = (3, 17)
METHODS = (
    # ("gaussian", "mean", "transformed", "nested"),
    # ("antithetic", "mean", "transformed", "nested"),
    # ("sobol", "mean", "transformed", "nested"),
    ("sobol_antithetic", "mean", "transformed", "nested"),
)


# %% [markdown]
# ## Build the DR1 Forest-MPG likelihood

# %%
config_file = Path(get_path_repo("cup1d")) / "configs" / "desi_dr1_tests" / CONFIG_NAMES[EMULATOR_LABEL]
args = Args.from_yaml(config_file, verbose=False)
# Keep this diagnostic out of the production DR1 output tree.
args.out_folder = "/tmp/cup1d_dr1_forestmpg_chi2"

# Build the same ForestFlow wrapper selected by the YAML, then set its P3D
# emulator's default number of latent realizations before cup1d evaluates it.
emulator = set_emulator(args.emulator_label)
if SAMPLING:
    emulator.emulator.Nrealizations = int(N_REALIZATIONS)
analysis = Analysis(args, emulator=emulator)
initial_point = analysis.fitter.initial_parameters()

print("Emulator:", analysis.emulator.emulator_label)
if SAMPLING:
    print("ForestFlow realizations:", analysis.emulator.emulator.Nrealizations)


# %% [markdown]
# ## Evaluate χ² at the standard initial point

# %%
def initial_chi2_for(n_realizations, method=("sobol", "mean", "transformed", "nested"), seed=0):
    """Evaluate the fixed initial point for one latent-sampling configuration."""
    if SAMPLING:
        n_realizations = int(n_realizations)
        if n_realizations < 1:
            raise ValueError("n_realizations must be a positive integer")
        sampler, statistic, aggregation_space, draw_policy = method
        analysis.emulator.emulator.Nrealizations = n_realizations
        analysis.emulator.set_sampling_options(
            sampler=sampler,
            statistic=statistic,
            aggregation_space=aggregation_space,
            draw_policy=draw_policy,
            seed=seed,
        )
    # Do not reuse any batched likelihood prediction after changing the
    # ForestFlow setting.
    analysis.emulator.clear_prediction_cache()
    start = perf_counter()
    chi2 = analysis.fitter.get_chi2(initial_point)
    return chi2, perf_counter() - start


chi2, seconds = initial_chi2_for(N_REALIZATIONS, seed=SEEDS[0])
setting = f"Nrealizations={N_REALIZATIONS}" if SAMPLING else "deterministic GP"
print(f"{EMULATOR_LABEL} ({setting}): chi2_initial={chi2:.6f} ({seconds:.3f} s)")


# %% [markdown]
# ## Optional realization-count scan
#
# Change this tuple (or set it to one value) and rerun the cell.  No parameters
# are fitted here: every entry evaluates exactly the same `initial_point`.

# %%
N_REALIZATIONS_SCAN = (1024, 2048)
if not SAMPLING:
    print("The GP is deterministic: no realization scan.")
    N_REALIZATIONS_SCAN, METHODS = (), ()
scan_rows = []
for method in METHODS:
    for n_realizations in N_REALIZATIONS_SCAN:
        for seed in SEEDS:
            chi2, seconds = initial_chi2_for(n_realizations, method, seed)
            scan_rows.append({"method": method, "Nrealizations": n_realizations,
                              "seed": seed, "chi2": chi2, "seconds": seconds})

summary = []
for method in METHODS:
    for n_realizations in N_REALIZATIONS_SCAN:
        selected = [row for row in scan_rows if row["method"] == method and row["Nrealizations"] == n_realizations]
        chi2s = np.asarray([row["chi2"] for row in selected])
        times = np.asarray([row["seconds"] for row in selected])
        summary.append({"method": method, "Nrealizations": n_realizations,
                        "chi2_mean": chi2s.mean(), "chi2_std": chi2s.std(ddof=1),
                        "time_mean_seconds": times.mean(), "time_std_seconds": times.std(ddof=1)})

fig, (chi2_axis, time_axis) = plt.subplots(1, 2, figsize=(12, 4))
for method in METHODS:
    selected = [row for row in summary if row["method"] == method]
    counts = [row["Nrealizations"] for row in selected]
    chi2_axis.errorbar(counts, [row["chi2_mean"] for row in selected],
                        yerr=[row["chi2_std"] for row in selected], marker="o", capsize=3,
                        label="/".join(method))
    time_axis.errorbar(counts, [row["time_mean_seconds"] for row in selected],
                       yerr=[row["time_std_seconds"] for row in selected], marker="o", capsize=3,
                       label="/".join(method))
for axis in (chi2_axis, time_axis):
    axis.set_xscale("log")
    axis.legend(fontsize=7)
chi2_axis.set_ylabel("DESI DR1 initial chi2 ± seed std")
time_axis.set_ylabel("evaluation time [s] ± seed std")
for axis in (chi2_axis, time_axis):
    axis.set_xlabel("ForestFlow realizations")
fig.tight_layout()

for row in summary:
    print(f"{'/'.join(row['method'])}, N={row['Nrealizations']:6d}: "
          f"chi2={row['chi2_mean']:.6f} ± {row['chi2_std']:.6f}; "
          f"time={row['time_mean_seconds']:.3f} ± {row['time_std_seconds']:.3f} s")

# %% [markdown]
# ## Compare the two emulators at the initial point
#
# Builds the likelihood once per emulator (cINN with `N_REALIZATIONS` draws, the
# GP deterministic) and evaluates the same initial point, repeating each
# evaluation to time it. The cINN is evaluated at its first seed.

# %%
comparison = {}
for label in ("forest_mpg_cinn", "forest_mpg_gp"):
    args_cmp = Args.from_yaml(
        Path(get_path_repo("cup1d")) / "configs" / "desi_dr1_tests" / CONFIG_NAMES[label],
        verbose=False,
    )
    args_cmp.out_folder = "/tmp/cup1d_dr1_forestmpg_chi2"
    emulator_cmp = set_emulator(label)
    if label == "forest_mpg_cinn":
        emulator_cmp.emulator.Nrealizations = int(N_REALIZATIONS)
        emulator_cmp.set_sampling_options(seed=SEEDS[0])
    analysis_cmp = Analysis(args_cmp, emulator=emulator_cmp)
    point = analysis_cmp.fitter.initial_parameters()
    analysis_cmp.fitter.get_chi2(point)  # warm-up
    times = []
    for _ in range(5):
        emulator_cmp.clear_prediction_cache()
        start = perf_counter()
        chi2_cmp = analysis_cmp.fitter.get_chi2(point)
        times.append(perf_counter() - start)
    comparison[label] = (chi2_cmp, float(np.median(times)))
    print(f"{label:16s} chi2={chi2_cmp:.4f}  median time {comparison[label][1] * 1e3:.1f} ms")

# %%

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
# # Convergence of the rebinning with two P1D samples (DESI DR1 + high resolution)
#
# `configs/hires/hires_forestflow.yaml` fits DESI DR1 QMLE3 and Karacayli2022,
# which have the same redshifts but very different bins: DESI has ~50-70
# linear bins up to ~0.03 s/km, Karacayli2022 11 overlapping log bins up to
# ~0.07 s/km. The model is averaged over each data bin on a fine k grid
# (`k_rebin_factor` points per bin), and the emulator is evaluated on
# `n_k_emu` log-spaced points per redshift and interpolated to the fine grid.
#
# Two ways of building the fine grid (`shared_k_grid`,
# [cup1d/utils/rebinning.py](../../cup1d/utils/rebinning.py)):
#
# * **separate**: one grid per data set (32 points per bin each), so the model
#   is evaluated and contaminated twice per redshift, and a Karacayli2022 bin
#   is averaged over only 32 points;
# * **shared** (default): one grid per redshift. The DESI grid is extended
#   with its spacing up to the lower edge of the next Karacayli2022 bin; beyond,
#   the grid has the spacing of the separate scheme for Karacayli2022. The
#   contaminated P1D is computed once, each data set applies its own
#   resolution correction (DESI only here), and it is averaged into every data
#   set's bins with overlap-length weights.
#
# The candidates are compared with a reference (shared grid, much finer
# rebinning, emulator at every fine point), per data set, at the initial point
# and at points displaced from it by 1% and 3% of each prior range.
# Requirement: $|\Delta\chi^2| < 0.05$ per data set. The time per $\chi^2$ call
# is reported too.
#
# Only the rebinning changes between candidates: one native `Analysis` is
# built, and its `Rebinning` replaced.

# %%
import os

# leave two cores free
N_THREADS = max(1, (os.cpu_count() or 3) - 2)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from cup1d import Analysis, Args
from cup1d.utils.rebinning import Rebinning
from cup1d.utils.utils import get_path_repo

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

CONFIG = Path(get_path_repo("cup1d")) / "configs/hires/hires_forestflow.yaml"
OUT_DIR = Path(get_path_repo("cup1d")) / "data/validation/convergence_shared_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE = 0.05
SHIFTS = (0.0, 0.01, 0.03)
N_PER_SHIFT = 3
REFERENCE = dict(shared=True, k_rebin_factor=128, n_k_emu=None)
CANDIDATES = {
    "separate 32 / 64": dict(shared=False, k_rebin_factor=32, n_k_emu=64),
    "shared 32 / 64 (default)": dict(shared=True, k_rebin_factor=32, n_k_emu=64),
    "shared 32 / 96": dict(shared=True, k_rebin_factor=32, n_k_emu=96),
    "shared 32 / 128": dict(shared=True, k_rebin_factor=32, n_k_emu=128),
    "shared 64 / 64": dict(shared=True, k_rebin_factor=64, n_k_emu=64),
    "shared 32 / every k": dict(shared=True, k_rebin_factor=32, n_k_emu=None),
    "separate 128 / every k": dict(shared=False, k_rebin_factor=128, n_k_emu=None),
}

# %%
args = Args.from_yaml(str(CONFIG), verbose=False)
args.out_folder = str(OUT_DIR / "native")
analysis = Analysis(args)
like = analysis.fitter.like
print("data sets:", list(like.data), " redshifts:", np.round(like.data["DESIY1_QMLE3"].z, 2))

start = {name: p["value"] for name, p in like.free_params.items()}
bounds = {name: (p["min_value"], p["max_value"]) for name, p in like.free_params.items()}
rng = np.random.default_rng(0)
points = []
for shift in SHIFTS:
    for ii in range(1 if shift == 0 else N_PER_SHIFT):
        point = {name: float(np.clip(value + shift * (bounds[name][1] - bounds[name][0]) * rng.normal(),
                                     *bounds[name])) for name, value in start.items()}
        points.append((f"shift {shift} #{ii}", point))


def set_rebinning(shared, k_rebin_factor, n_k_emu):
    """Replace the likelihood's fine grids; everything else is unchanged."""
    like.Rebin_data = Rebinning(like.data, k_rebin_factor=k_rebin_factor, n_k_emu=n_k_emu,
                                shared_k_grid=shared)
    like._union_grid = None  # cached union of the emulator grids


def chi2_per_dataset(settings):
    """chi2 per data set at every point, and seconds per chi2 call."""
    set_rebinning(**settings)
    like.get_chi2(points[0][1])  # warm up
    values = []
    t0 = time.perf_counter()
    for _, point in points:
        _, per_z = like.get_chi2(point, return_all=True)
        values.append({key: float(np.sum(v)) for key, v in per_z.items()})
    return values, (time.perf_counter() - t0) / len(points)


grid_size = {}
for label, settings in {"reference": REFERENCE, **CANDIDATES}.items():
    set_rebinning(**settings)
    grid_size[label] = sum(len(like.Rebin_data.k_kms[key][0]) for keys in like.Rebin_data.groups
                           for key in keys[:1])
    print(f"{label:>26s}: {grid_size[label]:5d} fine points at z = {like.data['DESIY1_QMLE3'].z[0]:.1f}")

# %% [markdown]
# ## Reference

# %%
reference, time_reference = chi2_per_dataset(REFERENCE)
print(f"reference: {time_reference:.2f} s per chi2")
for (label, _), values in zip(points, reference):
    print(f"{label:>14s}  " + "  ".join(f"{key} {value:9.3f}" for key, value in values.items()))

# %% [markdown]
# ## Candidates
#
# Largest $|\Delta\chi^2|$ over the points, per data set and total, and the time
# per call.

# %%
results = {}
for label, settings in CANDIDATES.items():
    values, seconds = chi2_per_dataset(settings)
    delta = {key: np.array([v[key] - r[key] for v, r in zip(values, reference)]) for key in like.data}
    delta["total"] = sum(delta[key] for key in like.data)
    results[label] = dict(delta=delta, seconds=seconds)

keys = list(like.data) + ["total"]
print(f"{'candidate':>26s}" + "".join(f"{key:>16s}" for key in keys) + f"{'s / chi2':>10s}  ok")
for label, result in results.items():
    worst = {key: result["delta"][key][np.argmax(np.abs(result["delta"][key]))] for key in keys}
    ok = all(abs(worst[key]) < TOLERANCE for key in like.data)
    print(f"{label:>26s}" + "".join(f"{worst[key]:16.4f}" for key in keys)
          + f"{result['seconds']:10.3f}  {'yes' if ok else 'NO'}")

# %% [markdown]
# $\Delta\chi^2$ against the reference at each test point.

# %%
fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), sharey=True)
x = np.arange(len(points))
for ax, key in zip(axes, keys):
    for label, result in results.items():
        ax.plot(x, result["delta"][key], "o-", ms=3, label=label)
    ax.axhspan(-TOLERANCE, TOLERANCE, color="0.9")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_title(key.replace("_", r"\_"))
    ax.set_xticks(x, [label for label, _ in points], rotation=60, fontsize=8)
axes[0].set_ylabel(r"$\chi^2 - \chi^2_{\rm ref}$")
axes[-1].legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "delta_chi2.png", dpi=150)

# %% [markdown]
# ## The shared grid at one redshift
#
# Fine-cell width against k, with the data bins of both samples: DESI spacing up
# to the lower edge of the next Karacayli2022 bin, then the separate-scheme
# spacing of Karacayli2022.

# %%
set_rebinning(**CANDIDATES["shared 32 / 64 (default)"])
from cup1d.utils.rebinning import shared_fine_grid

k, lower, upper = shared_fine_grid(list(like.data.values()), 0, 32)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(k, upper - lower, ".", ms=2, color="k", label="fine cells")
for key, color in zip(like.data, ("C0", "C1")):
    data = like.data[key]
    for lo, hi in zip(data.k_kms_min[0], data.k_kms_max[0]):
        ax.axvspan(lo, hi, ymin=0.0 if color == "C0" else 0.5, ymax=0.5 if color == "C0" else 1.0,
                   alpha=0.15, color=color, lw=0)
    ax.plot([], [], color=color, lw=6, alpha=0.3, label=key.replace("_", r"\_"))
ax.set_yscale("log")
ax.set_xlabel(r"$k_\parallel\ [{\rm s\,km^{-1}}]$")
ax.set_ylabel("cell width [s/km]")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()

del analysis, like
gc.collect();

# %%

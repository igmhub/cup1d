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
# # Plots of P1D and cov of DESI-DR1
#
# We make multiple comparisons between QMLE and FFT measurements
#
# We also plot covariance matrix, and look at the contributions to P(k) in the files

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

from cup1d.configuration import Args
from cup1d.p1ds.factory import set_p1d

# %% [markdown]
# Load the baseline and variations

# %%

baseline_args = Args.from_baseline()
baseline_data = set_p1d(baseline_args, baseline_args.data_label[0])
baseline_data.plot_p1d()

qmle_args = Args.from_variation("DESIY1_QMLE")
qmle_data = set_p1d(qmle_args, qmle_args.data_label[0])
qmle_data.plot_p1d()

fft3_dir_args = Args.from_variation("DESIY1_FFT3_dir")
fft3_dir_data = set_p1d(fft3_dir_args, fft3_dir_args.data_label[0])
fft3_dir_data.plot_p1d()

# %% [markdown]
# Compare the fractional P1D difference of QMLE and FFT3_dir with respect to
# the baseline. FFT3_dir is interpolated onto the baseline k grid over their
# common k range.

# %%
variations = {
    "QMLE": qmle_data,
    "FFT3_dir": fft3_dir_data,
}

fig, axes = plt.subplots(4, 3, figsize=(10, 10))
axes = axes.reshape(-1)

for iz, z in enumerate(baseline_data.z):
    axis = axes[iz]
    k_baseline = baseline_data.k_kms[iz]
    p_baseline = baseline_data.Pk_kms[iz]
    for label, variation_data in variations.items():
        k_variation = variation_data.k_kms[iz]
        mask = (k_baseline >= k_variation.min()) & (k_baseline <= k_variation.max())
        p_variation = np.interp(
            k_baseline[mask], k_variation, variation_data.Pk_kms[iz]
        )
        axis.plot(k_baseline[mask], p_variation / p_baseline[mask] - 1, label=label)
    axis.set_title(rf"$z={z:.1f}$")
    axis.axhline(0, color="k", ls="--", alpha=0.4)
    axis.axhline(0.01, color="k", ls=":", alpha=0.4)
    axis.axhline(-0.01, color="k", ls=":", alpha=0.4)

for axis in axes[len(baseline_data.z) :]:
    axis.set_axis_off()
axes[0].legend(loc="upper right")
fig.supxlabel(r"$k[\mathrm{km}^{-1}\,\mathrm{s}]$")
fig.supylabel(r"$P_\mathrm{variation}(k) / P_\mathrm{baseline}(k) - 1$")
plt.tight_layout()

# %% [markdown]
# Compare the P1D uncertainty, $\sigma_P = \sqrt{\mathrm{diag}(C)}$, for each
# variation relative to the baseline over the common k range.

# %%
fig, axes = plt.subplots(4, 3, figsize=(10, 10))
axes = axes.reshape(-1)

for iz, z in enumerate(baseline_data.z):
    axis = axes[iz]
    k_baseline = baseline_data.k_kms[iz]
    sigma_baseline = np.sqrt(np.diag(baseline_data.cov_Pk_kms[iz]))
    for label, variation_data in variations.items():
        k_variation = variation_data.k_kms[iz]
        mask = (k_baseline >= k_variation.min()) & (k_baseline <= k_variation.max())
        sigma_variation = np.interp(
            k_baseline[mask],
            k_variation,
            np.sqrt(np.diag(variation_data.cov_Pk_kms[iz])),
        )
        axis.plot(
            k_baseline[mask],
            sigma_variation / sigma_baseline[mask],
            label=label,
        )
    axis.set_title(rf"$z={z:.1f}$")
    axis.axhline(1, color="k", ls="--", alpha=0.4)

for axis in axes[len(baseline_data.z) :]:
    axis.set_axis_off()
axes[0].legend(loc="upper right")
fig.supxlabel(r"$k[\mathrm{km}^{-1}\,\mathrm{s}]$")
fig.supylabel(r"$\sigma_{P,\mathrm{variation}} / \sigma_{P,\mathrm{baseline}}$")
plt.tight_layout()

# %% [markdown]
# ## Covariance components
#
# For each measurement separately, show the statistical and every individual
# systematic contribution to $\sigma_P / P$. Each output figure contains one
# panel per redshift bin.

# %%
from cup1d.postprocessing.plotter import plot_cov
from cup1d.p1ds.observations.data_DESIY1 import set_p1d_filename

for label, args in {
    "Baseline (QMLE3)": baseline_args,
    "QMLE": qmle_args,
    "FFT3_dir": fft3_dir_args,
}.items():
    print(label)
    plot_cov(set_p1d_filename(args.data_label[0]))



# %%

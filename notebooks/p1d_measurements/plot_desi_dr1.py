# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Plots of P1D and cov of DESI-DR1
#
# We make multiple comparisons between QMLE and FFT measurements
#
# We also plot covariance matrix, and look at the contributions to P(k) in the files

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import set_P1D, set_emulator
# -

args = Args(emulator_label="CH24_mpgcen_gpr", training_set="Cabayol23")
emulator = set_emulator(
    emulator_label=args.emulator_label,
)
args.data_label = "DESIY1"
args.cov_syst_type = "red"

# +

folder = "/home/jchaves/Proyectos/projects/lya/data/DESI-DR1/"
# in NERSC
# /global/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-baseline/qmle_measurement/DataProducts/
# QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
# FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits

fname_qmle = folder + "/qmle_measurement/DataProducts/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
fname_qmle3 = folder + "/qmle_measurement/DataProducts/v3/desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
fname_fft = folder + "/fft_measurement/p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"

args.z_min = 2.1
args.z_max = 4.3
# args.z_max = 2.9

data = {}

args.p1d_fname=fname_qmle
data["qmle"] = set_P1D(
    args,
    true_cosmo=None,
    emulator=emulator,
    cull_data=False
)

args.p1d_fname= fname_qmle3
data["qmle3"] = set_P1D(
    args,
    true_cosmo=None,
    emulator=emulator,
    cull_data=False
)

args.p1d_fname = fname_fft
data["fft"] = set_P1D(
    args,
    true_cosmo=None,
    emulator=emulator,
    cull_data=False
)
# -

# ## Comparison of P1D

# +
fig, ax = plt.subplots(4, 3, figsize=(10, 8), sharey="row")
ax = ax.reshape(-1)

for iz in range(len(data["fft"].z)):
    zlab = str(np.round(data["fft"].z[iz], 2))
    col = "C"+str(iz)
    if iz == 0:
        lab1 = "FFT"
        lab2 = None
    elif iz == 1:
        lab1 = None
        lab2 = "QMLE3"
    kuse = data["qmle"].k_kms[iz].copy()
    y = np.interp(kuse, data["fft"].k_kms[iz], data["fft"].Pk_kms[iz])
    ax[iz].plot(kuse, y / data["qmle"].Pk_kms[iz]-1, ls="-", alpha=0.75, label=lab1)
    y = data["qmle3"].Pk_kms[iz]
    ax[iz].plot(kuse, y / data["qmle"].Pk_kms[iz]-1, ls="-", alpha=0.75, label=lab2)
    ax[iz].set_title(r"$z=$"+zlab)
    ax[iz].axhline(0, color="k", ls="--", alpha=0.4)
    ax[iz].axhline(0.01, color="k", ls=":", alpha=0.4)
    ax[iz].axhline(-0.01, color="k", ls=":", alpha=0.4)
    if iz < 2:
        ax[iz].legend(loc="upper right")
ax[-1].set_axis_off()
fig.supxlabel(r"$k[\mathrm{km}^{-1}\mathrm{s}]$")
fig.supylabel(r"$P_x(k)/P_\mathrm{QMLE}-1$")
plt.tight_layout()
# plt.savefig("figs/ratio_w_qmle.pdf")

# +
fig, ax = plt.subplots(4, 3, figsize=(12, 10))
ax = ax.reshape(-1)

for iz in range(len(data["fft"].z)):
    zlab = str(np.round(data["fft"].z[iz], 2))
    col = "C"+str(iz)
    if iz == 0:
        lab1 = "FFT"
        lab2 = None
    elif iz == 1:
        lab1 = None
        lab2 = "QMLE3"
    kuse = data["qmle"].k_kms[iz].copy()
    # y = np.interp(kuse, data["fft"].k_kms[iz], data["fft"].Pk_kms[iz])
    # ax[iz].plot(kuse, y / data["qmle"].Pk_kms[iz]-1, ls="-", alpha=0.75, label=lab1)
    y = data["qmle3"].Pk_kms[iz]
    y2 = np.sqrt(np.diag(data["qmle3"].cov_Pk_kms[iz]))
    # y2 = np.sqrt(np.diag(data["qmle3"].cov_Pk_kms[iz]) + np.diag(data["qmle"].cov_Pk_kms[iz]))
    ax[iz].errorbar(kuse, y / data["qmle"].Pk_kms[iz]-1, y2 / data["qmle"].Pk_kms[iz], ls="-")
    ax[iz].set_title(r"$z=$"+zlab)
    ax[iz].axhline(0, color="k", ls="--", alpha=0.4)
    ax[iz].axhline(0.01, color="k", ls=":", alpha=0.4)
    ax[iz].axhline(-0.01, color="k", ls=":", alpha=0.4)
    # if iz < 2:
    #     ax[iz].legend(loc="upper right")
ax[-1].set_axis_off()
fig.supxlabel(r"$k[\mathrm{km}^{-1}\mathrm{s}]$")
fig.supylabel(r"$P_\mathrm{QMLE3}(k)/P_\mathrm{QMLE}-1$")
plt.tight_layout()
# plt.savefig("figs/ratio_w_qmle.pdf")
# -

# ## Comparison cov matrix

# +
fig, ax = plt.subplots(4, 3, figsize=(12, 10))
ax = ax.reshape(-1)

for iz in range(len(data["fft"].z)):
    zlab = str(np.round(data["fft"].z[iz], 2))
    col = "C"+str(iz)
    if iz == 0:
        lab1 = "FFT"
        lab2 = None
    elif iz == 1:
        lab1 = None
        lab2 = "QMLE3"
    kuse = data["qmle"].k_kms[iz].copy()
    # y = np.interp(kuse, data["fft"].k_kms[iz], data["fft"].Pk_kms[iz])
    # ax[iz].plot(kuse, y / data["qmle"].Pk_kms[iz]-1, ls="-", alpha=0.75, label=lab1)
    y1 = np.sqrt(np.diag(data["qmle"].cov_Pk_kms[iz]))
    y2 = np.sqrt(np.diag(data["qmle3"].cov_Pk_kms[iz]))
    ax[iz].plot(kuse, y2 / y1, ls="-")
    ax[iz].set_title(r"$z=$"+zlab)
    ax[iz].axhline(1, color="k", ls="--", alpha=0.4)
    ax[iz].axhline(np.mean(y2/y1), color="C1", ls="-", alpha=0.4)
    # ax[iz].axhline(0.01, color="k", ls=":", alpha=0.4)
    # ax[iz].axhline(-0.01, color="k", ls=":", alpha=0.4)
    # if iz < 2:
    #     ax[iz].legend(loc="upper right")
ax[-1].set_axis_off()
fig.supxlabel(r"$k[\mathrm{km}^{-1}\mathrm{s}]$")
fig.supylabel(r"$\sigma_\mathrm{QMLE3}(k)/\sigma_\mathrm{QMLE}-1$")
plt.tight_layout()
plt.savefig("figs/ratio_sigma_qmle.pdf")
# -

# ## SNR

# +
fig, ax = plt.subplots(4, 3, figsize=(12, 10), sharey="row")
ax = ax.reshape(-1)

for iz in range(len(data["fft"].z)):
    zlab = str(np.round(data["fft"].z[iz], 2))
    col = "C"+str(iz)
    if iz == 0:
        lab1 = "QMLE"
        lab2 = None
        lab3 = None
    elif iz == 1:
        lab1 = None
        lab2 = "QMLE3"
        lab3 = None
    elif iz == 2:
        lab1 = None
        lab2 = None
        lab3 = "FFT"
    kuse = data["qmle"].k_kms[iz].copy()
    # y = np.interp(kuse, data["fft"].k_kms[iz], data["fft"].Pk_kms[iz])
    # ax[iz].plot(kuse, y / data["qmle"].Pk_kms[iz]-1, ls="-", alpha=0.75, label=lab1)
    y1 = data["qmle"].Pk_kms[iz]/np.sqrt(np.diag(data["qmle"].cov_Pk_kms[iz]))
    ax[iz].plot(kuse, y1, ls="-", label=lab1, alpha=0.8)
    ax[iz].axhline(np.mean(y1), color="C0", ls="--", alpha=0.4)
    
    y2 = data["qmle3"].Pk_kms[iz]/np.sqrt(np.diag(data["qmle3"].cov_Pk_kms[iz]))
    ax[iz].plot(kuse, y2, ls="-", label=lab2, alpha=0.8)
    ax[iz].axhline(np.mean(y2), color="C1", ls="--", alpha=0.4)
    
    y3 = data["fft"].Pk_kms[iz]/np.sqrt(np.diag(data["fft"].cov_Pk_kms[iz]))
    ax[iz].plot(data["fft"].k_kms[iz], y3, ls="-", label=lab3, alpha=0.8)
    ax[iz].axhline(np.mean(y3), color="C2", ls="--", alpha=0.4)
    
    ax[iz].set_title(r"$z=$"+zlab)
    # ax[iz].axhline(1, color="k", ls="--", alpha=0.4)
    # ax[iz].axhline(0.01, color="k", ls=":", alpha=0.4)
    # ax[iz].axhline(-0.01, color="k", ls=":", alpha=0.4)
    if iz < 3:
        ax[iz].legend(loc="upper right")
ax[-1].set_axis_off()
fig.supxlabel(r"$k[\mathrm{km}^{-1}\mathrm{s}]$")
fig.supylabel(r"SNR")
plt.tight_layout()
plt.savefig("figs/snr_all.pdf")
# -

# ## Covariance matrix

# +
from cup1d.likelihood.plotter import plot_cov

plot_cov(fname_qmle, save_directory='figs')


# -

# ## Contributions to QMLE P1D

# +

from astropy.io import fits


hdu = fits.open(fname_qmle)
_ = (hdu[1].data["Z"] == 2.2) & (hdu[1].data["K"] < 0.04)
plt.plot(hdu[1].data["K"][_], hdu[1].data["PLYA"][_])
plt.plot(hdu[1].data["K"][_], hdu[1].data["PRAW"][_])
plt.plot(hdu[1].data["K"][_], hdu[1].data["PNOISE"][_])
plt.plot(hdu[1].data["K"][_], hdu[1].data["ThetaP"][_])
plt.plot(hdu[1].data["K"][_], hdu[1].data["PRAW"][_] - hdu[1].data["PNOISE"][_])
plt.plot(hdu[1].data["K"][_], hdu[1].data["PFID"][_])
# -

# ## Syst to stat ratio

# +
hdu = fits.open(fname_qmle)

rat = np.diag(hdu[5].data)/np.diag(hdu[4].data)
zu = np.unique(hdu[1].data["Z"])
for zz in zu:
    _ = (hdu[1].data["Z"] == zz)
    plt.plot(hdu[1].data["K"][_], rat[_], label=str(zz))
plt.legend(ncol=3)
plt.xscale("log")
plt.yscale("log")
# -



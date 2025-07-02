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

# # Direct metal subtraction

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# +
# different contributions to FFT P1D

folder = "/home/jchaves/Proyectos/projects/lya/data/DESI-DR1/fft_measurement/"

p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7.fits"
hdu = fits.open(p1d_fname)

p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7_no_metal_corr.fits"
hdu_nometal = fits.open(p1d_fname)

p1d_fname = folder + "p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"
hdu_direct = fits.open(p1d_fname)
zuse = np.unique(hdu[1].data["Z"])

# for iz in range(len(zuse)):
for iz in range(1,2):

    _ = (hdu[1].data["Z"] == zuse[iz]) & (hdu[1].data["K"] < 0.03)
    p1d_no = hdu_nometal[1].data["PLYA"][_]
    p1d_fid = hdu[1].data["PLYA"][_]
    p1d_dir = hdu_direct[1].data["PLYA"][_]
    
    plt.plot(hdu[1].data["K"][_], p1d_fid/p1d_dir-1)
    plt.plot(hdu[1].data["K"][_], p1d_no/p1d_dir-1)
    
    # plt.plot(hdu[1].data["K"][_], p1d_fid)
    # plt.plot(hdu[1].data["K"][_], p1d_dir)
    # plt.plot(hdu[1].data["K"][_], p1d_no)
    
    # plt.plot(hdu[1].data["K"][_], p1d_no, label="no_metal_corr")
    # plt.plot(hdu[1].data["K"][_], p1d_dir/p1d_no-1, label="direct_metal_subtraction")
    # plt.plot(hdu[1].data["K"][_], (p1d_no - p1d_dir)/p1d_fid)
    # plt.plot(hdu[1].data["K"][_], p1d_no/p1d_fid)
    # plt.plot(hdu[1].data["K"][_], p1d_dir/p1d_fid)
    # y = p1d_no - p1d_yes
    # y = hdu[1].data["K"][_] * (p1d_no - p1d_yes)/np.pi
    # plt.plot(hdu[1].data["K"][_], y, label=str(z))
# plt.plot(hdu[1].data["K"][_], hdu[1].data["PRAW"][_])
# plt.plot(hdu[1].data["K"][_], hdu[1].data["PNOISE"][_])
plt.legend()
# plt.xscale("log")
# plt.yscale("log")
# plt.ylim(-0.05, 0.05)
# +
# from cup1d.nuisance.metal_correction import SB1_power

# folder = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/fft_measurement/"
# file_metal = folder + "param_fit_side_band_1_kms.pickle"
# Pk_cont = SB1_power(data["P1Ds"].z, data["P1Ds"].k_kms, file_metal)
# for iz in range(0, 10):
#     # fun = a * data["P1Ds"].k_kms[iz] ** (-b)
#     # fun2 = B1 * np.exp(-b1 * data["P1Ds"].k_kms[iz])
#     # fun3 = C1 * np.exp(-c1 * data["P1Ds"].k_kms[iz])
#     plt.plot(data["P1Ds"].k_kms[iz], Pk_cont[iz])
#     # plt.plot(data["P1Ds"].k_kms[iz], fun+fun2+fun3)
#     y = v1 * data["P1Ds"].k_kms[iz] ** -v2
#     plt.plot(data["P1Ds"].k_kms[iz], y, "k")
# # plt.plot(data["P1Ds"].k_kms[iz], fun, lw=3, color="k")


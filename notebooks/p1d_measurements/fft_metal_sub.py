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
# -



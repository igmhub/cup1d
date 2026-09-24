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
# # Noisy P1D mocks
#
# This notebook shows how cup1d creates noisy P1D mock measurements by drawing
# Gaussian realizations from their covariance matrices.

# %% [markdown]
# Import the current P1D mock classes and the YAML-based analysis helpers used
# to construct a Gadget mock.

# %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np

from cup1d.configuration import Args
from cup1d.emulator.archive import set_archive
from cup1d.inference import Analysis
from cup1d.p1ds.simulations.data_eBOSS_mock import P1D_eBOSS_mock
from cup1d.p1ds.simulations.data_gadget import Gadget_P1D

# %% [markdown]
# Load the fiducial eBOSS mock without perturbing its P1D values and display
# the measurement with its covariance-derived uncertainties.

# %%
eboss_fiducial = P1D_eBOSS_mock(add_noise=False)
eboss_fiducial.plot_p1d()

# %% [markdown]
# Draw one noisy eBOSS realization using a fixed seed. The covariance matrix is
# unchanged; only the measured P1D values are perturbed.

# %%
eboss_noisy = P1D_eBOSS_mock(add_noise=True, seed=0)
eboss_noisy.plot_p1d()

# %% [markdown]
# Construct the fiducial MP-Gadget mock with the current analysis theory and
# the MP-Gadget archive. The mock uses the Chabanier2019 covariance by default.

# %%
gadget_args = Args.from_baseline(verbose=False)
gadget_analysis = Analysis(gadget_args)
gadget_archive = set_archive(gadget_args.training_set)
gadget_testing_data = gadget_archive.get_testing_data("mpg_central")
gadget_fiducial = Gadget_P1D(
    theory=gadget_analysis.theory,
    testing_data=gadget_testing_data,
    input_sim="mpg_central",
    add_noise=False,
)
gadget_fiducial.plot_p1d()

# %% [markdown]
# Construct a noisy MP-Gadget realization with the same theory, simulation, and
# covariance settings as the fiducial mock.

# %%
gadget_noisy = Gadget_P1D(
    theory=gadget_analysis.theory,
    testing_data=gadget_testing_data,
    input_sim="mpg_central",
    add_noise=True,
    seed=0,
)
gadget_noisy.plot_p1d()

# %% [markdown]
# Draw many eBOSS realizations at one redshift bin and compare them with the
# fiducial P1D and its one-sigma covariance uncertainty.

# %%
number_of_samples = 50
redshift_index = 0
realizations = eboss_fiducial.get_Pk_iz_perturbed(
    eboss_fiducial.Pk_kms,
    eboss_fiducial.cov_Pk_kms,
    nsamples=number_of_samples,
    seed=0,
)

k_kms = eboss_fiducial.k_kms[redshift_index]
p1d = eboss_fiducial.Pk_kms[redshift_index]
sigma_p1d = np.sqrt(np.diag(eboss_fiducial.cov_Pk_kms[redshift_index]))

plt.errorbar(k_kms, p1d, sigma_p1d, fmt="o", label="fiducial")
for realization in realizations[redshift_index]:
    plt.plot(k_kms, realization, color="k", alpha=0.1)
plt.xlabel(r"$k\,[\mathrm{km}^{-1}\,\mathrm{s}]$")
plt.ylabel(r"$P_\mathrm{1D}(k)$")
plt.title(rf"$z={eboss_fiducial.z[redshift_index]:.1f}$")
plt.legend()
plt.tight_layout()

# %% [markdown]
# Verify that many noise realizations reproduce the fiducial mean and the
# covariance-derived standard deviation at the same redshift.

# %%
number_of_samples = 1000
realizations = eboss_fiducial.get_Pk_iz_perturbed(
    eboss_fiducial.Pk_kms,
    eboss_fiducial.cov_Pk_kms,
    nsamples=number_of_samples,
    seed=1,
)
mean_p1d = realizations[redshift_index].mean(axis=0)
measured_sigma_p1d = realizations[redshift_index].std(axis=0)

plt.plot(k_kms, mean_p1d / p1d - 1, label="realization mean / fiducial - 1")
plt.plot(
    k_kms,
    measured_sigma_p1d / sigma_p1d,
    label=r"realization $\sigma_P$ / covariance $\sigma_P$",
)
plt.axhline(0, color="k", ls="--", alpha=0.4)
plt.axhline(1, color="k", ls=":", alpha=0.4)
plt.xlabel(r"$k\,[\mathrm{km}^{-1}\,\mathrm{s}]$")
plt.title(rf"$z={eboss_fiducial.z[redshift_index]:.1f}$")
plt.legend()
plt.tight_layout()

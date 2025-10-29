# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parameterization of the HCD contamination
#
# Contamination from High Column Density (HCD) systems, following a model based on Rogers et al. (2018)

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
## Set default plot size, as normally its a bit too small
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.dpi'] = 120
from cup1d.nuisance import hcd_model_rogers_class

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %%
k_kms = np.logspace(-3, np.log10(0.04), 100)
labs = ["LLS", "subDLAs", "small\,DLAs", "large\,DLAs"]
ls = ["-", ":", "-.", "--"]
vals = [0.1, 3e-2, 1e-2, 1e-2]

fig, ax = plt.subplots(1, figsize=(8, 6))
ftsize = 20

for ii in range(4):

    coeffs = {
        "HCD_damp1":[0, -11.5],
        "HCD_damp2":[0, -11.5],
        "HCD_damp3":[0, -11.5],
        "HCD_damp4":[0, -11.5],
        "HCD_const":[0, 0],
    }

    val = vals[ii]
    coeffs["HCD_damp"+str(ii+1)] = [0, np.log(val)]
    
    hcd_model = hcd_model_rogers_class.HCD_Model_Rogers(coeffs = coeffs)
    cont = hcd_model.get_contamination(z=np.array([3]), k_kms=[k_kms])
    ax.plot(k_kms, cont, label=r"$f^\mathrm{HCD}_\mathrm{"+labs[ii]+"}=$"+str(val), ls=ls[ii], lw=2)

ax.tick_params(axis="both", which="major", labelsize=ftsize)
plt.axhline(1, color="k", ls=":")
plt.xscale("log")

plt.xlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
plt.ylabel("HCD contamination", fontsize=ftsize)

plt.legend(fontsize=ftsize)

plt.tight_layout()
plt.savefig("HCD_contamination.png")
plt.savefig("HCD_contamination.pdf")

# %%
k_kms = np.logspace(-3, np.log10(0.04), 100)
# labs = ["LLS", "subDLAs", "small\,DLAs", "large\,DLAs"]
labs = ["LLS", "LLS", "LLS", "LLS"]
add_labs = ["max prior", "best"] 
ls = ["-", ":", "-.", "--"]
vals = [-1.0, -1.25, -1.50]

fig, ax = plt.subplots(1, figsize=(8, 6))
ftsize = 18


for ii in range(2):

    val = vals[ii]

    coeffs = {
        "HCD_damp1":[0, val],
        "HCD_damp2":[0, -11.5],
        "HCD_damp3":[0, -11.5],
        "HCD_damp4":[0, -4.],
        "HCD_const":[0, 0],
    }

    # val = vals[ii]
    # coeffs["HCD_damp"+str(ii+1)] = [0, np.log(val)]
    
    hcd_model = hcd_model_rogers_class.HCD_Model_Rogers(coeffs = coeffs)
    cont = hcd_model.get_contamination(z=np.array([3]), k_kms=[k_kms])
    ax.plot(k_kms, cont, label=r"$f^\mathrm{HCD}_\mathrm{"+labs[ii]+"}=$"+str(val) + " ("+add_labs[ii]+")", ls=ls[ii], lw=2)

ax.tick_params(axis="both", which="major", labelsize=ftsize-2)
plt.axhline(1, color="k", ls=":")
plt.xscale("log")

plt.xlabel(r"$k\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
plt.ylabel("HCD contamination", fontsize=ftsize)

plt.legend(fontsize=ftsize-2)

plt.tight_layout()
plt.savefig("HCD_prior.png")
plt.savefig("HCD_prior.pdf")

# %%

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
# # Fig 6
#
# Illustrative example of contamination from High Column Density (HCD) systems, following a model based on Rogers et al. (2018)

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
from cup1d.contaminants import hcd_model_rogers_class

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### Fig. 6

# %%
k_kms = np.logspace(-3, np.log10(0.04), 100)
labs = ["LLS", "subDLAs", "small\,DLAs", "large\,DLAs"]
ls = ["-", ":", "-.", "--"]
vals = [0.1, 3e-2, 1e-2, 1e-2]

fig, ax = plt.subplots(1, figsize=(8, 6))
ftsize = 20

out_data = {}

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

    out_data["x"] = k_kms
    out_data["y"+str(ii)] = cont

ax.tick_params(axis="both", which="major", labelsize=ftsize)
plt.axhline(1, color="k", ls=":")
plt.xscale("log")

plt.xlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
plt.ylabel("HCD contamination", fontsize=ftsize)

plt.legend(fontsize=ftsize)

plt.tight_layout()
# plt.savefig("figs/HCD_contamination.png")
# plt.savefig("figs/HCD_contamination.pdf")

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_6.npy")
np.save(fname, out_data)

# %%

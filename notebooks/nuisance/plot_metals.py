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
from cup1d.nuisance import si_add, si_mult


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
# ### dv all metals

# %%
c_kms = 299792.458
metals = ["SiIIa_SiIIb", "Lya_SiIII", "SiIIb_SiIII",  
          "SiIIa_SiIII", "Lya_SiIIb", "Lya_SiIIa", 
          "CIVa_CIVb", "MgIIa-MgIIb", "Lya_SiIIc", "SiIIc_SiIII"
         ]

for metal_label in metals:

    if metal_label == "Lya_SiIII":
        lambda_rest = [1206.52, 1215.67]
    elif metal_label == "Lya_SiIIb":
        lambda_rest = [1193.28, 1215.67]
    elif metal_label == "Lya_SiIIa":
        lambda_rest = [1190.42, 1215.67]
    elif metal_label == "SiIIa_SiIIb":
        lambda_rest = [1190.42, 1193.28]  # SiIIa-SiIIb
    elif metal_label == "SiIIa_SiIII":
        lambda_rest = [1190.42, 1206.52]  # SiIIa-SiIII
    elif metal_label == "SiIIb_SiIII":
        lambda_rest = [1193.28, 1206.52]  # SiIIb-SiIII
    elif metal_label == "CIVa_CIVb":
        lambda_rest = [1548.20, 1550.78]  # CIV-CIV
    elif metal_label == "MgIIa-MgIIb":
        lambda_rest = [2795.53, 2802.70]  # MgII-MgII
    elif metal_label == "SiIIc_SiIII":
        lambda_rest = [1206.51, 1260.42]
    elif metal_label == "Lya_SiIIc":
        lambda_rest = [1215.67, 1260.42]
    else:
        print("NO", metal_label)

    dv = np.log(lambda_rest[1]/lambda_rest[0]) * c_kms
    # z=3
    # dk = 1 / dv * c_kms / np.mean(lambda_rest) / (1+z)
    # dk = 1/lambda_rest[0] / (np.exp(dv/c_kms)-1)
    print(metal_label, np.round(dv, 2), np.round(2 * np.pi/dv, 4))

# %%
# # dv1 = 6292.397594858781 
# # dv2 = 5573.0060260298
# # dv3 = 10837.387518476446
# dv1 = 3305.5345694089347 
# dv2 = 4024.926138237914
# plt.plot(k_kms, np.cos(dv1 * k_kms))
# plt.plot(k_kms, np.cos(dv2 * k_kms))
# # # plt.plot(k_kms, np.cos(dv3 * k_kms))
# plt.plot(k_kms, np.cos(dv1 * k_kms) + np.cos(dv2 * k_kms))
# # plt.plot(k_kms, np.cos(dv1 * k_kms) + np.cos(dv2 * k_kms) + np.cos(dv3 * k_kms))

# %%

# %%
k_kms = np.linspace(1e-3, 0.04, 1000)
labs = [r"Ly$\alpha$-SiIII", r"Ly$\alpha$-SiII", "SiIII-SiII", "SiII-SiII"]
# ls = ["-", ":", "-.", "--"]
vals = [0.11, 0.06, 0.03, 0.03]

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax = ax.reshape(-1)
ftsize = 20

remove = [
    {
        "SiIII_Lya": 1,
        "SiIIa_Lya": 0,
        "SiIIb_Lya": 0,
        "SiIII_SiIIa": 0,
        "SiIII_SiIIb": 0,
    },
    {
        "SiIII_Lya": 0,
        "SiIIa_Lya": 1,
        "SiIIb_Lya": 1,
        # "SiIIc_Lya": 1,
        "SiIII_SiIIa": 0,
        "SiIII_SiIIb": 0,
    },
    {
        "SiIII_Lya": 0,
        "SiIIa_Lya": 0,
        "SiIIb_Lya": 0,
        "SiIII_SiIIa": 1,
        "SiIII_SiIIb": 1,
        # "SiIII_SiIIc": 1,
    },
    {
        # "SiIIc_SiIIb": 1,
        # "SiIIc_SiIIa": 1,
        # "SiIIacbc": 1,
        # "SiIIacab": 1,
        # "SiIIbcab": 1,
    }
]
          

mF = np.array([0.75])

for ii in range(len(ax)):
# for ii in range(2, 3):

    if ii < 3:
        coeffs = {
            "f_Lya_SiIII": [0, -4],
            "s_Lya_SiIII": [0, 5],
            "f_Lya_SiII": [0, -3.5],
            "s_Lya_SiII": [0, 5],
            "f_SiIIa_SiIII": [0, 1],
            "f_SiIIb_SiIII": [0, 1],
        }
        met_model = si_mult.SiMult(coeffs = coeffs)
    else:
        coeffs = {
            "f_SiIIa_SiIIb": [0, -1],
            "s_SiIIa_SiIIb": [0, 4.5],
        }
        met_model = si_add.SiAdd(coeffs = coeffs)

    cont = met_model.get_contamination(z=np.array([3]), k_kms=[k_kms], mF=mF, remove=remove[ii])
    ax[ii].plot(k_kms, cont[0], label=labs[ii], ls="-", lw=1.5)

    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize-2)
    ax[ii].legend(fontsize=ftsize-2, loc="upper right")
    if ii < 3:
        ax[ii].axhline(1, color="k", ls=":")
    else:
        ax[ii].axhline(0, color="k", ls=":")

fig.supxlabel(r"$k_\parallel\,[\mathrm{km}^{-1} \mathrm{s}]$", fontsize=ftsize)
fig.supylabel("Metal contamination", fontsize=ftsize)



plt.tight_layout()
plt.savefig("metal_contamination.png")
plt.savefig("metal_contamination.pdf")

# %%
vals = [-4, 5, -3.5, 5, 1, 1, -1, 4.5]
for val in vals:
    print(np.exp(val))

# %%
1/148.4131591025766

# %%
1/90

# %%

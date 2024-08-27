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
# # Plot P1D measurement from Chabanier et al. (2019)

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 160
mpl.rcParams['figure.dpi'] = 160
from cup1d.p1ds import data_PD2013
from cup1d.p1ds import data_Chabanier2019

# %% jupyter={"outputs_hidden": false}
Cha2019=data_Chabanier2019.P1D_Chabanier2019(add_syst=True)
Cha2019.plot_p1d()

# %% [markdown]
# ### Compare to P1D from Palanque-Delabrouille et al. (2013)

# %% jupyter={"outputs_hidden": false}
PD2013=data_PD2013.P1D_PD2013(add_syst=True)
PD2013.plot_p1d()

# %% jupyter={"outputs_hidden": true}

# %%

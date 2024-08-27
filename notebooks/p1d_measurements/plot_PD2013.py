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
#     display_name: cup1d
#     language: python
#     name: cup1d
# ---

# %% [markdown]
# # Plot P1D measurement from Palanque-Delabrouille et al. (2013)

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.dpi'] = 120
from cup1d.data import data_PD2013

# %% [markdown]
# ### Read P1D from Palanque-Delabrouille et al. (2013)

# %% jupyter={"outputs_hidden": false}
PD2013=data_PD2013.P1D_PD2013()

# %% jupyter={"outputs_hidden": false}
PD2013.plot_p1d()

# %% jupyter={"outputs_hidden": true}

# %%

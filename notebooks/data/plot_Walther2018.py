# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Walther et al. (2018)
#
# https://ui.adsabs.harvard.edu/abs/2018ApJ...852...22W/abstract
#
# If using this, need to take a look at the function reading P1Ds in more detail

# %% jupyter={"outputs_hidden": false}
from cup1d.p1ds.observations import data_Walther2018

data = data_Walther2018.P1D_Walther2018()
data.plot_p1d(xlog=True)

# %%

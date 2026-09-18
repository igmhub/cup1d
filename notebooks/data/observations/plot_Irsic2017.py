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
# # Irsic et al. (2017)

# %% jupyter={"outputs_hidden": false}
from cup1d.p1ds.observations import data_Irsic2017

data = data_Irsic2017.P1D_Irsic2017()
data.plot_p1d()

# %%

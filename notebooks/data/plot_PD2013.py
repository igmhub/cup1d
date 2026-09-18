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
# # Palanque-Delabrouille et al. (2013)

# %% jupyter={"outputs_hidden": false}
from cup1d.p1ds.observations import data_PD2013

data = data_PD2013.P1D_PD2013()
data.plot_p1d()

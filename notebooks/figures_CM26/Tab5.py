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
# # Table 5
#
# With best-fitting results from all variations

# %%
from cup1d.plots_and_tables.table_variations import table_variations
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
table_variations(base)

# %%

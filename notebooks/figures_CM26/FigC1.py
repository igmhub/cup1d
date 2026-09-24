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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Figure C1
#
# Best-fitting constraints on IGM parameters using lace-lyssa

# %%
from cup1d.postprocessing.tables.igm import plot_table_igm

base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# save_fig = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/figs/test/"
store_data = plot_table_igm(base, name_variation="nyx", save_fig=None, chain="3", store_data=True)

# %%
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_C1.npy")
np.save(fname, store_data)

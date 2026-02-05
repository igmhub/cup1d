# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Figure C1
#
# Best-fitting constraints on IGM parameters using lace-lyssa

# +
from cup1d.plots_and_tables.plot_table_igm import plot_table_igm

base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# save_fig = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/figs/test/"
store_data = plot_table_igm(base, name_variation="nyx", save_fig=None, chain="3", store_data=True)

# +
import cup1d, os

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_C1.npy")
np.save(fname, store_data)

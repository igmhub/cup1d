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
# # Figure 1
#
# P1D from DESI DR1 data

# %%
# %load_ext autoreload
# %autoreload 2

from cup1d.configuration.args import Args
from cup1d.p1ds.factory import set_p1d
from cup1d.postprocessing.io import save_figure_data

# %%
args = Args.from_baseline()
data = set_p1d(args, args.data_label[0])

# %%
fig = data.plot_p1d()

# %%
save_figure_data(data.plot_p1d(store_data=True), "fig_1.npy")

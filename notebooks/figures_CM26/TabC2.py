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

# # Table C2
#
# Best-fitting nuisace parameters

# +
from cup1d.plots_and_tables.table_nuisance import table_nuisance

# my local machine
# folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
# nersc
folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"
table_nuisance(folder)
# -



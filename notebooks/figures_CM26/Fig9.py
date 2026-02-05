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

# # Figure 9
#
# Comparison between DESI DR1 data and best-fitting model to it

# +
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import cup1d
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
# -

args = Args(pre_defined="CM2026", system="local")
pip = Pipeline(args, out_folder=None)

# +
# my local machine
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/P1D_results/DESI_DR1/chain/"

data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# +
ylims=np.array([
    [0.85, 1.15],
    [0.85, 1.15],
    [0.7, 1.3],
    [0., 2.0],
])

out_data = pip.fitter.like.plot_p1d(
    p0, 
    residuals=True, 
    plot_panels=True, 
    print_chi2=False, 
    fix_cosmo=False,
    ylims=ylims, 
    plot_fname=None,
    # plot_fname="figs/residual_fid_opt_global",
    store_data=True
)
# -

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
fname = os.path.join(path_out, "fig_9.npy")
np.save(fname, store_data)



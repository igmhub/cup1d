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

# # Analyze the result of global fits
#
# Local fits are analyzed in compute_ic_at_a_time
#
# NEED to be updated, only showing plots after setting IC

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline
from cup1d.utils.utils import get_path_repo

# +
emu = "mpg"

args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="global_opt", fix_cosmo=True)
pip = Pipeline(args, out_folder=None)
# -

ndeg = 0
for ii in range(len(pip.fitter.like.data.k_kms)):
    ndeg += len(pip.fitter.like.data.k_kms[ii])
print(ndeg, ndeg - len(pip.fitter.like.free_param_names))

p0 = pip.fitter.like.sampling_point_from_parameters()
chi2 = pip.fitter.like.get_chi2(p0)
chi2

pip.fitter.like.theory.fid_cosmo['linP_params']

pip.fitter.like.plot_p1d(residuals=True, plot_panels=True, values=p0)

pip.fitter.like.plot_igm()

# +

pip.fitter.set_mle(p0, chi2)
# -

pip.fitter.like.plot_p1d(p0)

diru = None
plotter = Plotter(pip.fitter, save_directory=diru)
for zz in pip.fitter.like.data.z:
    plotter.plot_illustrate_contaminants_cum(p0, np.array([zz]))
plotter.plot_mle_cosmo()

# #### To unblind

# +
# pip.fitter.like.apply_unblinding(pip.fitter.mle_cosmo)
# -

zz = pip.fitter.like.data.z
plt.plot(zz, pip.fitter.like.theory.model_igm.models["F_model"].get_mean_flux(zz), "o--")

# #### Error from Hessian

# +
# # %%time
# Hessian
# err = fitter.like.get_error(fitter.mle_cube.copy())
# err

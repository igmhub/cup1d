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
# emu = "mpg"
emu = "nyx"
data_label = "DESIY1_QMLE3"
# data_label = "DESIY1_FFT_dir"

args = Args(data_label=data_label, emulator_label="CH24_"+emu+"cen_gpr")
args.set_baseline(fit_type="global_opt", fix_cosmo=True, P1D_type=data_label)
pip = Pipeline(args, out_folder=None)

ndeg = 0 
for ii in range(len(pip.fitter.like.data.k_kms)):
    ndeg += len(pip.fitter.like.data.k_kms[ii])
print(ndeg, ndeg - len(pip.fitter.like.free_param_names))

# -

try:
    data["P1Ds"].plot_igm()
except:
    print("Real data, no true IGM history")

# +
plot = False

if plot:
    fname = None
    # fname = "p1d_qmle3"
    # fname = "p1d_fftdir"    
    pip.fitter.like.data.plot_p1d(fname=fname)
    

# +
plot = False

if plot:
    fname = None
    # fname = "cov_to_pk_mpg_fftdir"
    # fname = "cov_to_pk_mpg_qmle3"
    pip.fitter.like.plot_cov_to_pk(save_directory="figs", fname=fname)

# +
plot = False

if plot:

    pip.fitter.like.plot_correlation_matrix()

# +
# pip.fitter.like.plot_hull_fid(like_params=pip.fitter.like.free_params)
# -

# p0 = pip.fitter.like.sampling_point_from_parameters()
# like_params = pip.fitter.like.parameters_from_sampling_point(p0)
# pip.fitter.like.plot_igm(cloud=True, free_params=like_params)
pip.fitter.like.plot_igm(cloud=True)
# pip.fitter.like.plot_igm()





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

# ### Use plotter

# +

# diru = "allz_snr3_cosmo_global"
# diru = "allz_snr3_cosmo_andreu2"
diru=None
plotter = Plotter(pip.fitter, save_directory=diru)
plotter.plot_p1d(plot_panels=True, residuals=True)

for zz in like.data.z:
    plotter.plot_illustrate_contaminants_cum(fitter.mle_cube.copy(), np.array([zz]))


plotter.plot_p1d(plot_panels=True, residuals=True)

plotter.plot_igm()
plotter.plot_p1d_errors()
plotter.plot_p1d(residuals=True)
plotter.plot_p1d(residuals=True, plot_panels=True)
plotter.plot_metal_cont(plot_data=True)

if args.fix_cosmo == False:
    plotter.plot_mle_cosmo()
plotter.plots_minimizer()

plotter.plot_metal_cont(plot_data=True)
plotter.plot_hcd_cont(plot_data=True)
# -



# #### Error from Hessian

# +
# # %%time
# Hessian
# err = fitter.like.get_error(fitter.mle_cube.copy())
# err
# -



# #### Rescale fiducial cosmology

# +
pstar = fitter.like.theory.fid_cosmo["cosmo"].get_linP_params()
print(pstar['Delta2_star'], pstar['n_star'])

target_params = {
    'Delta2_star': 0.42,
     'n_star': -2.33,
}

fitter.like.theory.rescale_fid_cosmo(target_params)

pstar = fitter.like.theory.fid_cosmo["cosmo"].get_linP_params()
print(pstar['Delta2_star'], pstar['n_star'])
# -



# #### Plot parameters at a function of z, important when multiple nodes

like_params = fitter.like.parameters_from_sampling_point(fitter.mle_cube)
fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z, like_params)
fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z, like_params)

# +
mask = np.arange(11)

like_params = fitter.like.parameters_from_sampling_point(fitter.mle_cube)

fold0 = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/allz_snr3_cosmo_global/"
folder = fold0 + "taueff"
oFmodel, ocFmodel = fitter.like.theory.model_igm.models["F_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "sigT"
oTmodel, ocTmodel = fitter.like.theory.model_igm.models["T_model"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Simult"
oSimult, ocSimult = fitter.like.theory.model_cont.metal_models["Si_mult"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "Siadd"
oSiadd, ocSiadd = fitter.like.theory.model_cont.metal_models["Si_add"].plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)
folder = fold0 + "HCD"
oHCD, ocHCD = fitter.like.theory.model_cont.hcd_model.plot_parameters(data["P1Ds"].z[mask], like_params, folder=folder)

models = [ocFmodel, ocTmodel, ocSimult, ocSiadd, ocHCD]
param_attime_all = {}
for mod in models:
    for key in mod:
        param_attime_all[key] = mod[key]
# -



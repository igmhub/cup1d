# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: sample data
#
# This notebook shows an illustrative example of how to run cup1d for extracting cosmological constraints from P1D data:
#
# - Set mock data
# - Set emulator
# - Set likelihood
# - Set sampler
# - Run sample for a small number of steps
#
# All these steps are implemented in cup1d/cup1d/likelihood/samplerpipeline.py. If you are interested in running cup1d, please take a look at cup1d/scripts/sam_sim.py. That script is parallelized using MPI and includes a bookkeeper taking care of all relevant options.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

# our own modules
from lace.cosmo import camb_cosmo
from lace.emulator.emulator_manager import set_emulator
from cup1d.likelihood import lya_theory, likelihood
from cup1d.likelihood.fitter import Fitter
from cup1d.likelihood.plotter import Plotter

from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline

from cup1d.p1ds.data_DESIY1 import P1D_DESIY1
from astropy.io import fits


from corner import corner

from cup1d.utils.utils import get_path_repo

from scipy.stats import chi2 as chi2_scipy
from cup1d.pipeline.set_archive import set_archive


# %%

base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"

from cup1d.plots_and_tables.table_nuisance import table_nuisance
table_nuisance(base + folder)

# from cup1d.plots_and_tables.table_variations import table_variations
# base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# table_variations(base)

# save_fig = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/figs/"

# from cup1d.plots_and_tables.plot_table_igm import plot_table_igm
# base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# plot_table_igm(base, name_variation=None, save_fig=save_fig)
# better from more IGM variation?

# %%
from cup1d.plots_and_tables.plots_corner import plots_chain

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"

variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "zmax": ["Data: $z \\leq 3.4$", "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"],
    "zmin": ["Data: $z \\geq 2.6$", "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE_mpg": ["Data: w/ low SNR", "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"],
    "DESIY1_FFT3_dir_mpg": ["Data: FFT", "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    # "DESIY1_FFT_dir_mpg": ["Data: FFT w/ low SNR", "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"],
    
    "no_emu_cov": ["Cov: w/o emu err", "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate": ["Cov: w/o 5\% err", "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"],
    "no_inflate_no_emu_cov": ["Cov: w/o emu, 5\% err", "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"],
    
    "DESIY1_QMLE3_nyx": ["Model: lace-lyssa", "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/"],
    
    "cosmo": ["Model: $\omega_0\omega_a$CDM", "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_high": ["Model: high $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"],
    "cosmo_low": ["Model: low $\Omega_\mathrm{M}h^2$", "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"],
    
    "more_igm": ["Model: IGM $n_z=8$", "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"],
    "less_igm": ["Model: IGM $n_z=4$", "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"],
    "Turner24": ["Model: $\\bar{F}\\, n_z=1$", "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"],
    
    "hcd_z": ["Model: HCD $n_z=2$", "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"],
    "dlas": ["Model: only DLAs", "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"],
    
    "metals_z": ["Model: metals $n_z=2$", "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"],
    "metal_trad": ["Model: trad metal", "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"],
    "metal_thin": ["Model: metal thin", "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"],
    "metal_deco": ["Model: no metal decorr", "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"],
    "metal_si2": ["Model: no SiII-SiII", "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"],
    
    "no_res": ["Model: no resolution", "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"],
    
    # "sim_mpg_central": ["Val: mpg-central simulation", "DESIY1_QMLE3/sim_mpg_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm": ["Val: mpg-central simulation only IGM", "DESIY1_QMLE3/sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"],
    # "sim_mpg_central_igm0": ["Val: mpg-central simulation only cosmo", "DESIY1_QMLE3/sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"],
    # "sim_nyx_central": ["Val: nyx-central simulation", "DESIY1_QMLE3/sim_nyx_central/CH24_mpgcen_gpr/chain_1/"],
    # "sim_sherwood": ["Val: sherwood simulation", "DESIY1_QMLE3/sim_sherwood/CH24_mpgcen_gpr/chain_1/"],
}
variations = {
    "DESIY1_QMLE3_mpg": ["Fiducial", "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"],
}

for ii, var in enumerate(variations):
    folder = os.path.join(base, variations[var][1])
    plots_chain(folder)

# %%

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
plots_chain(base + folder)
# chain = np.load(base + folder + "chain.npy")
# chain = 0
# blobs = np.load(base + folder + "blobs.npy")
# lnprob = np.load(base + folder + "lnprob.npy")

# mask, other = purge_chains(lnprob)

# for ii in range(other.shape[0]):
# # for ii in range(5):
#     # print(np.median(lnprob[:, other[ii]]))
#     plt.plot(lnprob[:, other[ii]])

# %%
# plt.hist2d(blobs["Delta2_star"].reshape(-1), blobs["n_star"].reshape(-1), bins=50);

# %% [markdown]
# ### Fisher

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"


data_label = "mock_DESIY1_QMLE3"
# data_label = "nyx_central"
# data_label = "nyx_seed"
# data_label = "nyx_cgan_base"
# data_label = "accel2"
# data_label = "sherwood"

if data_label == "mpg_central":
    zmin=2.25
    zmax=4.25
elif data_label == "nyx_central":
    zmin=2.2
    zmax=4.2
else:
    zmin=2.2
    zmax=4.2

zmin=2.25
zmax=4.25

true_cosmo_label = data_label
fid_cosmo_label = data_label
name_variation= "sim_" + data_label
# name_variation= "sim_" + data_label + "_igm"
# name_variation= "sim_" + data_label + "_igm0"
fit_type = "global_opt"

args = Args(
    data_label=data_label, 
    cov_label="DESIY1_QMLE3", 
    emulator_label=emulator_label,
    true_cosmo_label=true_cosmo_label,
    apply_smoothing=True,
    # add_noise=True,
    # seed_noise=0,
)
args.set_baseline(
    fit_type=fit_type, 
    fix_cosmo=False, 
    P1D_type="DESIY1_QMLE3",
    fid_cosmo_label=fid_cosmo_label,
    name_variation=name_variation,
    z_min=zmin,
    z_max=zmax,
    mcmc_conf="test"
)

# %% [markdown]
# ### Mocks

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"


data_label = "mpg_central"
# data_label = "nyx_central"
# data_label = "nyx_seed"
# data_label = "nyx_cgan_base"
# data_label = "accel2"
# data_label = "sherwood"

if data_label == "mpg_central":
    zmin=2.25
    zmax=4.25
elif data_label == "nyx_central":
    zmin=2.2
    zmax=4.2
else:
    zmin=2.2
    zmax=4.2

true_cosmo_label = data_label
fid_cosmo_label = data_label
name_variation= "sim_" + data_label
# name_variation= "sim_" + data_label + "_igm"
# name_variation= "sim_" + data_label + "_igm0"
fit_type = "global_opt"

args = Args(
    data_label=data_label, 
    cov_label="DESIY1_QMLE3", 
    emulator_label=emulator_label,
    true_cosmo_label=true_cosmo_label,
    apply_smoothing=True,
    # add_noise=True,
    # seed_noise=0,
)
args.set_baseline(
    fit_type=fit_type, 
    fix_cosmo=False, 
    P1D_type="DESIY1_QMLE3",
    fid_cosmo_label=fid_cosmo_label,
    name_variation=name_variation,
    z_min=zmin,
    z_max=zmax,
    mcmc_conf="test"
)

# %%
name_variation = "sim_1"
if (name_variation is not None) and name_variation.startswith("sim_"):
    print(name_variation)

# %%
# nyx_training_set = "models_Nyx_Sept2025_include_Nyx_fid_rseed"
# archive_mock = set_archive(training_set=nyx_training_set)
# pip = Pipeline(args, archive=archive_mock)
pip = Pipeline(args)

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
pip.fitter.like.get_chi2(p0)

# %%
for par in pip.fitter.like.free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %%
pip.fitter.like.plot_p1d()

# %%
pip.run_minimizer(p0, restart=True)

# %% [markdown]
# ### Data

# %%

# args = Args(data_label="DESIY1_QMLE3", emulator_label="CH24_nyxcen_gpr")
# args = Args(data_label="DESIY1_FFT_dir", emulator_label="CH24_nyxcen_gpr")
# args.set_baseline(fit_type="andreu2", fix_cosmo=True)

# args.set_baseline(fit_type="global_all", fix_cosmo=True)

# fid 0.18697989940945323
# HCD depend z, 0.13580926057347983
# same num IGM, new fid, 0.1755366851767883

# name_variation = None
# name_variation = "no_res"
# name_variation = "Turner24"
variations = [
    "fid",
    "no_inflate",  # no increase errors for 3, 3.6, and 4
    "all_inflate",
    "cosmo",  # different fiducial cosmo
    "metal_trad",  # 2 params for metals like eBOSS
    "metal_si2",  # no SiII-SiII cont
    "metal_deco",  # no decorrelation metals
    # "metal_thin",  # no desviation from optically-thin limit
    # "no_res",  # no resolution correction
    "Turner24",  # mF from Turner24 with 1 free param to scale
    "more_igm",  # 8 params for IGM evolution
    "less_igm",  # 4 params for IGM evolution
    # "metals_z",  # 2 params for z ev metals
    # "hcd_z",  # 2 params for z ev hcd
]

# metals_z 3.89548175069871

# name_variation = variations[12]
# name_variation = "metals_z"
# name_variation = "all_inflate"
# name_variation = "Turner24"

data_label = "DESIY1_QMLE3"
# data_label = "DESIY1_QMLE"
# data_label = "DESIY1_FFT3_dir"
name_variation = None
# name_variation = "no_inflate"
# name_variation = "no_emu_cov"
# name_variation = "no_inflate_no_emu_cov"

# name_variation = "metal_deco"
# name_variation = "metal_si2"
# name_variation = "no_res"
# name_variation = "HCD0"
# name_variation = "kF_kms"
# name_variation = "metal_thin"
# name_variation = "Gaikwad21"
# name_variation = "Gaikwad21T"
# name_variation = "Turner24"

# name_variation = "data_syst_diag"

# emu_cov_type = "block"
emu_cov_type = "diagonal"
# name_variation = "Ma2025"

# emulator_label="CH24_mpgcen_gpr"
emulator_label="CH24_nyxcen_gpr"

args = Args(data_label=data_label, emulator_label=emulator_label, emu_cov_type=emu_cov_type)
args.set_baseline(
    fit_type="global_opt", 
    fix_cosmo=False, 
    P1D_type=data_label, 
    name_variation=name_variation, 
)

pip = Pipeline(args, out_folder=args.out_folder)


# %%

pip.fitter.like.theory.model_cont.metal_models["Si_mult"].coeffs

# %%
# cov1
p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
data_lab = "DESIY1_QMLE3"
fit_type = "global_opt"
emu = "mpg"
folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_1/"
data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%

# %%
# base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
# chain = np.load(base + folder + "chain.npy")
# pip.fitter.like.plot_metal_cont_mult(chain=chain, save_directory="figs")
# pip.fitter.like.plot_metal_cont_add(free_params=free_params, chain=chain, save_directory="figs")
# pip.fitter.like.plot_hcd_cont(p0=p0, chain=chain, save_directory="figs")


# %%

# %%
# std_mpg = np.sqrt(np.diag(pip.fitter.like.emu_full_cov_Pk_kms)).copy()
# std_nyx = np.sqrt(np.diag(pip.fitter.like.emu_full_cov_Pk_kms)).copy()
# np.mean(std_nyx/std_mpg)

# %%
# pip.fitter.like.data.plot_p1d()

# pip.fitter.like.data.plot_p1d(fname="figs/p1d_qmle3")
# pip.fitter.like.plot_cov_to_pk(fname="figs/err2p1d_qmle")
# pip.fitter.like.plot_cov_to_pk()

# %%

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %%
p0 = pip.fitter.mle_cube
# p0 = data_best["mle_cube"].copy()
pip.fitter.like.get_chi2(p0)

# %%
new baseline, emu diagonal, no 5%, no opt thin, geom IGM
Passed out: 588.729654770444 (+37, 4 params less), 91
Almost out of bounds:
tau_eff_3 0.01525322049937576 -0.21413998755240043
gamma_0 0.9988588852833431 1.2642916038691219
gamma_3 7.277174836702373e-05 0.7636913741791961
kF_kms_0 0.016400906053116643 0.8210684569387425
kF_kms_3 0.96978692450308 1.2816956091087834
R_coeff_7 0.9995119157015389 0.019980476628061555
R_coeff_10 0.9600913519364411 0.018403654077457646
Delta2_star 0.40793
n_star -2.27768


new baseline, emu diagonal, no 5%, no opt thin, lin IGM
Passed out: 586.339000087996
Almost out of bounds:
tau_eff_3 0.0004793877378067514 -0.22071364914121383
gamma_0 0.9859609687718103 1.257827056674585
gamma_3 0.00023703338688466557 0.763773703531654
kF_kms_0 0.017705398759685742 0.8216987208075481
R_coeff_7 0.9971686113140668 0.01988674445256267
R_coeff_10 1.0 0.02
Delta2_star 0.43053
n_star -2.26356

new baseline, 
586.4304163703927
Delta2_star 0.42289
n_star -2.27332

# %%
# pip.fitter.mle

# %%
# pip.fitter.like.theory.model_cont.metal_models["Si_add"].coeffs

# %%
1.0832499884936857e-05 * 100

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)
# pip.fitter.like.plot_cov_to_pk(fname="figs/nyx_err2p1d_qmle3")

# %%
pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)
# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False, fix_cosmo=False, plot_fname="figs/residual_fid_opt_global")

# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)

# %%
len(pip.fitter.like.free_params)

# %%
# pip.fitter.like.plot_p1d(p0, residuals=True, plot_panels=True, print_chi2=False)

# %%
npoints = 0
for ii in range(len(pip.fitter.like.data.z)):
    npoints += len(pip.fitter.like.data.k_kms[ii])
npoints - len(pip.fitter.like.free_param_names)

# %%
# data_lab = "DESIY1_QMLE3"
# fit_type = "global_opt"
# emu = "mpg"
# folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
# data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()


# ii = 556
# type_prof = "prof_2d_deep"
# data_best = np.load(folder + type_prof + "/profile_"+str(ii)+ ".npy", allow_pickle=True).item()

# data_best.keys()

# %%

tar_cosmo = pip.fitter.like.apply_unblinding(data_best["blind_cosmo"])
pip.fitter.like.theory.rescale_fid_cosmo(tar_cosmo)

# %%
p0 = data_cen["mle_cube"].copy()
# p0 = data_best["mle_cube"].copy()
pip.fitter.like.get_chi2(p0)

# %%

# %%
# from cup1d.likelihood.cosmologies import set_cosmo

# cosmos = set_cosmo("mpg_central", return_all=True)

# p1 = np.zeros(len(cosmos))
# p2 = np.zeros(len(cosmos))
# for ii, key in enumerate(cosmos):
#     p1[ii] = cosmos[key]["star_params"]['Delta2_star']
#     p2[ii] = cosmos[key]["star_params"]['n_star']

# plt.scatter(p1, p2)

# print(p1.min(), p1.max())
# print(p2.min(), p2.max())

# %%

# %% [markdown]
# # OLD

# %% [markdown]
# ### Set emulator

# %% [markdown]
# #### Set either mock data or real data

# %%
# for forecast, just start label of observational data with mock
# choose_forecast = True 
choose_forecast = False
# to analyze data from simulations
choose_mock = False
# choose_mock = True
# to analyze data from observations
choose_data = False
# to analyze data from mock challenge
choose_challenge = False
# to analyze data from desiy1
choose_desiy1 = True

if choose_forecast:
    args.data_label_hires = None
    # for forecast, just start label of observational data with mock
    # args.data_label = "mock_Chabanier2019"
    # args.data_label="mock_Karacayli2024"
    # args.data_label_hires = "mock_Karacayli2022"
    args.data_label="mock_challenge_DESIY1"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits"
    # args.p1d_fname="/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    args.p1d_fname=folder + fname

    # # you need to provide true cosmology, IGM history, and contaminants
    # true_cosmo = set_cosmo(cosmo_label="nyx_central")
    # args.true_igm_label="nyx_central"
    # true_sim = "nyx_seed"
    true_sim = "nyx_seed_val"
    # true_sim = "sherwood"
    # true_sim = "accel2"
    # true_sim = "nyx_central"
    # true_sim = "mpg_central"
    # true_sim_cosmo = "mpg_central"
    args.mF_model_type="chunks"
    true_cosmo = set_cosmo(cosmo_label=true_sim)
    args.true_label_mF=true_sim
    args.true_label_T=true_sim
    
    # true_sim = "nyx_central"
    args.true_label_kF=true_sim
    
    # true_sim_cosmo = "Planck18_low"
    # args.true_label_kF="kF_both"
    
    # true_cosmo = set_cosmo(cosmo_label="mpg_22")
    # true_cosmo = set_cosmo(cosmo_label="Planck18")
    # args.true_igm_label="nyx_central"
    # args.true_igm_label="mpg_22"
    # from -11 to -4
    # args.true_SiIII=[[0, 0], [-10, -10]]
    # args.true_SiII=[[0, 0], [-10, -10]]
    # # from -7 to 0
    # args.true_HCD=[0, -6]
    # # from -5 to 2
    # args.true_SN=[0, -4]
    # # from -5 to 1.5
    # args.true_AGN=[0, -5]
    args.z_min = 2.1
    args.z_max = 4.3
elif choose_mock:    
    # true_cosmo=None
    # to analyze data from simulations
    true_sim = "nyx_central"
    # true_sim = "nyx_seed"
    # true_sim = "nyx_central"
    # args.data_label = "mock_DESIY1"
    args.true_cosmo_label=true_sim
    args.true_label_mF=true_sim
    # args.mF_model_type="pivot"
    args.mF_model_type="chunks"
    args.true_label_T=true_sim
    args.true_label_kF=true_sim
    # args.data_label="nyx_seed"
    # args.data_label_hires="mpg_central"
    args.data_label_hires = None
    # args.apply_smoothing=True
    args.apply_smoothing=False

    # args.cov_label = "DESIY1"
    version = "9fx"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-0."+version+"/"
    fname = "mockchallenge-0."+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_Sherwood_2048_40.fits.gz"
    args.cov_fname = folder + fname
    # args.p1d_fname = folder + fname
    # args.p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/cup1d/obs/p1d_fft_y1_measurement_kms_v3.fits"

    # provide cosmology only to cull the data
    # args.true_cosmo_label="mpg_central"
    # args.true_cosmo_label="nyx_central"
    true_cosmo = set_cosmo(cosmo_label=true_sim)
    # args.true_cosmo_label="nyx_seed"

    # you may provide contaminants
    # from 1 to 6, -11 to -4
    # args.true_SiIII=[[0, 0], [2, -10]]
    # args.true_SiII=[[0, 0], [2, -10]]
    # # from -5 to 0
    # args.true_HCD=[0, -4]
    # # from -5 to 2
    # args.true_SN=[0, -4]
    # args.true_AGN=[0, -5]
    args.z_max=4.3
elif choose_data:    
    true_cosmo=None
    args.data_label = "Chabanier2019"
    # args.data_label="Karacayli2024"
    args.data_label_hires = "Karacayli2022"
    args.z_max = 3.9
elif choose_challenge:
    args.data_label = "challenge_DESIY1"
    # version = "1.1qh"
    # version = "1.9fsh"
    version = "1.10qsh"
    folder = "/home/jchaves/Proyectos/projects/lya/data/mock_challenge/MockChallengeSnapshot/mockchallenge-"+version+"/"
    fname = "mockchallenge-"+version+"_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-"+version+"_fsiiii1.3e-03_fsiii1.1e-03_nonoise_fiducial.fits.gz"
    # fname = "mockchallenge-"+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_bar_ic_grid_3.fits.gz"
    # fname = "mockchallenge-0."+version+"_noise-42-0_fiducial.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_ACCEL2_6144_160.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_CGAN_4096_base.fits.gz"
    # fname = "mockchallenge-0."+version+"_nonoise_Sherwood_2048_40.fits.gz"
    args.p1d_fname = folder + fname
    if "fiducial" in args.p1d_fname:
        true_sim_label = "nyx_central"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "CGAN" in args.p1d_fname:
        true_sim_label = "nyx_seed"        
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "grid_3" in args.p1d_fname:
        true_sim_label = "nyx_3"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF=true_sim_label
    elif "Sherwood_2048_40" in args.p1d_fname:
        true_sim_label = "Sherwood_2048_40"
        args.true_label_mF=true_sim_label
        args.true_label_T="nyx_central"
        args.true_label_kF="nyx_central"
    elif "ACCEL2_6144_160" in args.p1d_fname:
        true_sim_label = "ACCEL2_6144_160"
        args.true_label_mF=true_sim_label
        args.true_label_T=true_sim_label
        args.true_label_kF="nyx_central"
    else:
        true_sim_label = None
    # true_sim_label = "nyx_central"

    true_cosmo = set_cosmo(cosmo_label=true_sim_label)
    
    args.z_min = 2.1
    # args.z_max = 4.3
    args.z_max = 2.7
    # args.z_min = 2.8
    # args.z_max = 3.2
elif choose_desiy1:
    true_cosmo = None
    args.true_igm_label= None
    args.data_label = "DESIY1"
    # args.cov_syst_type = "xred"
    # args.cov_syst_type = "fid"
    args.cov_syst_type = "red"
    folder = "/home/jchaves/Proyectos/projects/lya/data/DESI-DR1/"
    # in NERSC
    # /global/cfs/cdirs/desicollab/science/lya/y1-p1d/iron-baseline/qmle_measurement/DataProducts/
    # QMLE /global/cfs/cdirs/desicollab/users/naimgk/my-reductions/data/iron-v3/DataProducts/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate.fits
    # FFT /global/cfs/cdirs/desi/science/lya/y1-p1d/fft_measurement/v0/plots/baseline/notebook/measurement/p1d_fft_y1_measurement_kms.fits
    
    # args.p1d_fname=folder + "/qmle_measurement/DataProducts/v3/desi_y1_baseline_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    args.p1d_fname= folder + "/qmle_measurement/DataProducts/v3/desi_y1_snr3_p1d_sb1subt_qmle_power_estimate_contcorr_v3.fits"
    
    # args.p1d_fname = folder + "/fft_measurement/p1d_fft_y1_measurement_kms_v7_direct_metal_subtraction.fits"
    
    args.z_min = 2.1
    # args.z_min = 3.5
    args.z_max = 4.3
    # args.z_max = 2.7
    # args.z_max = 3.3
    # args.z_max = 2.9

# you do not need to provide the archive for obs data 
data = {"P1Ds": None, "extra_P1Ds": None}

data["P1Ds"] = set_P1D(
    args,
    archive=archive,
    true_cosmo=true_cosmo,
    emulator=emulator,
    cull_data=False
)
if args.data_label_hires is not None:
    data["extra_P1Ds"] = set_P1D(
        args,
        archive=archive,
        true_cosmo=true_cosmo,
        emulator=emulator,
        cull_data=False
    )

# %%

# %% [markdown]
# ## Read results from chains

# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/fid/chain_5/2.2/"
folder_priors = "/home/jchaves/Proyectos/projects/lya/data/obs/QMLE3/CH24_mpgcen_gpr/priors/chain_3/2.2/"
file = "fitter_results.npy"
# sampler_data = np.load(folder + file, allow_pickle=True).item()

# %%
plotter = Plotter(fname_chain=folder_data + file, fname_priors=folder_priors + file)
# plotter.fitter.like.plot_p1d(residuals=True)

# %% [markdown]
# ### Normal corner

# %%
# plotter.fitter.like.plot_p1d()
# plotter.save_directory = folder_data
# plotter.save_directory = None
# plotter.plot_corner()

# %% [markdown]
# ### Corner natural units

# %%
plotter.save_directory = folder_data

only_plot = [
    '$\\mathrm{ln}\\,\\tau_0$',
    '$\\mathrm{ln}\\,\\sigma^T_0$',
    '$\\mathrm{ln}\\,\\gamma_0$',
    '$\\mathrm{ln}\\,k^F_0$',
    '$\\mathrm{R}_0$'
]
plotter.plot_corner_1z_natural(2.2, only_plot=only_plot)

# %%

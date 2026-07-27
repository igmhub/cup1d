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

from astropy.io import fits


from corner import corner

from cup1d.utils.utils import get_path_repo

from scipy.stats import chi2 as chi2_scipy
from cup1d.pipeline.set_archive import set_archive


# %%
# data_label = ["DESIY1_QMLE3"]
data_label = ["DESIY1_QMLE3", "Karacayli2022"]
# data_label = ["DESIY1_QMLE3", "Karacayli2022", "Walther2018"]
# emulator_label = "CH24_mpgcen_gpr"
emulator_label = "forest_mpg"
name_variation = "no_res"

args = Args(
    data_label=data_label,
    emulator_label=emulator_label,
)

args.set_baseline(
    fit_type="global_opt",
    fix_cosmo=False,
    P1D_type=data_label,
    name_variation=name_variation,
)

pip = Pipeline(args)

# %%
p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.fitter.like.get_chi2(p0)

# %%
# %%time
for ii in range(10):
    pip.fitter.like.get_chi2(p0)

# %%
# %%time
for ii in range(10):
    p0[0] += 0.0001
    pip.fitter.like.get_chi2(p0)

# %%
8 times faster lace-mpg than forest-mpg

736/91.7

# %%
p0 = np.array([
    0.44370891, 0.41896411, 0.61988151, 0.40441576, 0.45867814,
       0.01096454, 0.67756427, 0.10833635, 0.08650839, 0.72379107,
       0.99996769, 0.69281722, 0.64864356, 0.00276913, 0.13844165,
       0.006074  , 0.30521981, 0.85987199, 0.45887747, 0.52643935,
       0.58099397, 0.57841591, 0.58993933, 0.31664493, 0.7344514 ,
       0.42520576, 0.56215998, 0.66432898, 0.59622537, 0.6936141 ,
       0.46279051, 0.39349512, 0.26083371, 0.73383231, 0.84770478,
       0.80593018, 0.17142966, 0.76663517, 0.54851977, 0.32703703,
       0.57129636, 0.62593543
])

# %%
# pip.fitter.like.theory.emulator.list_sim_cube

# %%
# np.diag(pip.fitter.like.full_icov_Pk_kms["Karacayli2022"])

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%
21.961857252395838

# %%
pip.run_minimizer(p0)
p0 = pip.fitter.mle_cube

# %%
DESI DR1 lace-mpg
7 s per step of the fit
fit prob 12.620160058817945
Delta2_star 0.44641
n_star -2.31264
alpha_star -0.21804
Almost out of bounds:
tau_eff_3 6.554882735132123e-05 -0.22089778801658366
gamma_2 0.9923529975743228 1.2610307967380079
gamma_3 0.0009226403192475971 0.7641173356498491
kF_kms_3 0.9776768775681337 1.2855076291518504

DESI DR1 forest-mpg
74 s per step of the fit (from 200 s before)
fit prob 24.215951076049226
Delta2_star 0.37374
n_star -2.31357
alpha_star -0.21804
Almost out of bounds:
tau_eff_3 0.010964543875754372 -0.2160482471924363
gamma_0 0.9999676876608057 1.2648473452008409
gamma_3 0.0027691300887306815 0.7650428122761227
kF_kms_1 0.006074003117882781 0.8160790278955864

DESI DR1 and high-res data forest-mpg
100 s per step of the fit
fit prob 1.888494208095926e-05
Delta2_star 0.39362
n_star -2.25305
alpha_star -0.21804
Almost out of bounds:
tau_eff_3 0.00030944217976317304 -0.22078926693147383
gamma_0 0.9997974400248681 1.2647620156147568
gamma_3 0.0007619700162554788 0.7640368063057861
kF_kms_0 0.0028824828434299483 0.8145370492356683
kF_kms_1 0.002099675278623822 0.814158836830309

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%
# kmax_kms = 0.05
p0 = np.array(
    [
        2.75118584e-01,
        5.17774010e-01,
        5.30625889e-01,
        1.83371658e-01,
        3.14888265e-01,
        2.56233809e-04,
        4.49190221e-01,
        8.08802364e-01,
        2.75376246e-01,
        7.60464392e-01,
        7.80380692e-01,
        2.13933769e-01,
        9.76238948e-01,
        2.12625196e-01,
        8.36634280e-03,
        6.11667700e-04,
        4.83328967e-01,
        4.62507818e-01,
        4.42825501e-01,
        5.07518881e-01,
        5.77167635e-01,
        5.82047638e-01,
        5.73636776e-01,
        2.43226185e-01,
        7.31152406e-01,
        3.84519394e-01,
        5.72142838e-01,
        6.79952839e-01,
        5.84233785e-01,
        7.22705904e-01,
        4.60296343e-01,
        3.86725805e-01,
        2.67293134e-01,
        7.58941796e-01,
        9.45021016e-01,
        9.02367918e-01,
        2.21158462e-01,
        6.56859007e-01,
        8.39572131e-02,
        1.92424581e-01,
        6.38898130e-01,
        6.63939405e-01,
    ]
)

# kmax_kms = 0.1
p1 = np.array(
    [
        7.33426056e-01,
        5.34725468e-01,
        3.21560156e-01,
        1.05118655e-01,
        9.22835705e-05,
        3.58845416e-06,
        6.17809858e-01,
        9.99986596e-01,
        9.99973173e-01,
        9.99879839e-01,
        8.49863134e-01,
        5.03862818e-01,
        5.42119000e-01,
        7.11304802e-01,
        5.86267647e-04,
        1.09726365e-05,
        2.83928640e-01,
        7.58647865e-06,
        4.26733863e-01,
        4.45587879e-01,
        5.78405300e-01,
        5.40943959e-01,
        5.38033093e-01,
        5.58162278e-01,
        7.12800869e-01,
        7.68411451e-01,
        5.68971975e-01,
        7.87179192e-01,
        5.83447873e-01,
        8.16674457e-01,
        4.46660740e-01,
        5.24128245e-01,
        2.86619375e-01,
        5.26975148e-01,
        9.15807378e-01,
        7.87923616e-01,
        7.19038867e-01,
        8.13452558e-01,
        7.00610726e-02,
        5.34434791e-02,
        6.25417538e-01,
        1.42502425e-02,
    ]
)

# forestflow DESI DR1
p0 = np.array(
    [
        0.44370891,
        0.41896411,
        0.61988151,
        0.40441576,
        0.45867814,
        0.01096454,
        0.67756427,
        0.10833635,
        0.08650839,
        0.72379107,
        0.99996769,
        0.69281722,
        0.64864356,
        0.00276913,
        0.13844165,
        0.006074,
        0.30521981,
        0.85987199,
        0.45887747,
        0.52643935,
        0.58099397,
        0.57841591,
        0.58993933,
        0.31664493,
        0.7344514,
        0.42520576,
        0.56215998,
        0.66432898,
        0.59622537,
        0.6936141,
        0.46279051,
        0.39349512,
        0.26083371,
        0.73383231,
        0.84770478,
        0.80593018,
        0.17142966,
        0.76663517,
        0.54851977,
        0.32703703,
        0.57129636,
        0.62593543,
    ]
)


# forestflow DESI DR1 + high-res
p0 = np.array(
    [
        3.61298262e-01,
        5.22894209e-01,
        4.94989807e-01,
        2.98809478e-01,
        3.04978420e-01,
        3.09442180e-04,
        9.87653865e-02,
        4.09004424e-01,
        4.59962792e-01,
        7.14441761e-01,
        9.99797440e-01,
        6.83919757e-01,
        7.91199875e-01,
        7.61970016e-04,
        2.88248284e-03,
        2.09967528e-03,
        4.97778270e-01,
        6.16933795e-02,
        4.41407264e-01,
        5.25453133e-01,
        5.74959555e-01,
        5.88260321e-01,
        5.77492490e-01,
        4.77830790e-01,
        7.36140573e-01,
        7.41602839e-01,
        5.68842364e-01,
        6.92868172e-01,
        5.78514223e-01,
        7.69904875e-01,
        4.58013030e-01,
        3.67787791e-01,
        2.63787399e-01,
        6.31085609e-01,
        8.89951239e-01,
        6.43353105e-01,
        7.22908393e-01,
        8.51732076e-01,
        1.17517034e-01,
        2.83270697e-01,
        6.10083973e-01,
        5.61151135e-01,
    ]
)

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False, residuals=True)

# %%
pip.fitter.mle_cosmo

# %%
# Name, Box Mpc, Resolution kpc; h=67.5
# MP-Gadget, 67.5, 87.9 # our suite
# l160_r25, 237, 37 # baseline
# l160_r50, 237, 74 # baseline, lower res 
# l320_r50, 474, 74 # bigger box, lower res

# l160_r25 and l160_r50 change in resolution
# l160_r50 and l320_r50 change in boxside

folder = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid"
sim_label = "l160_r25"


out_dict = load_data(folder, sim_label)

# load all axes, compute average of p1d and p3d

# P1D and P3D averages for fitting Arinyo parameters
# P1D average for generating mock z = 2, 2.6, 3, 3.6, 4, 5 (z=5 not validated?)

# %% [markdown]
# ### Fisher forecast analysis

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"


# data_label = "mock_DESIY1_QMLE3"
# data_label = "nyx_central"
# data_label = "nyx_seed"
# data_label = "nyx_cgan_base"
# data_label = "accel2"
# data_label = "sherwood"

# data_label = "mpg_central"
# data_label = "mpg_seed"
# data_label = "nyx_seed"

# if data_label == "mpg_central":
#     zmin=2.2
#     zmax=4.2
# elif data_label == "nyx_central":
#     zmin=2.2
#     zmax=4.2
# else:
#     zmin=2.2
#     zmax=4.2

cov_label="DESIY1_QMLE3"
# true_cosmo_label = data_label
# fid_cosmo_label = data_label
data_label = "forecast_DESIY1_QMLE3"
# name_variation= "sim_" + data_label
# name_variation= "sim_" + data_label + "_igm"
# name_variation= "sim_" + data_label + "_igm0"
# fit_type = "global_opt"
# name_variation = None

# args = Args(
#     data_label=data_label,
#     cov_label=cov_label,
#     emulator_label=emulator_label,
#     true_cosmo_label=true_cosmo_label,
#     apply_smoothing=True,
#     add_noise=False,
#     seed_noise=0,
#     emu_cov_type="full",
# )


# args.set_baseline(
#     fit_type=fit_type,
#     fix_cosmo=False,
#     fid_cosmo_label=fid_cosmo_label,
#     P1D_type=cov_label,
#     name_variation=name_variation,
#     z_min=zmin,
#     z_max=zmax,
#     mcmc_conf="explore",
# )

args = Args(
    data_label=data_label,
    emulator_label=emulator_label,
)

args.set_baseline(
    fit_type="global_opt",
    fix_cosmo=False,
    P1D_type=data_label,
)

pip = Pipeline(args)

# %%
pip.fitter.like.plot_p1d()

# %%

# archive_mock = set_archive(training_set="Pedersen21")
# dat = archive_mock.get_testing_data("mpg_central")

# %% [markdown]
# ### Mock analysis

# %%
emulator_label = "CH24_mpgcen_gpr"
# emulator_label = "CH24_nyxcen_gpr"


# data_label = "mock_DESIY1_QMLE3"
# data_label = "nyx_central"
# data_label = "nyx_seed"
# data_label = "nyx_cgan_base"
data_label = "accel2"
# data_label = "sherwood"

# data_label = "mpg_central"
# data_label = "mpg_seed"
# data_label = "nyx_seed"

if data_label == "mpg_central":
    zmin=2.2
    zmax=4.2
elif data_label == "nyx_central":
    zmin=2.2
    zmax=4.2
else:
    zmin=2.2
    zmax=4.2

cov_label="DESIY1_QMLE3"
true_cosmo_label = data_label
fid_cosmo_label = data_label
name_variation= "sim_" + data_label
# name_variation= "sim_" + data_label + "_igm"
# name_variation= "sim_" + data_label + "_igm0"
fit_type = "global_opt"
# name_variation = None

args = Args(
    data_label=data_label,
    cov_label=cov_label,
    emulator_label=emulator_label,
    true_cosmo_label=data_label,
    apply_smoothing=True,
    add_noise=False,
    seed_noise=0,
    emu_cov_type="full",
)

args.path_data = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid"

args.set_baseline(
    fit_type=fit_type,
    fix_cosmo=False,
    fid_cosmo_label=data_label,
    P1D_type=cov_label,
    name_variation=name_variation,
    z_min=zmin,
    z_max=zmax,
    mcmc_conf="explore",
)


# nyx_training_set = "models_Nyx_Sept2025_include_Nyx_fid_rseed"
# archive_mock = set_archive(training_set=nyx_training_set)
# pip = Pipeline(args, archive=archive_mock)
pip = Pipeline(args)

# %%
pip.fitter.like.plot_p1d()

# %%
# dict_out = {
#     "k_kms": pip.fitter.like.data.k_kms,
#     "Pk_kms": pip.fitter.like.data.Pk_kms,
#     "cov_Pk_kms": pip.fitter.like.data.cov_Pk_kms,
#     "z": pip.fitter.like.data.z,
# }
# np.save("smooth_" + data_label + ".npy", dict_out)

# data = np.load("smooth_" + data_label + ".npy", allow_pickle=True).item()
# data.keys()

# %%
p0 = pip.fitter.like.sampling_point_from_parameters()
p0[:] = 0.5
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
for par in free_params:
    print(par.name, par.value, par.min_value, par.max_value)

# %%
pip.fitter.like.plot_p1d(p0)

# %%
pip.run_minimizer(p0, restart=True)

# %%
pip.fitter.like.plot_p1d(pip.fitter.mle_cube)

# %%
# pip.fitter.mle

# %%

# %% [markdown]
# ### Data analysis

# %%
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

# name_variation = "metals_z"
# name_variation = "all_inflate"
# name_variation = "Turner24"
# name_variation = None
# name_variation = "no_inflate"
# name_variation = "no_emu_cov"
# name_variation = "no_inflate_no_emu_cov"

# name_variation = "metal_deco"
# name_variation = "metal_si2"
# name_variation = "no_res"
# name_variation = "HCD0"
# name_variation = "kF_kms"
# name_variation = "Gaikwad21"
# name_variation = "Gaikwad21T"
# name_variation = "Turner24"

# name_variation = "data_syst_diag"

# emu_cov_type = "block"
# emu_cov_type = "diagonal"
# name_variation = "Metals_Ma2025"
# name_variation = "HCD_BOSS"

# name_variation = "more_igm"
# name_variation = "LLS_nz4"
# name_variation = "IGM_priors"
# name_variation = "bias_eBOSS"

###
name_variation = None
###


# data_label = ["DESIY1_QMLE3"]
# data_label = "DESIY1_QMLE"
data_label = ["DESIY1_FFT3_dir"]

emu_cov_type = "full"
# emu_cov_type = "block"
# emu_cov_type = "diagonal"


emulator_label = "CH24_mpgcen_gpr"
# emulator_label="CH24_nyxcen_gpr"
# name_variation = "cosmo_h74"
# name_variation = "cosmo_mnu_varh"
# name_variation = "cosmo_low_3sig"
# name_variation = "cosmo_high_3sig"
# name_variation = "infl_emu_cov"

# name_variation = "Metals_Ma2025"

name_variation = None
p1d_fname = None
name_variation = "DLA_TAN"
p1d_fname = "/home/jchaves/Proyectos/projects/lya/data/in_DESI_DR1/ting_tan/p1d_fft_y1_measurement_kms_tingdla_nocrossexp_snr3noweights_directmetalsubtraction.fits"

args = Args(
    data_label=data_label,
    emulator_label=emulator_label,
    emu_cov_type=emu_cov_type,
    p1d_fname=p1d_fname,
)

args.set_baseline(
    fit_type="global_opt",
    fix_cosmo=False,
    P1D_type=data_label,
    name_variation=name_variation,
)

pip = Pipeline(args)


# %%
# cov1 = pip.fitter.like.data.cov_Pk_kms.copy()
# cov2 = pip.fitter.like.data.cov_Pk_kms.copy()

# pk1 = pip.fitter.like.data.Pk_kms.copy()
# pk2 = pip.fitter.like.data.Pk_kms.copy()

# %%
from cup1d.plots_and_tables.table_nuisance import table_nuisance

# %%
folder_in = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_2/"
table_nuisance(folder_in)

# %%
folder_in = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_FFT3_dir/DLA_TAN/CH24_mpgcen_gpr/chain_2/"
table_nuisance(folder_in)

# %%
# cov1[0]/cov2[0]

# %%
# kk = pip.fitter.like.data.k_kms
# for ii in range(len(cov2)):
#     plt.plot(kk[ii], np.diag(cov1[ii])/np.diag(cov2[ii]), label="z=" + str(pip.fitter.like.data.z[ii]))
# plt.yscale("log")
# plt.ylabel("covariance diagonal ratio new/original")
# plt.xlabel("k [h/Mpc]")
# plt.legend()

# %%
for ii, par in enumerate(pip.fitter.like.free_params):
    print(ii, par.name, par.value, par.min_value, par.max_value)

# %%
# p0[18:26] = np.array(
#     [
#         0.1,
#         0.1,
#         0.3,
#         0.3,
#         0.66,
#         0.70,
#         0.52,
#         0.52,
#     ]
# )

# %%
# plt.plot(pip.fitter.like.data.full_cov_stat_Pk_kms[100])
# plt.plot(-pip.fitter.like.data.full_cov_stat_Pk_kms[100], alpha=0.5)
# plt.yscale("log")

# %%

p0 = pip.fitter.like.sampling_point_from_parameters().copy()
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%
# data_lab = "DESIY1_QMLE3"
data_lab = "DESIY1_FFT3_dir"

emu = "mpg"

fit_type = "global_opt"
# fit_type = "emu_diag"
# fit_type = "emu_block"


if data_lab == "DESIY1_QMLE3":
    folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_7/"
elif data_lab == "DESIY1_FFT3_dir":
    if name_variation is None:
        folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/chain_2/"
    else:
        folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_FFT3_dir/DLA_TAN/CH24_mpgcen_gpr/chain_2/"

data = np.load(folder + "fitter_results.npy", allow_pickle=True).item()
p0 = data["fitter"]["mle_cube"]
free_params = pip.fitter.like.parameters_from_sampling_point(p0)
pip.fitter.like.get_chi2(p0)

# %%

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%
pip.fitter.like.plot_p1d(
    p0, 
    residuals=True, 
    plot_panels=True, 
    print_chi2=False, 
    fix_cosmo=False, 
    plot_fname=None
)


# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
chain = np.load(base + folder + "chain.npy")


# %%
pip.run_minimizer(p0)

# %%
p0 = pip.fitter.mle_cube

# %%
pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %%

# %%

# %%

# %%

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

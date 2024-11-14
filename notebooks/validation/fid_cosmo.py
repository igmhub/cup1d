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
# # Does the fiducial cosmology matter?
#
# This test examines whether choosing a particular fiducial cosmology matters for cup1d constraints. This cosmology enters into the calculation of power spectra from CAMB, which gets rescaled during inference

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import glob
import matplotlib.pyplot as plt

# %%
folder_out = "/home/jchaves/Proyectos/projects/lya/data/cup1d/validate_cosmo/"
folder_cov = "Chabanier2019"
arr_folder_emu = ["Pedersen23_ext", "Cabayol23+", "Nyx_alphap_cov"]
dict_sims = {
    "mpg":30,
    "nyx":17,
}

jj = 2
if "Nyx" in arr_folder_emu[jj]:
    b_sim = "nyx"
else:
    b_sim = "mpg"
nsims = dict_sims[b_sim]

arr_star = np.zeros((nsims, 3))
true_star = np.zeros((nsims, 3))

for ii in range(nsims):
    if(ii == 14):
        ii += 1
    
    file = folder_out + "/" + folder_cov + "/" + arr_folder_emu[jj] + "/" + b_sim + "_" + str(ii) + "/chain_1/minimizer_results.npy"
    res = np.load(file, allow_pickle=True).item()

    true_star[ii, 0] = res['truth']['$\\Delta^2_\\star$']
    true_star[ii, 1] = res['truth']['$n_\\star$']
    
    arr_star[ii, 0] = res['mle']['$\\Delta^2_\\star$']
    arr_star[ii, 1] = res['mle']['$n_\\star$']
    if '$\\alpha_\\star$' in res['truth']:
        true_star[ii, 2] = res['truth']['$\\alpha_\\star$']        
        arr_star[ii, 2] = res['mle']['$\\alpha_\\star$']

# %%
fontsize = 16
if "Nyx" in arr_folder_emu[jj]:
    nax = 3
    fig, ax = plt.subplots(1, 3, figsize=(14, 6)) 
else:
    nax = 1
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax = [ax]
    
for ii in range(nax):
    if ii == 0:
        jj0 = 0
        jj1 = 1
    elif ii == 1:
        jj0 = 0
        jj1 = 2
    elif ii == 2:
        jj0 = 1
        jj1 = 2

    x0 = (arr_star[:, jj0]/true_star[:, jj0]-1) * 100
    y0 = (arr_star[:, jj1]/true_star[:, jj1]-1) * 100        
    ax[ii].scatter(x0, y0, marker=".", color="blue")

    ax[ii].axhline(color="black", linestyle=":")
    ax[ii].axvline(color="black", linestyle=":")

if "Nyx" in arr_folder_emu[jj]:
    ax[0].set_xlabel(r"$\Delta(\Delta^2_\star)$ [%]", fontsize=fontsize)
    ax[0].set_ylabel(r"$\Delta(n_\star)$ [%]", fontsize=fontsize)
    ax[1].set_xlabel(r"$\Delta(\Delta^2_\star)$ [%]", fontsize=fontsize)
    ax[1].set_ylabel(r"$\Delta(\alpha_\star)$ [%]", fontsize=fontsize)
    ax[2].set_xlabel(r"$\Delta(n_\star)$ [%]", fontsize=fontsize)
    ax[2].set_ylabel(r"$\Delta(\alpha_\star)$ [%]", fontsize=fontsize)
else:
    ax[0].set_xlabel(r"$\Delta(\Delta^2_\star)$ [%]", fontsize=fontsize)
    ax[0].set_ylabel(r"$\Delta(n_\star)$ [%]", fontsize=fontsize)
plt.tight_layout()

# %%

# %%

# %% [markdown]
# ## Read Data

# %% [markdown]
# ### Options

# %%
from cup1d.likelihood.sampler_pipeline import path_sampler
from matplotlib.ticker import MaxNLocator
from lace.archive import gadget_archive, nyx_archive
from cup1d.likelihood import lya_theory
from lace.cosmo.camb_cosmo import (
    get_camb_results,
    get_Nyx_cosmology,
    get_cosmology_from_dictionary,
)
from lace.cosmo.fit_linP import parameterize_cosmology_kms
from cup1d.likelihood import CAMB_model


# %% [markdown]
# #### check data

# %%
def get_data(emu_label, mock_sim, igm_sim, cosmo_sim, chain="1", nigm=0):
    folder0 = "/home/jchaves/Proyectos/projects/lya/data/cup1d/sampler/v3/"
    folder_cov = "cov_Chabanier2019/"
    # folder1 = f"emu_Cabayol23_cov_Chabanier2019_mocksim_{mock_sim}_cosmosim_{cosmo_sim}_igmsim_{igm_sim}_nigm_2_ydrop_ypoly/"
    folder1 = f"mock_{mock_sim}_igm_{igm_sim}_cosmo_{cosmo_sim}_nigm_{nigm}_drop_smooth/"
    name = folder0 + emu_label + '/' + folder_cov + folder1 + "chain_"+chain+"/results.npy"
    try:
        results = np.load(name, allow_pickle=True).item()
    except:
        print(name)
        raise
    return results



# %%
par_get = ["$\\Delta^2_\\star$", "$n_\\star$"]
# emu_labs = ["P23_ext", "C23", "CH24", "C23+"]
# emu_labs = ["P23_ext", "C23", "C23+"]
emu_labs = ["P23_ext", "C23+"]
sim_labs = []

# arr_emu_label = ["emu_Pedersen23_ext", "emu_Cabayol23", "emu_CH24", "emu_Cabayol23+"]
# arr_emu_label = ["emu_Pedersen23_ext", "emu_Cabayol23", "emu_Cabayol23+"]
arr_emu_label = ["emu_Pedersen23_ext", "emu_Cabayol23+"]

sim_label_cen = "mpg_central"

lio_true = np.zeros((30, len(par_get)))
lio_best = np.zeros((len(arr_emu_label), 30, len(par_get), 3))
lio_ml = np.zeros((len(arr_emu_label), 30, len(par_get)))

sim_labs = []
for iemu, emu_label in enumerate(arr_emu_label):
    for isim in range(30):
        sim_label = "mpg_" + str(isim)
        if(iemu == 0):
            sim_labs.append(sim_label)
        if((emu_label == "emu_Cabayol23+")):
            chain = "1"
        else:
            chain = "1"
        try:
            dat = get_data(emu_label, sim_label, sim_label, sim_label,chain=chain, nigm=0)
        except:
            print(emu_label, isim)
            continue
        # for opt in range(1):
        #     if opt == 0:
        #         lio_best[iemu, isim] = check_data(emu_label, sim_label, sim_label, sim_label)
        #     elif opt == 1:
        #         dat = check_data(emu_label, sim_label, sim_label, sim_label_cen)
        #     elif opt == 2:
        #         dat = check_data(emu_label, sim_label, sim_label_cen, sim_label)
        #     elif opt == 3:
        #         dat = check_data(emu_label, sim_label, sim_label_cen, sim_label_cen)

        # print(emu_label, sim_label)
        ii0 = 0
        for ii, par in enumerate(dat["param_names"]):
            if par in par_get:
                if iemu == 0:
                    lio_true[isim, ii0] = dat["truth"][par]
                lio_best[iemu, isim, ii0]  = dat["param_percen"][ii]
                # if(emu_label == "emu_Cabayol23+"):
                lio_ml[iemu, isim, ii0]  = dat["param_mle"][par]
                # else:
                    # lio_ml[iemu, isim, ii0]  = dat["param_ml"][ii]
                ii0 += 1
sim_labs = np.array(sim_labs)

# %%
print(lio_ml[-1, 9])
print(lio_true[9, :])
print(lio_ml[-1, 9]/lio_true[9, :]-1)

# %%
P23_ext par 0 bias -7.14 std 7.79
C23 par 0 bias -10.64 std 13.67
C23+ par 0 bias 11.13 std 8.6
P23_ext par 1 bias 0.51 std 0.36
C23 par 1 bias 0.62 std 0.82
C23+ par 1 bias -0.82 std 0.66

# %%
diff
P23_ext par 0 bias 8.17 std 11.14
C23 par 0 bias -2.18 std 7.3
C23+ par 0 bias 13.08 std 13.04
P23_ext par 1 bias -0.01 std 0.3
C23 par 1 bias 0.32 std 0.44
C23+ par 1 bias -0.4 std 0.54

# %% [markdown]
# ### Plot data

# %%
out_folder = (
    os.environ["LYA_DATA_PATH"] + "/cup1d/sampler/v3/figs/"
)
index = 0
plot_lio(
    out_folder,
    emu_labs,
    sim_labs,
    par_get,
    lio_true,
    lio_best,
    lio_ml,
    type_plot="diff_mle",
    index=index,
    nigm=0,
    save=True
)

# %%
lio_best.shape


# %%
def plot_lio(
    out_folder, emu_labs, sim_labs, par_labs, truth, best, mle, type_plot="both", index=0, save=False, nigm=0
):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    lim_par = [[-10, 10], [-0.5, 0.5]]

    # over params
    for ipar in range(truth.shape[1]):
        if ipar == 0:
            label1 = "Truth"
            if(nigm == 0):
                ylim = 15
            else:
                ylim = 40
        else:
            label0 = None
            label1 = None
            if(nigm == 0):
                ylim = 1
            else:
                ylim = 2

        # over emus
        for iemu in range(best.shape[0]):
            col = "C"+str(iemu)
            label0 = emu_labs[iemu]

            y_best = best[iemu, :, ipar, 1]
            if(type_plot == 'diff_mle'):
                y_best = mle[iemu, :, ipar]
            y_err = np.abs(best[iemu, :, ipar, np.array([0, 2])] - y_best[iemu, None])
            y_true = truth[:, ipar]

            per_err = (y_best / y_true - 1) * 100
            _ = y_best != 0
            bias = np.round(np.median(per_err[_]),2)
            std = np.round(np.std(per_err[_]),2)
            print(label0, "par", ipar, "bias", bias, "std", std)

            if type_plot == "both":
                ax[ipar].errorbar(
                    sim_labs,
                    y_best,
                    yerr=y_err,
                    color=col,
                    ls="",
                    marker="o",
                    label=label0,
                    alpha=0.5,
                )

                if(iemu == 0):
                    ax[ipar].plot(
                        sim_labs,
                        y_true,
                        ls="",
                        color=col,
                        marker="x",
                        label=label1,
                    )
                
            else:
                ax[ipar].plot(
                    sim_labs,
                    (y_best / y_true - 1) * 100,
                    # yerr=y_err / np.abs(y_true) * 100,
                    color=col,
                    ls="",
                    marker="o",
                    label=label0,
                    alpha=0.5,
                    # capsize=3,
                )
                # ax[ipar].errorbar(
                #     sim_labs,
                #     (y_best / y_true - 1) * 100,
                #     yerr=y_err / np.abs(y_true) * 100,
                #     color=col,
                #     ls="",
                #     marker="o",
                #     label=label0,
                #     alpha=0.5,
                #     capsize=3,
                # )

        plt.xticks(rotation=45, ha="right")
        ax[ipar].axhline(0, ls=":", color="k", alpha=0.5)
        if(ipar == 0):
            ax[ipar].legend(loc='upper right')
        if (type_plot == "both"):
            ax[ipar].set_ylabel(par_labs[ipar])
        else:
            for ii in range(2):
                ax[ipar].axhline(lim_par[ipar][ii], ls=":", color="k", alpha=0.5)
            ax[ipar].set_ylabel(r"$\Delta$(" + par_labs[ipar] + ")[%]")
            ax[ipar].set_ylim(-ylim, ylim)

    if index == 0:
        tit = "Mock L1O, IGM L1O, Cosmo L1O"
    elif index == 1:
        tit = "Mock L1O, IGM L1O, Cosmo central"
    elif index == 2:
        tit = "Mock L1O, IGM central, Cosmo L1O"
    elif index == 3:
        tit = "Mock L1O, IGM central, Cosmo central"
    if type_plot == "both":
        msg = "Best-fitting solution and truth"
    else:
        msg = "Percentage difference between best-fitting solution and truth"

    plt.suptitle(tit + "\n\n" + msg)
    plt.tight_layout()

    if(save):
        folds = ["", "pdf/", "png/"]
        for ii, fold in enumerate(folds):
            _out_folder = out_folder + fold
            print(_out_folder)
            if not os.path.exists(_out_folder):
                os.makedirs(_out_folder)
            if ii == 1:
                plt.savefig(
                    _out_folder + "lio_" + type_plot + "_" + str(index) + ".pdf"
                )
            elif ii == 2:
                plt.savefig(
                    _out_folder + "lio_" + type_plot + "_" + str(index) + ".png"
                )


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ### OLD VERSION

# %%
def get_fid_cosmo(archive, z_star=3.0, kp_kms=0.009):
    if archive.list_sim[0][:3] == "mpg":
        get_cosmo = get_cosmology_from_dictionary
    else:
        get_cosmo = get_Nyx_cosmology
    arr_fid_cosmo = {}
    arr_fid_cosmo["Delta2_star"] = np.zeros((len(archive.list_sim)))
    arr_fid_cosmo["n_star"] = np.zeros((len(archive.list_sim)))
    for ii, sim in enumerate(archive.list_sim):
        # this simulation is problematic
        if sim != "nyx_14":
            ind = np.argwhere(archive.sim_label == sim)[0, 0]
            cosmo_fid = get_cosmo(archive.data[ind]["cosmo_params"])
            cosmo_model_fid = CAMB_model.CAMBModel(
                zs=archive.list_sim_redshifts,
                cosmo=cosmo_fid,
                z_star=z_star,
                kp_kms=kp_kms,
            )
            par = cosmo_model_fid.get_linP_params()
            arr_fid_cosmo["Delta2_star"][ii] = par["Delta2_star"]
            arr_fid_cosmo["n_star"][ii] = par["n_star"]
    return arr_fid_cosmo


def summary2array(summary, par_use):
    out = np.zeros((len(par_use), 3))
    ii = 0
    for par in file["summary"]:
        if par in par_use:
            out[ii, 0] = file["summary"][par].lower
            out[ii, 1] = file["summary"][par].center
            out[ii, 2] = file["summary"][par].upper
            ii += 1
    return out

# %%
# list of options to set

# training_set = "Cabayol23"
# emulator_label = "Cabayol23"
# add_hires = False
# arr_drop_sim = [True, False]
# arr_n_igm = [0, 1, 2]
# chain_lab = "chain_1"


# training_set = "Nyx23_Oct2023"
# emulator_label = "Nyx_v0"
# add_hires = False
# arr_drop_sim = [True]
# arr_n_igm = [1]
# chain_lab = "chain_1"

training_set = "Nyx23_Oct2023"
emulator_label = "Nyx_v0"
add_hires = False
arr_drop_sim = [True]
arr_n_igm = [2]
chain_lab = "chain_2"

# training_set = "Nyx23_Oct2023"
# emulator_label = "Nyx_v0_extended"
# add_hires = True
# arr_drop_sim = [True]
# arr_n_igm = [1, 2]
# chain_lab = "chain_1"

# emulator_label = "Cabayol23_extended"
# add_hires = True
use_polyfit = True
cov_label = "Chabanier2019"
override = False


if (training_set == "Pedersen21") | (training_set == "Cabayol23"):
    list_sims = [
        "mpg_0",
        "mpg_1",
        "mpg_2",
        "mpg_3",
        "mpg_4",
        "mpg_5",
        "mpg_6",
        "mpg_7",
        "mpg_8",
        "mpg_9",
        "mpg_10",
        "mpg_11",
        "mpg_12",
        "mpg_13",
        "mpg_14",
        "mpg_15",
        "mpg_16",
        "mpg_17",
        "mpg_18",
        "mpg_19",
        "mpg_20",
        "mpg_21",
        "mpg_22",
        "mpg_23",
        "mpg_24",
        "mpg_25",
        "mpg_26",
        "mpg_27",
        "mpg_28",
        "mpg_29",
        "mpg_central",
        "mpg_seed",
        "mpg_growth",
        "mpg_neutrinos",
        "mpg_curved",
        "mpg_running",
        "mpg_reio",
    ]
elif training_set[:5] == "Nyx23":
    list_sims = [
        "nyx_0",
        "nyx_1",
        "nyx_2",
        "nyx_3",
        "nyx_4",
        "nyx_5",
        "nyx_6",
        "nyx_7",
        "nyx_8",
        "nyx_9",
        "nyx_10",
        "nyx_11",
        "nyx_12",
        "nyx_13",
        "nyx_14",
        "nyx_15",
        "nyx_16",
        "nyx_17",
        "nyx_central",
        "nyx_seed",
        "nyx_wdm",
    ]

# %%
emu_params = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
if training_set == "Pedersen21":
    archive = gadget_archive.GadgetArchive(postproc=training_set)
elif training_set == "Cabayol23":
    archive = gadget_archive.GadgetArchive(postproc=training_set)
elif training_set[:5] == "Nyx23":
    archive = nyx_archive.NyxArchive(nyx_version=training_set[6:])
fid_cosmo = get_fid_cosmo(archive)

# %%
ind = np.argwhere(np.array(list_sims) == "nyx_7")[:, 0]
print(fid_cosmo["Delta2_star"][ind], fid_cosmo["n_star"][ind])

# %% [markdown]
# ### Load

# %% [markdown]
# #### Minimizer

# %%
# # chi2, As, ns
# chi2 = np.zeros((len(arr_drop_sim), len(arr_n_igm), len(list_sims)))
# # diff_As, diff_ns
# dat = np.zeros((len(arr_drop_sim), len(arr_n_igm), len(list_sims), 2))
# # err_As, err_ns
# err = np.zeros((len(arr_drop_sim), len(arr_n_igm), len(list_sims), 2))

# for ii, drop_sim in enumerate(arr_drop_sim):
#     for jj, n_igm in enumerate(arr_n_igm):
#         for kk, sim_label in enumerate(list_sims):
#             args = Args()

#             args.training_set = training_set
#             args.emulator_label = emulator_label
#             args.add_hires = add_hires
#             args.use_polyfit = use_polyfit
#             args.cov_label = cov_label

#             args.drop_sim = drop_sim
#             args.n_igm = n_igm
#             args.test_sim_label = sim_label

#             fname = fname_minimize(args)
#             file = np.load(fname, allow_pickle=True).item()
#             chi2[ii, jj, kk] = file["best_chi2"]
#             diff = file["best_parameters"][:2] - file["truth_parameters"][:2]
#             dat[ii, jj, kk] = diff
#             err[ii, jj, kk] = file["err_best_parameters"][:2]

# %% [markdown]
# #### Sampler

# %%

# %%
par_use = ["$\\Delta^2_\\star$", "$n_\\star$"]

# chi2, As, ns
# chi2 = np.zeros((len(arr_drop_sim), len(arr_n_igm), len(list_sims)))
# diff_As, diff_ns
diff_dat = np.zeros(
    (len(arr_drop_sim), len(arr_n_igm), len(list_sims), len(par_use))
)
diff_errdat = np.zeros(
    (len(arr_drop_sim), len(arr_n_igm), len(list_sims), len(par_use), 2)
)
rel_dat = np.zeros(
    (len(arr_drop_sim), len(arr_n_igm), len(list_sims), len(par_use))
)
rel_errdat = np.zeros(
    (len(arr_drop_sim), len(arr_n_igm), len(list_sims), len(par_use), 2)
)
# err_As, err_ns
# err = np.zeros((len(arr_drop_sim), len(arr_n_igm), len(list_sims), 3))


sim_avoid = ["nyx_14", "nyx_15", "nyx_16", "nyx_17", "nyx_seed", "nyx_wdm"]

for ii, drop_sim in enumerate(arr_drop_sim):
    for jj, n_igm in enumerate(arr_n_igm):
        for kk, sim_label in enumerate(list_sims):
            if sim_label not in sim_avoid:
                args = Args()

                args.training_set = training_set
                args.emulator_label = emulator_label
                args.add_hires = add_hires
                args.use_polyfit = use_polyfit
                args.cov_label = cov_label

                args.drop_sim = drop_sim
                args.n_igm = n_igm
                args.test_sim_label = sim_label

                fname = path_sampler(args) + "/" + chain_lab + "/results.npy"
                try:
                    file = np.load(fname, allow_pickle=True).item()
                except:
                    print("cannot load " + fname)
                else:
                    _sum = summary2array(file["summary"], par_use)
                    _cen = _sum[:, 1]
                    _top = _sum[:, 2] - _sum[:, 1]
                    _bot = _sum[:, 1] - _sum[:, 0]

                    diff_dat[ii, jj, kk, 0] = (
                        _cen[0] - fid_cosmo["Delta2_star"][kk]
                    )
                    diff_dat[ii, jj, kk, 1] = _cen[1] - fid_cosmo["n_star"][kk]
                    diff_errdat[ii, jj, kk, :, 0] = _bot
                    diff_errdat[ii, jj, kk, :, 1] = _top

                    rel_dat[ii, jj, kk, 0] = (
                        _cen[0] / fid_cosmo["Delta2_star"][kk] - 1
                    )
                    rel_dat[ii, jj, kk, 1] = (
                        _cen[1] / fid_cosmo["n_star"][kk] - 1
                    )
                    rel_errdat[ii, jj, kk, 0, 0] = (
                        _bot[0] / fid_cosmo["Delta2_star"][kk]
                    )
                    rel_errdat[ii, jj, kk, 0, 1] = (
                        _top[0] / fid_cosmo["Delta2_star"][kk]
                    )
                    rel_errdat[ii, jj, kk, 1, 0] = (
                        _bot[1] / fid_cosmo["n_star"][kk]
                    )
                    rel_errdat[ii, jj, kk, 1, 1] = (
                        _top[1] / fid_cosmo["n_star"][kk]
                    )
rel_errdat = np.abs(rel_errdat)

# %%

# %% [markdown]
# ## Plot

# %%
fnames = "/home/jchaves/Proyectos/projects/lya/data/planck/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE.paramnames"
_ = np.loadtxt(fnames, dtype="str")
planck_names = _[:, 0]
planck_names_descr = _[:, 1]

dict_plack = {"w": [], "As": [], "ns": []}
fchain = "/home/jchaves/Proyectos/projects/lya/data/planck/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE_"
for ii in range(1, 5):
    file = np.loadtxt(fchain + str(ii) + ".txt")
    dict_plack["w"].append(file[:, 0])
    #     dict_plack["lnprop"].append(file[:, 1])
    dict_plack["As"].append(np.exp(file[:, 6]) / 1e10)
    dict_plack["ns"].append(file[:, 7])
for par in dict_plack:
    dict_plack[par] = np.concatenate(dict_plack[par])

# %%
from chainconsumer import ChainConsumer, Chain, Truth
import pandas as pd

# %%
pd_data = pd.DataFrame(data=dict_plack)

# %%
chain = Chain(samples=pd_data, weight_column="w", name="a")

# %%
c = ChainConsumer()
c.add_chain(chain)

# %%
fig = c.plotter.plot()

# %%

# %% [markdown]
# #### minimizer

# %%
# folder = os.environ["CUP1D_PATH"] + "/data/minimize/Cabayol23_lres/"

# for ii in range(len(arr_drop_sim)):
#     for jj in range(len(arr_n_igm)):
#         samples = list_sims
#         values1 = dat[ii, jj, :, 0] * 1e8
#         err1 = np.sqrt(err[ii, jj, :, 0] * 1e8)
#         values2 = dat[ii, jj, :, 1]
#         err2 = np.sqrt(err[ii, jj, :, 1])
#         values3 = chi2[ii, jj, :]

#         fig, ax = plt.subplots(1, 3, sharey=True, figsize=(5, 10))

#         # Create a scatter plot with two points for each sample
#         ax[0].errorbar(
#             values1, samples, yerr=None, xerr=err1, ls="", marker="o"
#         )
#         #         ax[0].scatter(values1, samples, marker="o", color="r")
#         ax[1].errorbar(
#             values2, samples, yerr=None, xerr=err2, ls="", marker="o"
#         )

#         #         ax[1].scatter(values2, samples)
#         ax[2].scatter(values3, samples)
#         lvls = [-0.05, 0, 0.05]
#         lss = [":", "--", ":"]
#         for ls, lvl in zip(lss, lvls):
#             ax[0].axvline(lvl, ls=ls, color="k")
#             ax[1].axvline(lvl, ls=ls, color="k")
#         lvls = [1, 10, 100]
#         for lvl in lvls:
#             ax[2].axvline(lvl, ls="--", color="k")
#         ax[0].set_xlabel(r"$\Delta A_s \times 10^8$")
#         ax[1].set_xlabel(r"$\Delta n_s$")
#         ax[2].set_xlabel(r"$\chi^2$")
#         ax[2].set_xscale("log")
#         ax[0].set_xlim([-0.1, 0.1])
#         ax[1].set_xlim([-0.1, 0.1])
#         ax[2].set_xlim([0.1, 1e3])

#         if arr_drop_sim[ii]:
#             flag = "leave 1 out,"
#             flag2 = "yl1O"
#         else:
#             flag = "no leave 1 out,"
#             flag2 = "nl1O"

#         plt.suptitle(flag + " n_igm=" + str(arr_n_igm[jj]))
#         plt.tight_layout()
#         #         plt.savefig(folder + flag2 + "nigm" + str(arr_n_igm[jj]) + ".png")
#         break
#     break

# %% [markdown]
# ### sampler

# %% [markdown]
# #### difference

# %%
folder = (
    os.path.dirname(os.environ["CUP1D_PATH"])
    + "/data/cup1d/sampler/"
    + training_set
)

vlines = np.array([[0.0, 0.05, -0.05], [0.0, 0.01, -0.01]])

mask = []
for sim in archive.list_sim:
    if (sim in sim_avoid) | (sim in archive.list_sim_test[1:]):
        mask.append(False)
    else:
        mask.append(True)

if add_hires:
    folder += "_hres/"
else:
    folder += "_lres/"
folder += "figs/"

lab_par_use = []
for par in par_use:
    lab_par_use.append("Diff. " + par)

for ii in range(len(arr_drop_sim)):
    for jj in range(len(arr_n_igm)):
        samples = np.array(list_sims)[mask]

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5, 10))

        # Create a scatter plot with two points for each sample
        for kk in range(2):
            values = diff_dat[ii, jj, mask, kk]
            xerr = diff_errdat[ii, jj, mask, kk, :].T

            ax[kk].plot(
                values,
                samples,
                ls="",
                color="C0",
                marker="o",
            )
            ax[kk].errorbar(
                values,
                samples,
                xerr=xerr,
                color="C0",
                ls="",
            )
            std_values = np.percentile(values, [16, 84])
            ax[kk].axvspan(std_values[0], std_values[1], alpha=0.3, color="C0")
            #             ax[kk].scatter(values, samples, marker="o", color="r")
            ax[kk].set_xlabel(lab_par_use[kk])
            if kk == 0:
                ax[kk].set_xlim([-0.2, 0.2])
            else:
                ax[kk].set_xlim([-0.07, 0.07])

            for uu in range(vlines.shape[1]):
                ax[kk].axvline(vlines[kk, uu], ls=":", color="k")

        if arr_drop_sim[ii]:
            flag = "leave 1 out,"
            flag2 = "flag2_yl1O"
        else:
            flag = "no leave 1 out,"
            flag2 = "flag2_nl1O"

        plt.suptitle(flag + " n_igm=" + str(arr_n_igm[jj]))
        plt.tight_layout()
        plt.savefig(
            folder + "png/" + flag2 + "_nigm" + str(arr_n_igm[jj]) + ".png"
        )
        plt.savefig(
            folder + "pdf/" + flag2 + "_nigm" + str(arr_n_igm[jj]) + ".pdf"
        )
#         break
#     break

# %% [markdown]
# #### rel difference

# %%
folder = (
    os.path.dirname(os.environ["CUP1D_PATH"])
    + "/data/cup1d/sampler/"
    + training_set
)

vlines = np.array([[0.0, 0.2, -0.2], [0.0, 0.01, -0.01]])

mask = []
for sim in archive.list_sim:
    if (sim in sim_avoid) | (sim in archive.list_sim_test[1:]):
        mask.append(False)
    else:
        mask.append(True)

if add_hires:
    folder += "_hres/"
else:
    folder += "_lres/"
folder += "figs/"

lab_par_use = []
for par in par_use:
    lab_par_use.append("Rel. diff. " + par + " [%]")

for ii in range(len(arr_drop_sim)):
    for jj in range(len(arr_n_igm)):
        samples = np.array(list_sims)[mask]

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5, 10))

        # Create a scatter plot with two points for each sample
        for kk in range(2):
            values = rel_dat[ii, jj, mask, kk]
            xerr = rel_errdat[ii, jj, mask, kk, :].T

            ax[kk].plot(
                values,
                samples,
                ls="",
                color="C0",
                marker="o",
            )
            ax[kk].errorbar(
                values,
                samples,
                xerr=xerr,
                color="C0",
                ls="",
            )
            std_values = np.percentile(values, [16, 84])
            ax[kk].axvspan(std_values[0], std_values[1], alpha=0.3, color="C0")
            #             ax[kk].scatter(values, samples, marker="o", color="r")
            ax[kk].set_xlabel(lab_par_use[kk])
            if kk == 0:
                ax[kk].set_xlim([-0.5, 0.5])
            else:
                ax[kk].set_xlim([-0.03, 0.03])

            for uu in range(vlines.shape[1]):
                ax[kk].axvline(vlines[kk, uu], ls=":", color="k")

        if arr_drop_sim[ii]:
            flag = "leave 1 out,"
            flag2 = "flag2_yl1O"
        else:
            flag = "no leave 1 out,"
            flag2 = "flag2_nl1O"

        plt.suptitle(flag + " n_igm=" + str(arr_n_igm[jj]))
        plt.tight_layout()
        plt.savefig(
            folder + "png/rel_" + flag2 + "_nigm" + str(arr_n_igm[jj]) + ".png"
        )
        plt.savefig(
            folder + "pdf/rel_" + flag2 + "_nigm" + str(arr_n_igm[jj]) + ".pdf"
        )
# #         break
# #     break

# %% [markdown]
# ### rel diff extra sims

# %%
folder = (
    os.path.dirname(os.environ["CUP1D_PATH"])
    + "/data/cup1d/sampler/"
    + training_set
)

vlines = np.array([[0.0, 0.2, -0.2], [0.0, 0.01, -0.01]])

mask = []
for sim in archive.list_sim:
    if (sim in sim_avoid) | (sim in archive.list_sim_cube):
        mask.append(False)
    else:
        mask.append(True)

if add_hires:
    folder += "_hres/"
else:
    folder += "_lres/"
folder += "figs/"

lab_par_use = []
for par in par_use:
    lab_par_use.append("Rel. diff. " + par + " [%]")

for ii in range(len(arr_drop_sim)):
    for jj in range(len(arr_n_igm)):
        samples = np.array(list_sims)[mask]

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5, 10))

        # Create a scatter plot with two points for each sample
        for kk in range(2):
            values = rel_dat[ii, jj, mask, kk]
            xerr = rel_errdat[ii, jj, mask, kk, :].T

            ax[kk].plot(
                values,
                samples,
                ls="",
                color="C0",
                marker="o",
            )
            ax[kk].errorbar(
                values,
                samples,
                xerr=xerr,
                color="C0",
                ls="",
            )
            #             std_values = np.percentile(values, [16, 84])
            #             ax[kk].axvspan(std_values[0], std_values[1], alpha=0.3, color="C0")
            #             ax[kk].scatter(values, samples, marker="o", color="r")
            ax[kk].set_xlabel(lab_par_use[kk])
            if kk == 0:
                ax[kk].set_xlim([-0.5, 0.5])
            else:
                ax[kk].set_xlim([-0.03, 0.03])

            for uu in range(vlines.shape[1]):
                ax[kk].axvline(vlines[kk, uu], ls=":", color="k")

        if arr_drop_sim[ii]:
            flag = ""
            flag2 = "flag2_yl1O"
        else:
            flag = ""
            flag2 = "flag2_nl1O"

        plt.suptitle(flag + " n_igm=" + str(arr_n_igm[jj]))
        plt.tight_layout()
        plt.savefig(
            folder
            + "png/rel_"
            + flag2
            + "_nigm"
            + str(arr_n_igm[jj])
            + "_especial.png"
        )
        plt.savefig(
            folder
            + "pdf/rel_"
            + flag2
            + "_nigm"
            + str(arr_n_igm[jj])
            + "_especial.pdf"
        )
# #         break
# #     break

# %% [markdown]
# #### both

# %%
folder = (
    os.path.dirname(os.environ["CUP1D_PATH"])
    + "/data/cup1d/sampler/"
    + training_set
)

vlines = np.array([[0.0, 0.05, -0.05], [0.0, 0.01, -0.01]])


mask = []
for sim in archive.list_sim:
    if (sim in sim_avoid) | (sim in archive.list_sim_test[1:]):
        mask.append(False)
    else:
        mask.append(True)

if add_hires:
    folder += "_hres/"
else:
    folder += "_lres/"
folder += "figs/"

lab_par_use = []
for par in par_use:
    lab_par_use.append(par)

for ii in range(len(arr_drop_sim)):
    for jj in range(len(arr_n_igm)):
        samples = np.array(list_sims)[mask]

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5, 10))

        # Create a scatter plot with two points for each sample
        par_labs = ["Delta2_star", "n_star"]
        for kk in range(2):
            if kk == 0:
                label0 = "Model"
                label1 = "Truth"
            else:
                label0 = None
                label1 = None

            values = diff_dat[ii, jj, mask, kk] + fid_cosmo[par_labs[kk]][mask]
            xerr = diff_errdat[ii, jj, mask, kk, :].T

            ax[kk].plot(
                values,
                samples,
                ls="",
                color="C0",
                marker="o",
            )
            ax[kk].errorbar(
                values,
                samples,
                xerr=xerr,
                color="C0",
                ls="",
            )

            ax[kk].plot(
                fid_cosmo[par_labs[kk]][mask],
                samples,
                ls="",
                color="C1",
                marker="x",
                label=label1,
            )

            ax[kk].set_xlabel(lab_par_use[kk])
            if kk == 0:
                ax[kk].legend()
            #             if kk == 0:
            #                 ax[kk].set_xlim([-0.2, 0.2])
            #             else:
            #                 ax[kk].set_xlim([-0.07, 0.07])

        #             for uu in range(vlines.shape[1]):
        #                 ax[kk].axvline(vlines[kk, uu], ls=":", color="k")

        if arr_drop_sim[ii]:
            flag = "leave 1 out,"
            flag2 = "flag2_yl1O"
        else:
            flag = "no leave 1 out,"
            flag2 = "flag2_nl1O"

        plt.suptitle(flag + " n_igm=" + str(arr_n_igm[jj]))
        plt.tight_layout()
        plt.savefig(
            folder
            + "png/both_"
            + flag2
            + "_nigm"
            + str(arr_n_igm[jj])
            + ".png"
        )
        plt.savefig(
            folder
            + "pdf/both_"
            + flag2
            + "_nigm"
            + str(arr_n_igm[jj])
            + ".pdf"
        )
#         break
#     break

# %%

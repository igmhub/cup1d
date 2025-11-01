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

# # Variations

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
from cup1d.utils.fit_ellipse import fit_ellipse, plot_ellipse
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from scipy.stats import chi2 as chi2_scipy


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
cont = np.array([0, 1, 2, 3, 4, 5])
prob_levels = np.zeros(len(cont))
chi2_levels = np.zeros(len(cont))

for ii in range(len(cont)):
    prob = chi2_scipy.cdf(cont[ii]**2, 1)
    chi2 = chi2_scipy.ppf(prob, 2)
    print(cont[ii], cont[ii]**2, chi2, prob)
    prob_levels[ii] = prob
    chi2_levels[ii] = chi2

print(prob_levels)
print(chi2_levels)
# -

# #### Contours from chains

from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
import matplotlib.cm as cm



# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/"
folder = base + "sim_mpg_central/CH24_mpgcen_gpr/chain_1/"
dat_mpg_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_sim = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"
dat_mpg_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_igm = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"
dat_mpg_igm0 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_igm0 = np.load(folder + "summary.npy", allow_pickle=True).item()

# folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_2/"
# dat_nyx_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_3/"
dat_nyx_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "sim_sherwood/CH24_mpgcen_gpr/chain_1/"
dat_sherwood = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# +
print(sum_mpg_sim["delta2_star_err"]/sum_mpg_igm["delta2_star_err"])
print(sum_mpg_sim["n_star_err"]/sum_mpg_igm["n_star_err"])

print(sum_mpg_sim["delta2_star_err"]/sum_mpg_igm0["delta2_star_err"])
print(sum_mpg_sim["n_star_err"]/sum_mpg_igm0["n_star_err"])

# +

print(1-sum_qmle["delta2_star_err"]/sum_mpg["delta2_star_err"])
print(1-sum_qmle["n_star_err"]/sum_mpg["n_star_err"])

# +

print(sum_nyx["delta2_star_err"]/sum_mpg["delta2_star_err"])
print(sum_nyx["n_star_err"]/sum_mpg["n_star_err"])
# -
nyx
emu_diag
emu_block

# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"

folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg = np.load(folder + "summary.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_mpg_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

## emu
# TBL
# folder = base + "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/"
# dat_nyx = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
# sum_nyx = np.load(folder + "summary.npy", allow_pickle=True).item()

## data

folder = base + "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_fft3 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_qmle = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_qmle = np.load(folder + "summary.npy", allow_pickle=True).item()

# folder = base + "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"
# dat_fft = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"
dat_zmax = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"
dat_zmin = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## cov

folder = base + "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"
dat_no_inflate = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"
dat_no_emu_cov = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/emu_diag/CH24_mpgcen_gpr/chain_1/"
# dat_emu_diag = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/emu_block/CH24_mpgcen_gpr/chain_1/"
# dat_emu_block = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/data_syst_diag/CH24_mpgcen_gpr/chain_1/"
dat_syst_diag = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## IGM

folder = base + "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"
dat_more_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## cosmo

folder = base + "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"
dat_cosmo = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_high = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_high_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_low = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"
dat_cosmo_low_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

## ingredients

folder = base + "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"
dat_dlas = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/HCD0/CH24_mpgcen_gpr/chain_1/"
dat_HCD0 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/HCD_BOSS/CH24_mpgcen_gpr/chain_1/"
dat_HCD_BOSS = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"
dat_metal_deco = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"
dat_metal_si2 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"
dat_metal_trad = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"
dat_metal_thin = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/Metals_Ma2025/CH24_mpgcen_gpr/chain_1/"
dat_Metals_Ma2025 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

## no more

# folder = base + "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"
# dat_turner = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"
# dat_hcd_z = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"
# dat_less_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()



# folder = base + "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"
# dat_metals_z = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()



# folder = base + "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"
# dat_no_inflate_no_emu_cov = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"
# dat_no_res = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()



# folder = base + "DESIY1_QMLE3/kF_kms/CH24_mpgcen_gpr/chain_2/"
# dat_kF = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# -
from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
import matplotlib.cm as cm


# +

ls = ["-", "--"]

lw = [3, 2]
col = [0.7, 0.3]
ftsize = 28
cmaps = ["Blues", "Oranges", "Greens", "Reds", "Purples"]

dict_trans = {
    "DESIY1_QMLE3_mpg":"Fiducial", 
    
    "DESIY1_QMLE_mpg":"Data: w/ low SNR", 
    "DESIY1_FFT3_dir_mpg": "Data: FFT",
    # "DESIY1_FFT_dir_mpg":"Data: FFT w/ low SNR", 
    "zmin": "Data: $z \geq 2.6$",  # restricted zrange
    "zmax": "Data: $z \leq 3.4$",  # restricted zrange
    
    "no_emu_cov":"Cov: w/o emu err", # no emu error
    "no_inflate":"Cov: w/o 5% err",
    "dat_syst_diag": "Cov: uncorr syst", # systematics data uncorrelated
    
    "DESIY1_QMLE3_nyx":"Emulator: lace-lyssa",
    
    "cosmo": "Cosmo: $\omega_0\omega_a$CDM",  # different fiducial cosmo
    "cosmo_low": "Cosmo: low $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    "cosmo_high": "Cosmo: high $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    
    "more_igm": "IGM: $n_z=6$",  # 6 params for IGM evolution
    # "less_igm": "IGM: $n_z=4$",  # 4 params for IGM evolution
    # "Turner24": r"IGM: $\bar{F},\, n_z=1$",  # mF from Turner24 with 1 free param to scale ERROR
    # "kF_kms": r"IGM: w/ $k_\mathrm{F}$",

    # "hcd_z": "HCD: $n_z=2$",  # 2 params for z ev hcd
    "DLAs": "HCD: only DLAs",  # no LLS, sub-DLA
    "HCD0": "HCD: w/ $f_\\mathrm{const}^\\mathrm{HCD}$", # w/ constant term
    "HCD_BOSS": "HCD: BOSS",
    
    # "metals_z": "Metals: $n_z=2$",  # 2 params for z ev metals
    
    "metal_si2": "Metals: no SiII-SiII",  # no SiII-SiII cont
    "metal_deco": "Metals: no H-Si decorr",  # no decorrelation metals
    "metal_thin": "Metals: opt thin",  # no desviation from optically-thin limit ERROR
    
    "metal_trad": "Metals: BOSS",  # 2 params for metals like eBOSS
    "Metals_Ma2025": "Metals: Ma+2025",
    
    # "no_res": "Model: no resolution",  # no resolution correction

    "sim_mpg_central": "mpg-central", 
    "sim_mpg_central_all": "Model: cosmo, IGM, cont, syst", 
    "sim_mpg_central_igm": "Model: cosmo, IGM",
    "sim_mpg_central_igm0": "Model: cosmo", 
    "sim_nyx_central": "lyssa-central", 
    "sim_sherwood": "sherwood", 
}


fname = [
    "data_type",
    "zsplit",
    "cov_data",
    "cov_emu",
    "cosmo",
    "cosmo_Asns",
    "metals_ing",
    "metals_models",
    "DLAs",
    "igm",
    # "model_ing_no",
    # "data",
    # "emu",
    # "val_sims",
    # "val_sims_model",
    # "test",
]

for image in range(9, 10):

    # if image in [3, 4, 5]:
    #     ftsize = 26
    # else:
    #     ftsize = 22
    factx = 1

    if image == 0:
        variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE_mpg", "DESIY1_FFT3_dir_mpg"]
        dats = [dat_mpg, dat_qmle, dat_fft3]
    elif image == 1:
        variations = ["DESIY1_QMLE3_mpg", "zmin", "zmax"]
        dats = [dat_mpg, dat_zmin, dat_zmax]
    elif image == 2:
        variations = ["DESIY1_QMLE3_mpg", "no_inflate", "dat_syst_diag"]
        dats = [dat_mpg, dat_no_inflate, dat_syst_diag]
    elif image == 3:
        variations = ["DESIY1_QMLE3_mpg", "no_emu_cov"]
        dats = [dat_mpg, dat_no_emu_cov]
    elif image == 4:
        variations = ["DESIY1_QMLE3_mpg", "cosmo", "cosmo_low", "cosmo_high"]
        dats = [dat_mpg, dat_cosmo, dat_cosmo_low, dat_cosmo_high]
    elif image == 5:
        variations = ["DESIY1_QMLE3_mpg", "cosmo", "cosmo_low", "cosmo_high"]
        dats = [dat_mpg_Asns, dat_cosmo_Asns, dat_cosmo_low_Asns, dat_cosmo_high_Asns]
        factx = 1e9
    elif image == 6:
        variations = ["DESIY1_QMLE3_mpg", "metal_deco", "metal_thin", "metal_si2"]
        dats = [dat_mpg, dat_metal_deco, dat_metal_thin, dat_metal_si2]
    elif image == 7:
        variations = ["DESIY1_QMLE3_mpg", "metal_trad", "Metals_Ma2025"]
        dats = [dat_mpg, dat_metal_trad, dat_Metals_Ma2025]
    elif image == 8:
        variations = ["DESIY1_QMLE3_mpg", "HCD0", "DLAs", "HCD_BOSS"]
        dats = [dat_mpg, dat_HCD0, dat_dlas, dat_HCD_BOSS]
    # elif image == 7:
    #     variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE3_nyx"]
    #     dats = [dat_mpg, dat_nyx]
    elif image == 9:
        variations = ["DESIY1_QMLE3_mpg", "more_igm"]
        dats = [dat_mpg, dat_more_igm]
    # elif image == 10:
    #     variations = ["sim_mpg_central", "sim_nyx_central", "sim_sherwood"]
    #     dats = [dat_mpg_sim, dat_nyx_sim, dat_sherwood]
    # elif image == 11:
    #     variations = ["sim_mpg_central_all", "sim_mpg_central_igm", "sim_mpg_central_igm0"]
    #     dats = [dat_mpg_sim, dat_mpg_igm, dat_mpg_igm0]
    else:
        continue


    dict_diff = {
        "xcen": np.median(dats[0][0.68][0][0]),
        "ycen": np.median(dats[0][0.68][0][1]),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    
    fit_type = "global_opt"
    x0 = 0
    y0 = 0
    for ii, var in enumerate(variations):
        print()
        dat = dats[ii].copy()
        cmap = plt.colormaps[cmaps[ii]]

        if var.startswith("sim_"):
            if "mpg_central" in var:
                clabel = "mpg_central"
            else:
                clabel = var[4:]
            cosmo = set_cosmo(cosmo_label=clabel)
            like_cosmo = CAMB_model.CAMBModel(np.array([3]), cosmo=cosmo)
            true_cosmo = like_cosmo.get_linP_params()
            ds_diff = true_cosmo["Delta2_star"]
            ns_diff = true_cosmo["n_star"]
            print(var, ds_diff, ns_diff)
        else:
            ds_diff = dict_diff["xcen"]
            ns_diff = dict_diff["ycen"]

        for inum, num in enumerate([0.68, 0.95]):
            if inum == 0:
                label=dict_trans[var]
            else:
                label=None
            for jj in range(len(dat[num])):
                x = (dat[num][jj][0] - ds_diff) * factx
                y = dat[num][jj][1] - ns_diff
                ax.plot(x, y, color=cmap(col[inum]), label=label, lw=lw[inum], alpha=0.75)
                ax.fill(x, y, color=cmap(col[inum]), alpha=0.5)

    if image != 8:
        ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize+2)
        ax.set_ylabel(r"$\Delta(n_\star)$", fontsize=ftsize+2)
    else:
        ax.set_xlabel(r"$\Delta(A_s)[\times 10^{-9}]$", fontsize=ftsize+2)
        ax.set_ylabel(r"$\Delta(n_s)$", fontsize=ftsize+2)
    ax.tick_params(
        axis="both", which="major", labelsize=ftsize - 2
    )
    ax.axhline(color="k", ls=":")
    ax.axvline(color="k", ls=":")


# fname = [
#     "data_diff",
#     "cov",
#     "cosmo",
#     "modelz",
#     "model_ing_yes",
#     "model_ing_no",
#     "data",
#     "emu",
#     "cosmo_Asns",
#     "DLAs",
#     "val_sims",
#     "val_sims_model",
#     "test",
# ]
    
    if fname[image] in ["cosmo", "cosmo_Asns", "metals_ing", "model_ing_no",  "DLAs"]:
        ymin, ymax = plt.ylim()
        yrange = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.2 * yrange)
        
    if fname[image] in ["data"]:
        loc = "lower right"
    elif fname[image] in ["val_sims"]:
        loc = "upper left"
    else:
        loc = "upper right"
    
    plt.legend(fontsize=ftsize-6, loc=loc, ncol=1)
    plt.tight_layout()
    plt.savefig("figs/vars/variations_"+fname[image]+".pdf")
    plt.savefig("figs/vars/variations_"+fname[image]+".png")
# -


# from matplotlib.patches import Ellipse


# +
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
ls = ["-", "--"]

variations = ["sim_nyx_central"]
# variations = ["sim_mpg_central", "sim_nyx_central"]
dict_trans = {
    "sim_mpg_central":"mpg-central", 
    "sim_nyx_central":"nyx-central", 
    "sim_sherwood":"sherwood", 
}
var_deg = [550-26, 681-26, 670-26]



fit_type = "global_opt"
x0 = 0
y0 = 0
for ii, var in enumerate(variations):
    print()
    file = "out_pl/"+ var + ".npy"
    out_dict = np.load(file, allow_pickle=True).item()
    
    prob = chi2_scipy.sf(out_dict['chi2'], var_deg[ii]) * 100
    print(var, np.round(out_dict['chi2'], 1), f'{prob:.1e}')

    cosmo = set_cosmo(cosmo_label=var[4:])
    like_cosmo = CAMB_model.CAMBModel(np.array([3]), cosmo=cosmo)
    true_cosmo = like_cosmo.get_linP_params()

    consist = 0
    for key in ["Delta2_star", "n_star"]:
        print(np.round(out_dict[key], 3), np.round(out_dict["err_" + key], 3))
        print("diff", np.round(out_dict[key] - true_cosmo[key], 3), np.round(out_dict["err_" + key], 3))
        consist += (out_dict[key] - true_cosmo[key])**2/out_dict["err_" + key]**2

    prob_var = chi2_scipy.sf(consist, 2) * 100
    print(np.round(prob_var, 1))

    col = "C"+str(ii)
    ax.scatter(
        out_dict["Delta2_star"] - true_cosmo["Delta2_star"], 
        out_dict["n_star"] - true_cosmo["n_star"], 
        color=col, marker="x")

    for jj in range(1, 2):
        if jj == 1:
            lab = dict_trans[var]
        else:
            lab= None
        ax.plot(
            out_dict["xell"+str(jj)]- true_cosmo["Delta2_star"], 
            out_dict["yell"+str(jj)]- true_cosmo["n_star"], 
            col+ls[jj-1], lw=3, label=lab+" w/o errors")

    nseed = 400
    xy_all = np.zeros((nseed, 2))
    for jj in range(nseed):
        folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/"+var+"/CH24_mpgcen_gpr/"
        data_cen = np.load(folder + "seed_" + str(jj) + "/best_dircosmo.npy", allow_pickle=True).item()
        x = data_cen["mle_cosmo_cen"]["Delta2_star"] - true_cosmo["Delta2_star"]
        y = data_cen["mle_cosmo_cen"]["n_star"] - true_cosmo["n_star"]
        # print(data_cen["mle"]['$f_{\rm HCD1}_0$'])
        xy_all[jj, 0] = x
        xy_all[jj, 1] = y
        plt.scatter(x, y, marker=".", color="C1")


    # plot ellipse containing 68%
    mean = xy_all.mean(axis=0)
    cov = np.cov(xy_all, rowvar=False)
    rho = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])
    
    # Eigen-decomposition of covariance
    # vals, vecs = np.linalg.eigh(cov)
    # order = vals.argsort()[::-1]
    # vals, vecs = vals[order], vecs[:, order]
    
    # # Scale to the 68% quantile of chi-square with 2 dof
    # chi2_val = chi2_scipy.ppf(0.68, df=2)
    # _ = vals < 0
    # width, height = 2 * np.sqrt(vals * chi2_val)
    # angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
    # ellipse = Ellipse(mean, width, height, angle, edgecolor='red', facecolor='none', lw=2)
    # ax.add_patch(ellipse)
    
    plot_ellipse(
        np.sqrt(cov[0, 0]),
        np.sqrt(cov[1, 1]),
        rho,
        [mean[0], mean[1]],
        ax=ax,
        label=lab + " noisy realizations"
    )


ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k", linestyle="--")

ax.set_xlim(-0.1, 0.15)
ax.set_ylim(-0.1, 0.15)


ax.set_ylabel(r"$\Delta(n_\star)$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-2, loc="upper right")
plt.tight_layout()
# plt.savefig("figs/validation_2d.pdf")
# plt.savefig("figs/validation_2d.png")
# -



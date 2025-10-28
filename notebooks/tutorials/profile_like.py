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

# # Profile likelihood

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

# ### Get data

# +
fit_type = "global_opt"
data_lab = "DESIY1_QMLE3"
# data_lab = "DESIY1_QMLE"
# data_lab = "DESIY1_FFT3_dir"
# data_lab = "DESIY1_FFT_dir"
emu = "mpg"
# emu = "nyx"

variations = [
    # None,
    # "nyx",
    # "DESIY1_QMLE",
    # "DESIY1_FFT3_dir",
    # "DESIY1_FFT_dir",
    # "no_inflate",  # no increase errors
    # "no_emu_cov",  # no emu error
    # "no_inflate_no_emu_cov",  # no emu error, no increase errors for 3, 3.6, and 4
    # "cosmo",  # different fiducial cosmo
    # "cosmo_low",  # different fiducial cosmo
    "cosmo_high",  # different fiducial cosmo
    # "metal_trad",  # 2 params for metals like eBOSS
    # "metal_si2",  # no SiII-SiII cont
    # "metal_deco",  # no decorrelation metals
    # "metal_thin",  # no desviation from optically-thin limit
    # "no_res",  # no marginalize over resolution
    # "Turner24",  # mF from Turner24 with 1 free param to scale
    # "more_igm",  # 8 params for IGM evolution
    # "less_igm",  # 4 params for IGM evolution
    # "metals_z",  # 2 params for z ev metals
    # "hcd_z",  # 2 params for z ev hcd
    # "zmin",
    # "zmax",
    # "sim_mpg_central",
    # "sim_nyx_central",
    # "sim_sherwood",
]

variation = variations[0]
# variation = "no_inflate"

# variation = "sim_mpg_central"
# variation = "sim_nyx_central"
# variation = "sim_sherwood"

# type_prof = "prof_2d"
# nelem = 64

type_prof = "prof_2d_deep2"
nelem = 900

if variation is not None:
    folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+variation+"/CH24_"+emu+"cen_gpr/"
else:
    folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()
mle_cube_cen = data_cen["mle_cube"].copy()

chi2 = np.zeros(nelem)
params = np.zeros((nelem, 2))
mle_cube = np.zeros((nelem, len(mle_cube_cen)-2))
hcd0 = np.zeros((nelem))
tau3 = np.zeros((nelem))
all_pars = np.zeros((nelem, len(mle_cube_cen)+1))
mle = []
for ii in range(nelem):
    try:
        data = np.load(folder + type_prof + "/profile_"+str(ii)+ ".npy", allow_pickle=True).item()
    except:
        continue
    chi2[ii] = data["chi2"]
    params[ii, 0] = data["blind_cosmo"]["Delta2_star"]
    params[ii, 1] = data["blind_cosmo"]["n_star"]
    hcd0[ii] = data["mle"]['$f_{\rm HCD1}_0$']
    if variation != "Turner24":
        tau3[ii] = data["mle"]['$\tau_{\rm eff_3}$']

    jj = 0
    for key in data_cen["mle"]:
        if key in data["mle"]:
            all_pars[ii, jj] = data["mle"][key]
            jj+=1
    
    # mle_cube[ii] = data["mle_cube"]
    mle.append(data["mle"])
    # if data["mle"]['$f_{\rm HCD1}_0$'] > -1:
    #     if data["chi2"]-data_cen['best_chi2'] < 7:
    #         print(ii, data["mle"]['$f_{\rm HCD1}_0$'], params[ii, :], data["chi2"]-data_cen['best_chi2'])

_ = chi2 != 0
min_chi2 = np.min([chi2[_].min(), data_cen['best_chi2']])
print(min_chi2, data_cen['best_chi2']-min_chi2, chi2[_].min()-min_chi2, np.argmin(chi2))

np.sum(chi2 == 0)
# -

data_cen['best_chi2']

data_cen["mle"]


# +
def fix_key(s):
    # Re-encode the string so we see escape codes again
    return s.encode('unicode_escape').decode().replace('\\\\', '\\')

fixed = {fix_key(k): v for k, v in data_cen["mle"].items()}

# +
plot = True

if plot:

    fig, ax = plt.subplots(2, 3, figsize=(12, 10), sharex="col")
    ax = ax.reshape(-1)
    
    
    _ = chi2 != 0
    ax[0].scatter(params[_, 0], params[_, 1], c=chi2[_]-data_cen['best_chi2'], cmap="tab20")
    ax[1].scatter(params[_, 0], hcd0[_], c=chi2[_]-data_cen['best_chi2'], cmap="tab20")
    ax[2].scatter(params[_, 1], hcd0[_], c=chi2[_]-data_cen['best_chi2'], cmap="tab20")
    ax[4].scatter(params[_, 0], tau3[_], c=chi2[_]-data_cen['best_chi2'], cmap="tab20")
    ax[5].scatter(params[_, 1], tau3[_], c=chi2[_]-data_cen['best_chi2'], cmap="tab20")
    
    # ax[1].axhline(-1.5, c="k", lw=2)
    # ax[2].axhline(-1.5, c="k", lw=2)
    # ax[2].axvline(-2.25, c="k", lw=2)
    # ax[0].axhline(-2.25, c="k", lw=2)
    ax[3].set_xlabel("D2star")
    ax[4].set_xlabel("D2star")
    ax[5].set_xlabel("nstar")
    
    ax[0].set_ylabel("nstar")
    ax[1].set_ylabel("LLS")
    ax[2].set_ylabel("LLS")
    ax[4].set_ylabel("tau3")
    ax[5].set_ylabel("tau3")
    plt.tight_layout()
    # plt.savefig("figs/HCD_cosmo.pdf")

# +
plot = False

if plot:
    _ = chi2 - min_chi2 < 2.3
    
    ihcd = 1
    for jj in range(all_pars.shape[1]-3):
    # for jj in range(1):
        jj2 = 0
        for key in fixed:
            if jj2 == jj+2:
                key2 = key
            jj2 += 1
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 8), sharex="col")
        ax = ax.reshape(-1)
        ax[0].scatter(params[:, 0], params[:, 1], c=chi2-data_cen['best_chi2'], cmap="tab20")
        ax[1].scatter(params[:, 0], all_pars[:,jj], c=chi2-data_cen['best_chi2'], cmap="tab20")
        ax[2].scatter(params[:, 1], all_pars[:,jj], c=chi2-data_cen['best_chi2'], cmap="tab20")
        c1 = np.corrcoef(params[_,0], all_pars[_, jj])[1, 0]
        c2 = np.corrcoef(params[_,1], all_pars[_, jj])[1, 0]
        ax[1].text(
            0.1, 
            0.1, 
            "r=" + str(np.round(c1, 2)), 
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax[1].transAxes
        )
        ax[2].text(
            0.1, 
            0.1, 
            "r=" + str(np.round(c2, 2)), 
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax[2].transAxes
        )
        
        # if "\\" in key2:
        #     out = key2.replace('\\\\', '\\')
        # else:
        #     out = key2
        #     fig.suptitle(key2.encode('unicode_escape').decode())
        # else:
        #     fig.suptitle(key2)
        # print(out.encode('unicode_escape').decode())
        if "HCD" in key2:
            key2 = "HCD" + str(ihcd)
            ihcd += 1
    
        fig.suptitle(key2)
    
        
        plt.tight_layout()
        plt.savefig("fig_corr/"+str(jj)+".png")
        plt.close()

# +
# data_cen["mle"]
# -





# +
# data_cen["mle"]
# data_cen["mle_cube"]
# -

print(data_cen["mle_cosmo_cen"])
print(params[np.argmin(chi2)])
mle[np.argmin(chi2)]



# ## Get 1D errors

# +
out_dict = {}
n2d = int(np.sqrt(nelem))

xparams = params[:,0].reshape(n2d, n2d)
yparams = params[:,1].reshape(n2d, n2d)
zchi2 = chi2.reshape(n2d, n2d)
ind2 = np.argmin(chi2)

if min_chi2 == chi2.min():
    x_min = params[ind2, 0]
    y_min = params[ind2, 1]
else:
    x_min = data_cen["mle_cosmo_cen"]["Delta2_star"]
    y_min = data_cen["mle_cosmo_cen"]["n_star"]

out_dict["mle_cosmo_cen"] = {}
out_dict["mle_cosmo_cen"]["Delta2_star"] = data_cen["mle_cosmo_cen"]["Delta2_star"]
out_dict["mle_cosmo_cen"]["n_star"] = data_cen["mle_cosmo_cen"]["n_star"]

xinter = np.linspace(xparams[0,:].min(), xparams[0,:].max(), 1000)
chi2_inter = np.interp(xinter, xparams[0,:], zchi2.min(axis=0) - min_chi2)
_ = chi2_inter < 1
xerr = 0.5 * (xinter[_].max() - xinter[_].min())
print("Delta2_star", np.round(x_min, 3), np.round(xerr, 3))

yinter = np.linspace(yparams[:,0].min(), yparams[:,0].max(), 1000)
chi2_inter = np.interp(yinter, yparams[:,0], zchi2.min(axis=1) - min_chi2)
_ = chi2_inter < 1
yerr = 0.5 * (yinter[_].max() - yinter[_].min())
print("n_star", np.round(y_min, 3), np.round(yerr, 3))

out_dict["Delta2_star"] = x_min
out_dict["err_Delta2_star"] = xerr
out_dict["n_star"] = y_min
out_dict["err_n_star"] = yerr
out_dict["chi2"] = min_chi2

# +
plot = True

if plot:
    plt.plot(xparams[0,:], zchi2.min(axis=0) - min_chi2)
    _ = (zchi2.min(axis=0) - min_chi2) < 1
    sig_d2s = 0.5 * (xparams[0, _].max() - xparams[0, _].min())
    # print(np.round(sig_d2s, 3))
    # plt.axvline(xparams[0, _].max())
    # plt.axvline(xparams[0, _].min())
    
    xinter = np.linspace(xparams[0,:].min(), xparams[0,:].max(), 1000)
    chi2_inter = np.interp(xinter, xparams[0,:], zchi2.min(axis=0) - min_chi2)
    _ = chi2_inter < 1
    print(np.round(0.5 * (xinter[_].max() - xinter[_].min()), 3))
    plt.axvline(xinter[_].max(), color="C1")
    plt.axvline(xinter[_].min(), color="C1")

    plt.ylabel(r"$\Delta\chi2$")
    plt.xlabel(r"$\Delta^2_\star$")

    plt.axhline(1)
    plt.ylim(0, 20)
    
    # plt.savefig("figs/pl1d_D2s_qmle3.pdf")
    # plt.savefig("figs/pl1d_D2s_qmle3.png")

# +
plot = True

if plot:
    
    plt.plot(yparams[:,0], zchi2.min(axis=1) - min_chi2)
    _ = (zchi2.min(axis=1) - min_chi2) < 1
    # print(np.round(0.5 * (yparams[_, 0].max() - yparams[_, 0].min()), 3))
    # plt.axvline(yparams[_, 0].max())
    # plt.axvline(yparams[_, 0].min())
    
    yinter = np.linspace(yparams[:,0].min(), yparams[:,0].max(), 1000)
    chi2_inter = np.interp(yinter, yparams[:,0], zchi2.min(axis=1) - min_chi2)
    _ = chi2_inter < 1
    print(np.round(0.5 * (yinter[_].max() - yinter[_].min()), 3))
    plt.axvline(yinter[_].max(), color="C1")
    plt.axvline(yinter[_].min(), color="C1")
    
    plt.ylabel(r"$\Delta\chi2$")
    plt.xlabel(r"$n_\star$")
    
    plt.axhline(1)
    plt.ylim(0, 20)
    # plt.savefig("figs/pl1d_ns_qmle3.pdf")
    # plt.savefig("figs/pl1d_ns_qmle3.png")
# -

# #### Get correlation from 2d-ellipse

# +
ind = chi2 != 0
nelem = 50
grid = np.meshgrid(
    np.linspace(params[ind,0].min(), params[ind,0].max(), nelem), 
    np.linspace(params[ind,1].min(), params[ind,1].max(), nelem)
)
xi = np.zeros((nelem*nelem, 2))
xi[:,0] = grid[0].reshape(-1)
xi[:,1] = grid[1].reshape(-1)
interp = griddata(params[ind], chi2[ind], xi, method="linear")
ind2 = np.isfinite(interp)

print("grid", min_chi2, data_cen['best_chi2'])
print("cen", data_cen["mle_cosmo_cen"])
# min_chi2 = np.min([chi2[ind].min(), interp.min()])
vmin = 0
vmax = np.max([chi2[ind].max(), interp.max()]) - min_chi2
vmax = 10

CS = plt.contour(
        xi[:, 0].reshape(nelem, nelem), 
        xi[:, 1].reshape(nelem, nelem), 
        interp.reshape(nelem, nelem) - min_chi2, 
        chi2_levels,
        colors=["C0", "C1", "C2", "C3"],
    )

for jj in range(1, 0, -1):
    print(jj)
    p = CS.collections[jj].get_paths()
    x = []
    y = []
    for ii in range(len(p)):
        v = p[ii].vertices
        print(jj, ii, v.shape)
        if v.shape[0] > 10:
            x.append(v[:,0])
            y.append(v[:,1])
    x = np.concatenate(x)
    y = np.concatenate(y)
    
    xfit, yfit, rho = fit_ellipse(x, y)
    plt.plot(xfit, yfit, "C"+str(jj)+"--")

    out_dict["xell" + str(jj)] = xfit
    out_dict["yell" + str(jj)] = yfit
    out_dict["xcen_2d"] = xfit.mean()
    out_dict["ycen_2d"] = yfit.mean()
    out_dict["rho"] = rho

if (type_prof == "prof_2d_deep") | (type_prof == "prof_2d_deep2"):
    if variation is None:
        file = "out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
    else:
        file = "out_pl/"+ variation + ".npy"

    print(file)
    np.save(file, out_dict)


ind3 = np.argsort(chi2[ind])
print((params[ind])[ind3[:3]].mean(axis=0))
print((params[ind])[ind3[:3]])

# +

fact1 = 1
fact2 = 1
plot_ellipse(
    out_dict["err_Delta2_star"]* fact1,
    out_dict["err_n_star"]* fact2,
    out_dict["rho"],
    [out_dict["xcen_2d"], out_dict["ycen_2d"]]
)
plt.plot(out_dict["xell1"], out_dict["yell1"], color="C0")
plt.scatter(out_dict["Delta2_star"], out_dict["n_star"], marker="x", color="k")

# plt.xlim(0.4, 0.63)
# plt.ylim(-2.34, -2.2)
# -

print(out_dict["err_Delta2_star"], out_dict["err_n_star"])

# +
fig, ax = plt.subplots(1, figsize=(8, 6))
ftsize = 20

ind = chi2 != 0
xx = params[ind, 0]
yy = params[ind, 1]
delta_chi = chi2[ind] - min_chi2


CS = ax.contourf(
    xi[:, 0].reshape(nelem, nelem), 
    xi[:, 1].reshape(nelem, nelem), 
    interp.reshape(nelem, nelem) - min_chi2, 
    chi2_levels,
    colors=["C0", "C1", "C2", "C3", "C4"],
    alpha=0.8
)


for ii in range(len(chi2_levels)-1):
    ind = (delta_chi >= chi2_levels[ii]) & (delta_chi < chi2_levels[ii+1])
    # print(ind)
    ax.scatter(
        xx[ind], 
        yy[ind], 
        c="C"+str(ii), 
        marker="o", 
        alpha=1
    )
# ii = 5
# ind = (delta_chi >= chi2_levels[-1])
# ax.scatter(
#     xx[ind], 
#     yy[ind], 
#     c="C"+str(ii), 
#     marker="o", 
#     alpha=1
# )


# ax[0].scatter(
#     data_cen['mle_cosmo_cen']['Delta2_star'],
#     data_cen['mle_cosmo_cen']['n_star'],
#     c = "k",
#     marker = "X"
# )

# patch1 = mpatches.Patch(color='C0', label=r"0.5 $\sigma$")
patch2 = mpatches.Patch(color='C0', label=r"1 $\sigma$")
patch3 = mpatches.Patch(color='C1', label=r"2 $\sigma$")
patch4 = mpatches.Patch(color='C2', label=r"3 $\sigma$")
patch5 = mpatches.Patch(color='C3', label=r"4 $\sigma$")
patch6 = mpatches.Patch(color='C4', label=r"5 $\sigma$")
# patch7 = mpatches.Patch(color='C5', label=r">5 $\sigma$")

ax.text(0.05, 0.07, r"$\chi^2_\mathrm{min}$="+str(np.round(min_chi2, 1)), 
            transform=ax.transAxes, fontsize=ftsize)


ax.set_ylabel(r"$n_\star$", fontsize=ftsize)
ax.legend(handles=[patch2, patch3, patch4, patch5, patch6], fontsize=ftsize-2, loc="upper right")

ax.set_xlabel(r"$\Delta^2_\star$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)
# ax[jj].plot(x, y, "k")
ax.plot(xfit, yfit, "k--")


best_ell = {
    "Delta2_star": xfit.mean(),
    "n_star": yfit.mean(),
}
# print(np.round(xfit.mean(), 2), np.round(0.5 * (xfit.max()-xfit.min()), 2))
# print(np.round(yfit.mean(), 2), np.round(0.5 * (yfit.max()-yfit.min()), 2))

ind = chi2 != 0
ind_min = np.argmin(chi2[ind])
print(chi2[ind][ind_min], 
      np.round(params[ind, :][ind_min], 2), 
      np.round(0.5 * (xfit.max()-xfit.min()), 2),
      np.round(0.5 * (yfit.max()-yfit.min()), 2)
)

# plot_ellipse(
#     out_dict["err_Delta2_star"],
#     out_dict["err_n_star"],
#     out_dict["rho"],
#     [out_dict["xcen_2d"], out_dict["ycen_2d"]],
#     ax=ax[0]
# )

ax.scatter(out_dict["Delta2_star"], out_dict["n_star"], marker="x", color="k")

# ind_min = np.argmin(interp[ind2])
# print(interp[ind2][ind_min], np.round(xi[ind2, :][ind_min], 2))


plt.tight_layout()

if variation is not None:
    plt.savefig("figs/pl_qmle3_"+variation+".pdf")
    plt.savefig("figs/pl_qmle3_"+variation+".png")
else:
    plt.savefig("figs/pl_qmle3.pdf")
    plt.savefig("figs/pl_qmle3.png")
# -


# ## All together



out_dict.keys()

# #### with errors

# +

ftsize = 22
ls = ["-", "--"]

dict_trans = {
    "DESIY1_QMLE3_mpg":"Fiducial", 
    
    "DESIY1_QMLE_mpg":"Data: w/ low SNR", 
    "DESIY1_FFT3_dir_mpg": "Data: FFT",
    "zmin": "Data: $z \geq 2.6$",  # restricted zrange
    "zmax": "Data: $z \leq 3.4$",  # restricted zrange
    
    "no_inflate":"Cov: no extra 5%",
    "no_emu_cov":"Cov: no emu err", # no emu error
    "no_inflate_no_emu_cov":"Cov: no emu err, no extra 5%", 
    
    "DESIY1_QMLE3_nyx":"Model: emulator",
    "cosmo": "Model: $\omega_0\omega_a$CDM",  # different fiducial cosmo
    # "cosmo_low": "Model: $\Lambda$CDM, low $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    # "cosmo_high": "Model: $\Lambda$CDM, high $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    "cosmo_low": "Model: low $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    "cosmo_high": "Model: high $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    
    "more_igm": "Model: IGM $n_z=8$",  # 8 params for IGM evolution
    "less_igm": "Model: IGM $n_z=4$",  # 4 params for IGM evolution
    "metals_z": "Model: metals $n_z=2$",  # 2 params for z ev metals
    "hcd_z": "Model: HCD $n_z=2$",  # 2 params for z ev hcd
    
    # "Turner24": r"Model: Turner+24 $\bar F$",  # mF from Turner24 with 1 free param to scale ERROR
    "metal_trad": "Model: simple metal",  # 2 params for metals like eBOSS
    "metal_si2": "Model: no SiII-SiII",  # no SiII-SiII cont
    "metal_deco": "Model: no metal decorr",  # no decorrelation metals
    # "metal_thin": "Model: metal thin",  # no desviation from optically-thin limit ERROR
    "no_res": "Model: no resolution",  # no resolution correction
}

fname = ["data", "cov", "model", "modelz", "model_other"]


for image in range(2, 3):

    if image == 0:
        variations = ["DESIY1_QMLE3_mpg", "zmin", "zmax", "DESIY1_QMLE_mpg", "DESIY1_FFT3_dir_mpg"]
    elif image == 1:
        variations = ["DESIY1_QMLE3_mpg", "no_inflate", "no_emu_cov", "no_inflate_no_emu_cov"]
    elif image == 2:
        variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE3_nyx", "cosmo_low", "cosmo_high", "cosmo"]
    elif image == 3:
        variations = ["DESIY1_QMLE3_mpg", "more_igm", "less_igm", "metals_z", "hcd_z"]
    elif image == 4:
        variations = ["DESIY1_QMLE3_mpg", "no_res", "metal_deco", "metal_si2", "metal_trad"]
        # variations = ["DESIY1_QMLE3_mpg"]


    fig, ax = plt.subplots(figsize=(8, 6))
    
    var_deg = 638
    
    fit_type = "global_opt"
    x0 = 0
    y0 = 0
    for ii, var in enumerate(variations):
        print()
        data_lab = "DESIY1_QMLE3"
        emu = "mpg"
    
        if "nyx" in var:
            emu = "nyx"
        if "FFT3_dir" in var:
            data_lab = "DESIY1_FFT3_dir"
        if var == "DESIY1_QMLE_mpg":
            data_lab = "DESIY1_QMLE"
        
        if "DESIY1" in var:
            file = "out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
        else:
            file = "out_pl/"+ var + ".npy"
            
        out_dict = np.load(file, allow_pickle=True).item()
        
        prob = chi2_scipy.sf(out_dict['chi2'], var_deg)
        print(var, np.round(out_dict['chi2'], 1), f'{prob:.1e}')
        if ii == 0:
            dict_diff = {
                "Delta2_star": out_dict["Delta2_star"],
                "n_star": out_dict["n_star"],
                "err_Delta2_star": out_dict["err_Delta2_star"],
                "err_n_star": out_dict["err_n_star"],
                "xcen": out_dict["xcen_2d"],
                "ycen": out_dict["ycen_2d"],
            }
    
        consist = 0
        for key in ["Delta2_star", "n_star"]:
            print(np.round(out_dict[key], 3), np.round(out_dict["err_" + key], 3))
            print("diff", np.round(out_dict[key] - dict_diff[key], 3), np.round(out_dict["err_" + key], 3))
            consist += (out_dict[key] - dict_diff[key])**2/np.max([dict_diff["err_"+key], out_dict["err_" + key]])**2
    
        prob_var = chi2_scipy.sf(consist, 2)
        print(np.round(prob_var, 1))
    
        col = "C"+str(ii)
        # ax.scatter(out_dict["Delta2_star"], out_dict["n_star"], color=col, marker="x", alpha=0.75)
    
        for jj in range(1, 2):
            if jj == 1:
                lab = dict_trans[var]
            else:
                lab= None
                
            if var == "DESIY1_QMLE3_mpg":
                ax.fill(out_dict["xell"+str(jj)] - dict_diff["xcen"], out_dict["yell"+str(jj)] - dict_diff["ycen"], color=col, lw=3, label=lab, alpha=0.75)
            else:
                ax.plot(out_dict["xell"+str(jj)] - dict_diff["xcen"], out_dict["yell"+str(jj)] - dict_diff["ycen"], col+ls[jj-1], lw=3, label=lab)
    
    # ax.scatter(0.4, -2.28, color="k", marker="x", label="Planck")
    
    ax.set_ylabel(r"$n_\star$", fontsize=ftsize+2)
    ax.set_xlabel(r"$\Delta^2_\star$", fontsize=ftsize+2)
    ax.tick_params(
        axis="both", which="major", labelsize=ftsize - 2
    )
    ax.axhline(color="k", ls=":")
    ax.axvline(color="k", ls=":")

    if image == 2:
        loc = "lower left"
    else:
        loc = "upper right"
    
    
    plt.legend(fontsize=ftsize-4, loc=loc, ncol=1)
    plt.tight_layout()
    plt.savefig("figs/variations_2d_"+fname[image]+".pdf")
    plt.savefig("figs/variations_2d_"+fname[image]+".png")


# +
0.027 * 14

0.38

# +
0.017 * 14

0.2


# -

# #### No errors

# +
def format_last_column(values):
    """Format last column with trailing zeros or LaTeX scientific notation."""
    formatted = []
    for val in values:
        if abs(val) >= 1e-3:
            s = f"{val:.4f}"   # fixed 4 decimals
        else:
            coeff, exp = f"{val:.1e}".split("e")
            exp = int(exp)
            s = f"${coeff}\\times10^{{{exp}}}$"
        formatted.append(s)
    width = max(len(s) for s in formatted)
    return [f"{s:>{width}}" for s in formatted]


def format_column(values, sigfigs=2, force_decimals=True, one_decimal=False, two_decimals=False):
    formatted = []
    for val in values:
        if one_decimal:
            s = f"{val:.1f}"
        elif two_decimals:
            s = f"{val:.2f}"
        elif force_decimals:
            s = f"{val:.3f}"
        else:
            s = f"{val:.{sigfigs}g}"
        formatted.append(s)
    width = max(len(s) for s in formatted)
    return [f"{s:>{width}}" for s in formatted]



# -

from matplotlib import colormaps

# +
base_folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
data_lab = "DESIY1_QMLE3"

formats = [
    "{:>8.3f}",   # first number, 3 decimal places, right aligned width 8
    "{:>7.3f}",   # second number
    "{:>5.2f}",   # third number
    "{:>5.2f}",    # integer
    "{:>7.4f}"    # last number
]
table = []

fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
sym = ["x", "o"]
cmap = colormaps["tab20"]

variations = {
    "fid":[638, 1, "Fiducial"], # QMLE3
    "no_inflate":[638, 1, "Cov: no extra 5\%"],  # no increase errors
    "no_emu_cov":[638, 0, "Cov: no emu err"], # no emu error
    "no_inflate_no_emu_cov":[638, 0, "Cov: no emu err, no extra 5\%"], # no emu error, no increase errors
    "low_snr":[638, 1, "Data: w/ low SNR"], # QMLE
    "fft":[638, 1, "Data: FFT"], # FFT3_dir
    "fft_low_snr":[638, 1, "Data: FFT w/ low SNR"], # FFT_dir
    "cosmo":[638, 1, "Model: fid cosmo"],  # different fiducial cosmo
    "more_igm":[632, 1, "Model: IGM $n_z=8$"],  # 8 params for IGM evolution
    "less_igm":[644, 1, "Model: IGM $n_z=4$"],  # 4 params for IGM evolution
    "metals_z":[632, 1, "Model: metals $n_z=2$"],  # 2 params for z ev metals
    "hcd_z":[634, 1, "Model: HCD $n_z=2$"],  # 2 params for z ev hcd
    "Turner24":[643, 1, r"Model: Turner+24 $\bar F$"],  # mF from Turner24 with 1 free param to scale
    "metal_trad":[646, 0, "Model: simple metal"],  # 2 params for metals like eBOSS
    "metal_si2":[642, 0, "Model: no \siisii"],  # no SiII-SiII cont
    "metal_deco":[640, 0, "Model: no metal decorr"],  # no decorrelation metals
    "metal_thin":[640, 0, "Model: metal thin"],  # no desviation from optically-thin limit
    "no_res":[647, 0, "Model: no resolution"],  # no resolution correction
}


fit_type = "global_opt"

x0 = 0
y0 = 0
table = []
for ii, var in enumerate(variations):
    row = []
    
    print()
    if var == "fid":
        file = "out_pl/"+data_lab+"_mpg_" + fit_type + ".npy"
        key_chi2 = "chi2"
    elif var == "low_snr":
        file = base_folder + "/DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/best_dircosmo.npy"
        key_chi2 = "best_chi2"
    elif var == "fft":
        file = base_folder + "/DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/best_dircosmo.npy"
        key_chi2 = "best_chi2"
    elif var == "fft_low_snr":
        file = base_folder + "/DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/best_dircosmo.npy"
        key_chi2 = "best_chi2"
    else:
        file = base_folder + data_lab + "/"+var+"/CH24_mpgcen_gpr/best_dircosmo.npy"
        key_chi2 = "best_chi2"
        
    out_dict = np.load(file, allow_pickle=True).item()
    # if var == "metals_z":
    #     for key in out_dict["mle"]:
    #         print(key, out_dict["mle"][key])
    
    prob = chi2_scipy.sf(out_dict[key_chi2], variations[var][0])
    if prob < 1e-4:
        str_chi2 = np.round(out_dict[key_chi2], 1)
        str_prob = prob
    else:
        str_chi2 = np.round(out_dict[key_chi2], 1)
        str_prob = np.round(prob,4)
    print(var, str(str_chi2), prob)
    if ii == 0:
        dict_diff = {
            "Delta2_star": out_dict["Delta2_star"],
            "n_star": out_dict["n_star"],
            "err_Delta2_star": out_dict["err_Delta2_star"],
            "err_n_star": out_dict["err_n_star"],
        }
        print("err", 
              np.round(out_dict["err_Delta2_star"], 3), 
              np.round(out_dict["err_n_star"], 3)
             )
        
        # print(
        #     "check_ellipse",
        #       np.round(0.5 * (out_dict["xell1"].max()-out_dict["xell1"].min()), 2),
        #       np.round(0.5 * (out_dict["yell1"].max()-out_dict["yell1"].min()), 2)
        # )

    if ii > 0:
        row.append(variations[var][2])
        consist = 0
        for key in ["Delta2_star", "n_star"]:
            # print(np.round(out_dict[key], 3), np.round(out_dict["err_" + key], 3))
            diffx = np.round(out_dict['mle_cosmo_cen'][key] - dict_diff[key], 3)
            diffy = np.round(dict_diff["err_" + key], 3)
            row.append(diffx)
            # row.append(diffy)
            
            print("diff", 
                  diffx, 
                  diffy
                 )
            consist += (out_dict['mle_cosmo_cen'][key] - dict_diff[key])**2/dict_diff["err_"+key]**2

        prob_var = chi2_scipy.sf(consist, 2)
        print("prob_const", np.round(prob_var, 2))
        row.append(np.round(prob_var, 2))
        row.append(str_chi2)
        row.append(str_prob)
        table.append(row)
    
    

    col = cmap(ii)
    if ii == 0:
        labs = None
        # ax.scatter(
        #     out_dict["Delta2_star"]- dict_diff["Delta2_star"], 
        #     out_dict["n_star"] - dict_diff["n_star"], color=col, marker="s")
        ax.axhline(0, c="k", ls=":")
        ax.axvline(0, c="k", ls=":")
    else:
        labs = var
    # else:
        ax.scatter(
            out_dict['mle_cosmo_cen']["Delta2_star"] - dict_diff["Delta2_star"], 
            out_dict['mle_cosmo_cen']["n_star"] - dict_diff["n_star"], 
            color=col, 
            marker=sym[variations[var][1]], 
            label=labs
        )
        

    if ii == 0:
        for jj in range(1, 2):
            if jj == 1:
                lab = var
            else:
                lab= None
            ax.plot(
                out_dict["xell"+str(jj)] - dict_diff["Delta2_star"], 
                out_dict["yell"+str(jj)] - dict_diff["n_star"], 
                color=col, ls="-", lw=3, label=lab)



ax.set_ylabel(r"$\Delta(n_\star)$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-8, loc="upper right", ncol=1)
plt.tight_layout()
# -







# ### Validation



from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model

# 26 params

# +
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
ls = ["-", "--"]

variations = ["sim_mpg_central", "sim_nyx_central", "sim_sherwood"]
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
            col+ls[jj-1], lw=3, label=lab)


ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k", linestyle="--")



ax.set_ylabel(r"$\Delta(n_\star)$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-2)
plt.tight_layout()
# plt.savefig("figs/validation_2d.pdf")
# plt.savefig("figs/validation_2d.png")
# -

# # Profile above, MCMC below

# #### Contours from chains

from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model
import matplotlib.cm as cm



# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/"
folder = base + "sim_mpg_central/CH24_mpgcen_gpr/chain_2/"
dat_mpg_sim = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_sim = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_mpg_central_igm/CH24_mpgcen_gpr/chain_1/"
dat_mpg_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_igm = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_mpg_central_igm0/CH24_mpgcen_gpr/chain_1/"
dat_mpg_igm0 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg_igm0 = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "sim_nyx_central/CH24_mpgcen_gpr/chain_2/"
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
sum_mpg


# +
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"

folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_mpg = np.load(folder + "summary.npy", allow_pickle=True).item()
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_mpg_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/global_opt/CH24_nyxcen_gpr/chain_1/"
# dat_nyx = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
# sum_nyx = np.load(folder + "summary.npy", allow_pickle=True).item()

folder = base + "DESIY1_FFT3_dir/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_fft3 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_FFT_dir/global_opt/CH24_mpgcen_gpr/chain_1/"
# dat_fft = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_qmle = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
sum_qmle = np.load(folder + "summary.npy", allow_pickle=True).item()


folder = base + "DESIY1_QMLE3/no_emu_cov/CH24_mpgcen_gpr/chain_1/"
dat_no_emu_cov = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/zmax/CH24_mpgcen_gpr/chain_1/"
dat_zmax = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

folder = base + "DESIY1_QMLE3/zmin/CH24_mpgcen_gpr/chain_1/"
dat_zmin = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/Turner24/CH24_mpgcen_gpr/chain_1/"
# dat_turner = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"
# dat_cosmo = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
# folder = base + "DESIY1_QMLE3/cosmo/CH24_mpgcen_gpr/chain_1/"
# dat_cosmo_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"
# dat_cosmo_high = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
# folder = base + "DESIY1_QMLE3/cosmo_high/CH24_mpgcen_gpr/chain_1/"
# dat_cosmo_high_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"
# dat_cosmo_low = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
# folder = base + "DESIY1_QMLE3/cosmo_low/CH24_mpgcen_gpr/chain_1/"
# dat_cosmo_low_Asns = np.load(folder + "line_sigmas_Asns.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/hcd_z/CH24_mpgcen_gpr/chain_1/"
# dat_hcd_z = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/less_igm/CH24_mpgcen_gpr/chain_1/"
# dat_less_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/more_igm/CH24_mpgcen_gpr/chain_1/"
# dat_more_igm = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/metal_deco/CH24_mpgcen_gpr/chain_1/"
# dat_metal_deco = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/metal_si2/CH24_mpgcen_gpr/chain_1/"
# dat_metal_si2 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/metal_thin/CH24_mpgcen_gpr/chain_1/"
# dat_metal_thin = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/metal_trad/CH24_mpgcen_gpr/chain_1/"
# dat_metal_trad = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/metals_z/CH24_mpgcen_gpr/chain_1/"
# dat_metals_z = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/no_inflate/CH24_mpgcen_gpr/chain_1/"
# dat_no_inflate = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/no_inflate_no_emu_cov/CH24_mpgcen_gpr/chain_1/"
# dat_no_inflate_no_emu_cov = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/no_res/CH24_mpgcen_gpr/chain_1/"
# dat_no_res = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/DLAs/CH24_mpgcen_gpr/chain_1/"
# dat_dlas = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()


# folder = base + "DESIY1_QMLE3/kF_kms/CH24_mpgcen_gpr/chain_2/"
# dat_kF = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()

# folder = base + "DESIY1_QMLE3/HCD0/CH24_mpgcen_gpr/chain_2/"
# dat_HCD0 = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
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
    "DESIY1_FFT_dir_mpg":"Data: FFT w/ low SNR", 
    "zmin": "Data: $z \geq 2.6$",  # restricted zrange
    "zmax": "Data: $z \leq 3.4$",  # restricted zrange
    
    "no_emu_cov":"Cov: w/o emu err", # no emu error
    "no_inflate":"Cov: w/o 5% err",
    "no_inflate_no_emu_cov":"Cov: w/o emu, 5% err", 
    
    "DESIY1_QMLE3_nyx":"Emulator: lace-lyssa",
    
    "cosmo": "Cosmo: $\omega_0\omega_a$CDM",  # different fiducial cosmo
    "cosmo_low": "Cosmo: low $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    "cosmo_high": "Cosmo: high $\Omega_\mathrm{M}h^2$",  # different fiducial cosmo
    
    "more_igm": "IGM: $n_z=8$",  # 8 params for IGM evolution
    "less_igm": "IGM: $n_z=4$",  # 4 params for IGM evolution
    "Turner24": r"IGM: $\bar{F},\, n_z=1$",  # mF from Turner24 with 1 free param to scale ERROR
    "kF_kms": r"IGM: w/ $k_\mathrm{F}$",

    "hcd_z": "HCD: $n_z=2$",  # 2 params for z ev hcd
    "DLAs": "HCD: only DLAs",  # no LLS, sub-DLA
    "HCD0": "HCD: w/ $f_\\mathrm{const}^\\mathrm{HCD}$", # w/ constant term
    
    "metals_z": "Metals: $n_z=2$",  # 2 params for z ev metals
    
    "metal_trad": "Metals: traditional",  # 2 params for metals like eBOSS
    "metal_si2": "Metals: no SiII-SiII",  # no SiII-SiII cont
    "metal_deco": "Metals: no H-Si decorr",  # no decorrelation metals
    "metal_thin": "Metals: opt thin",  # no desviation from optically-thin limit ERROR
    
    "no_res": "Model: no resolution",  # no resolution correction

    "sim_mpg_central": "mpg-central", 
    "sim_mpg_central_all": "Model: cosmo, IGM, cont, syst", 
    "sim_mpg_central_igm": "Model: cosmo, IGM",
    "sim_mpg_central_igm0": "Model: cosmo", 
    "sim_nyx_central": "lyssa-central", 
    "sim_sherwood": "sherwood", 
}


fname = [
    "data_diff",
    "cov",
    "cosmo",
    "modelz",
    "model_ing_yes",
    "model_ing_no",
    "data",
    "emu",
    "cosmo_Asns",
    "DLAs",
    "val_sims",
    "val_sims_model",
    "test",
]

for image in range(10, 11):

    # if image in [3, 4, 5]:
    #     ftsize = 26
    # else:
    #     ftsize = 22
    factx = 1

    if image == 0:
        variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE_mpg", "DESIY1_FFT3_dir_mpg"]
        dats = [dat_mpg, dat_qmle, dat_fft3]
    elif image == 1:
        # variations = ["DESIY1_QMLE3_mpg", "no_inflate", "no_emu_cov", "no_inflate_no_emu_cov"]
        # dats = [dat_mpg, dat_no_inflate, dat_no_emu_cov, dat_no_inflate_no_emu_cov]
        variations = ["DESIY1_QMLE3_mpg", "no_emu_cov"]
        dats = [dat_mpg, dat_no_emu_cov]
    elif image == 2:
        variations = ["DESIY1_QMLE3_mpg", "cosmo", "cosmo_low", "cosmo_high"]
        dats = [dat_mpg, dat_cosmo, dat_cosmo_low, dat_cosmo_high]
    elif image == 3:
        variations = ["DESIY1_QMLE3_mpg", "more_igm", "less_igm", "kF_kms", "Turner24"]
        dats = [dat_mpg, dat_more_igm, dat_less_igm, dat_kF, dat_turner]
    elif image == 4:
        variations = ["DESIY1_QMLE3_mpg", "no_res", "metals_z", "metal_trad"]
        dats = [dat_mpg, dat_no_res, dat_metals_z, dat_metal_trad]
    elif image == 5:
        variations = ["DESIY1_QMLE3_mpg", "metal_thin", "metal_deco", "metal_si2"]
        dats = [dat_mpg, dat_metal_thin, dat_metal_deco, dat_metal_si2]
    elif image == 6:
        variations = ["DESIY1_QMLE3_mpg", "zmin", "zmax"]
        dats = [dat_mpg, dat_zmin, dat_zmax]
    elif image == 7:
        variations = ["DESIY1_QMLE3_mpg", "DESIY1_QMLE3_nyx"]
        dats = [dat_mpg, dat_nyx]
    elif image == 8:
        variations = ["DESIY1_QMLE3_mpg", "cosmo", "cosmo_low", "cosmo_high"]
        dats = [dat_mpg_Asns, dat_cosmo_Asns, dat_cosmo_low_Asns, dat_cosmo_high_Asns]
        factx = 1e9
    elif image == 9:
        variations = ["DESIY1_QMLE3_mpg", "hcd_z", "DLAs", "HCD0"]
        dats = [dat_mpg, dat_hcd_z, dat_dlas, dat_HCD0]
    elif image == 10:
        variations = ["sim_mpg_central", "sim_nyx_central", "sim_sherwood"]
        dats = [dat_mpg_sim, dat_nyx_sim, dat_sherwood]
    elif image == 11:
        variations = ["sim_mpg_central_all", "sim_mpg_central_igm", "sim_mpg_central_igm0"]
        dats = [dat_mpg_sim, dat_mpg_igm, dat_mpg_igm0]


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
            cosmo = set_cosmo(cosmo_label=var[4:])
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
    
    if fname[image] in ["cosmo", "modelz", "model_ing_yes", "model_ing_no", "cosmo_Asns", "DLAs"]:
        ymin, ymax = plt.ylim()
        yrange = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.2 * yrange)

    if fname[image] in ["modelz"]:
        ncol = 2
    else:
        ncol = 1
        
    if fname[image] in ["data"]:
        loc = "lower right"
    elif fname[image] in ["val_sims"]:
        loc = "upper left"
    else:
        loc = "upper right"
    
    plt.legend(fontsize=ftsize-6, loc=loc, ncol=ncol)
    plt.tight_layout()
    plt.savefig("figs/variations_"+fname[image]+".pdf")
    plt.savefig("figs/variations_"+fname[image]+".png")
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



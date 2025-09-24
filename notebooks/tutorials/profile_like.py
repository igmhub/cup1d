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
# data_lab = "DESIY1_FFT_dir"
# data_lab = "DESIY1_FFT"
emu = "mpg"
# emu = "nyx"

variation = None
# variation = "cov"
# variation = "sim_mpg_central"
# variation = "sim_nyx_central"
# variation = "sim_sherwood"

# type_prof = "prof_2d"
# nelem = 64

type_prof = "prof_2d_deep"
nelem = 900

if variation is not None:
    folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+variation+"/"
else:
    folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"+data_lab+"/"+fit_type+"/CH24_"+emu+"cen_gpr/"
data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()
mle_cube_cen = data_cen["mle_cube"].copy()

chi2 = np.zeros(nelem)
params = np.zeros((nelem, 2))
mle_cube = np.zeros((nelem, len(mle_cube_cen)-2))
hcd0 = np.zeros((nelem))
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
    # hcd0[ii] = data["mle"]['$\tau_{\rm eff_3}$']
    
    # mle_cube[ii] = data["mle_cube"]
    mle.append(data["mle"])
    if data["mle"]['$f_{\rm HCD1}_0$'] > -1:
        if data["chi2"]-data_cen['best_chi2'] < 7:
            print(ii, data["mle"]['$f_{\rm HCD1}_0$'], params[ii, :], data["chi2"]-data_cen['best_chi2'])

np.sum(chi2 == 0)
# -

data["mle"]

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].scatter(params[:, 0], params[:, 1], c=chi2-data_cen['best_chi2'], cmap="tab20")
ax[1].scatter(params[:, 0], hcd0, c=chi2-data_cen['best_chi2'], cmap="tab20")
ax[2].scatter(params[:, 1], hcd0, c=chi2-data_cen['best_chi2'], cmap="tab20")
# ax[1].axhline(-1.5, c="k", lw=2)
# ax[2].axhline(-1.5, c="k", lw=2)
# ax[2].axvline(-2.25, c="k", lw=2)
# ax[0].axhline(-2.25, c="k", lw=2)
ax[0].set_xlabel("D2star")
ax[0].set_ylabel("nstar")
ax[1].set_xlabel("D2star")
ax[2].set_xlabel("nstar")
ax[1].set_ylabel("LLS")
ax[2].set_ylabel("LLS")
plt.tight_layout()
# plt.savefig("figs/HCD_cosmo.pdf")





data_cen

# +
# data_cen["mle"]
# data_cen["mle_cube"]
# -

print(data_cen["mle_cosmo_cen"])
print(params[np.argmin(chi2)])
mle[np.argmin(chi2)]



min_chi2 = np.min([chi2.min(), data_cen['best_chi2']])
print(min_chi2, data_cen['best_chi2']-min_chi2, chi2.min()-min_chi2)

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

if type_prof == "prof_2d_deep":
    if variation is None:
        file = "out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
    else:
        file = "out_pl/"+ variation + ".npy"

    print(file)
    np.save(file, out_dict)


ind3 = np.argsort(chi2[ind])
print((params[ind])[ind3[:3]].mean(axis=0))
print((params[ind])[ind3[:3]])
# -

fact1 = 0.9
fact2 = 0.8
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

print(out_dict["err_Delta2_star"], out_dict["err_n_star"])

 0.11 0.05

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


plt.savefig("figs/pl_qmle3.pdf")
plt.savefig("figs/pl_qmle3.png")
# -


# ## All together

# +

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# -

out_dict.keys()

# #### with errors

# +


fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
ls = ["-", "--"]

variations = ["DESIY1_QMLE3_mpg", "DESIY1_FFT_dir_mpg", "DESIY1_QMLE_mpg", "DESIY1_FFT_mpg", "DESIY1_QMLE3_nyx", "cov"]
# variations = ["DESIY1_QMLE3_mpg", "DESIY1_FFT_dir_mpg", "DESIY1_QMLE_mpg", "DESIY1_FFT_mpg", "DESIY1_QMLE3_nyx"]
dict_trans = {
    "DESIY1_QMLE3_mpg":"QMLE3", 
    "DESIY1_FFT_dir_mpg":"FFTDM", 
    "DESIY1_QMLE_mpg":"QMLE", 
    "DESIY1_FFT_mpg":"FFT", 
    "DESIY1_QMLE3_nyx":"Emulator",
    "cov":"cov"
}
var_deg = 657

fit_type = "global_opt"
x0 = 0
y0 = 0
for ii, var in enumerate(variations):
    print()
    if var == "cov":
        file = "out_pl/"+ var + ".npy"
    else:
        file = "out_pl/"+ var + "_" + fit_type + ".npy"
    out_dict = np.load(file, allow_pickle=True).item()
    
    prob = chi2_scipy.sf(out_dict['chi2'], var_deg)
    print(var, np.round(out_dict['chi2'], 1), f'{prob:.1e}')
    if ii == 0:
        dict_diff = {
            "Delta2_star": out_dict["Delta2_star"],
            "n_star": out_dict["n_star"],
            "err_Delta2_star": out_dict["err_Delta2_star"],
            "err_n_star": out_dict["err_n_star"],
        }

    consist = 0
    for key in ["Delta2_star", "n_star"]:
        print(np.round(out_dict[key], 3), np.round(out_dict["err_" + key], 3))
        print("diff", np.round(out_dict[key] - dict_diff[key], 3), np.round(out_dict["err_" + key], 3))
        consist += (out_dict[key] - dict_diff[key])**2/np.max([dict_diff["err_"+key], out_dict["err_" + key]])**2

    prob_var = chi2_scipy.sf(consist, 2)
    print(np.round(prob_var, 1))

    col = "C"+str(ii)
    ax.scatter(out_dict["Delta2_star"], out_dict["n_star"], color=col, marker="x")

    for jj in range(1, 2):
        if jj == 1:
            lab = dict_trans[var]
        else:
            lab= None
        ax.plot(out_dict["xell"+str(jj)], out_dict["yell"+str(jj)], col+ls[jj-1], lw=3, label=lab)

# ax.scatter(0.4, -2.28, color="k", marker="x", label="Planck")

ax.set_ylabel(r"$n_\star$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta^2_\star$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-2)
plt.tight_layout()
# plt.savefig("figs/variations_2d.pdf")
# plt.savefig("figs/variations_2d.png")
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
    "fid":[638, 1, "Fiducial"],
    "no_inflate":[638, 1, "No inflate"],  # no increase errors for 3, 3.6, and 4
    "no_emu_cov":[638, 1, "No emu cov"], # no emu error
    "no_inflate_no_emu_cov":[638, 1, "No inflate, no emu"], # no emu error, no increase errors for 3, 3.6, and 4
    "cosmo":[638, 1, "Cosmo"],  # different fiducial cosmo
    "Turner24":[643, 1, r"Turner+24 $\bar F$"],  # mF from Turner24 with 1 free param to scale
    "more_igm":[632, 1, "IGM $n_z=8$"],  # 8 params for IGM evolution
    "less_igm":[644, 1, "IGM $n_z=4$"],  # 4 params for IGM evolution
    "metals_z":[632, 1, "Metals $n_z=2$"],  # 2 params for z ev metals
    "hcd_z":[634, 1, "HCD $n_z=2$"],  # 2 params for z ev hcd
    "metal_trad":[646, 0, "Simple metal"],  # 2 params for metals like eBOSS
    "metal_si2":[642, 0, "No \siisii"],  # no SiII-SiII cont
    "metal_deco":[640, 0, "No metal decorr"],  # no decorrelation metals
    "metal_thin":[640, 0, "Metal thin"],  # no desviation from optically-thin limit
    "no_res":[647, 0, "No resolution"],  # no resolution correction
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
    else:
        file = base_folder + data_lab + "/"+var+"/CH24_mpgcen_gpr/best_dircosmo.npy"
        key_chi2 = "best_chi2"
        
    out_dict = np.load(file, allow_pickle=True).item()
    if var == "metals_z":
        for key in out_dict["mle"]:
            print(key, out_dict["mle"][key])
    
    prob = chi2_scipy.sf(out_dict[key_chi2], variations[var][0])
    if prob < 1e-4:
        str_chi2 = np.round(out_dict[key_chi2], 1)
        str_prob = prob
    else:
        str_chi2 = np.round(out_dict[key_chi2], 1)
        str_prob = np.round(prob,4)
    print(var, str(str_chi2))
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
                color=col, ls=ls[jj-1], lw=3, label=lab)



ax.set_ylabel(r"$\Delta(n_\star)$", fontsize=ftsize)
ax.set_xlabel(r"$\Delta(\Delta^2_\star)$", fontsize=ftsize)
ax.tick_params(
    axis="both", which="major", labelsize=ftsize - 2
)

plt.legend(fontsize=ftsize-6, loc="upper right", ncol=3)
plt.tight_layout()

# +
# Transpose numeric columns
cols = list(zip(*[row[1:] for row in table]))

# Apply formatting rules per column
formatted_cols = [
    format_column(cols[0], force_decimals=True),   # col 2 (3 decimals)
    format_column(cols[1], force_decimals=True),   # col 3 (3 decimals)
    format_column(cols[2], two_decimals=True),     # col 4 (2 decimals)
    format_column(cols[3], one_decimal=True),      # col 5 (1 decimal)
    format_last_column(cols[4]),                   # col 6 (special rules)
]

# Rebuild table
tablex = []
for i, row in enumerate(table):
    label = f"{row[0]:<22}"
    nums = [col[i] for col in formatted_cols]
    line = label + " & " + " & ".join(nums) + "\\\\"
    tablex.append(line)

for line in tablex:
    print(line)
# -



out_dict



# ### Validation

from cup1d.likelihood.cosmologies import set_cosmo
from cup1d.likelihood import CAMB_model

# 26 params

# +
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
ls = ["-", "--"]

variations = ["sim_mpg_central", "sim_nyx_central", "sim_sherwood"]
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
plt.savefig("figs/validation_2d.pdf")
plt.savefig("figs/validation_2d.png")
# -

prob



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
cont = np.array([0, 1, 2, 3])
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
# data_lab = "DESIY1_QMLE3"
data_lab = "DESIY1_QMLE"
# data_lab = "DESIY1_FFT_dir"
# data_lab = "DESIY1_FFT"
# emu = "mpg"
emu = "nyx"

variation = None
# variation = "cov"

# type_prof = "prof_2d"
# nelem = 100

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
mle = []
for ii in range(nelem):
    try:
        data = np.load(folder + type_prof + "/profile_"+str(ii)+ ".npy", allow_pickle=True).item()
    except:
        continue
    chi2[ii] = data["chi2"]
    params[ii, 0] = data["blind_cosmo"]["Delta2_star"]
    params[ii, 1] = data["blind_cosmo"]["n_star"]
    mle_cube[ii] = data["mle_cube"]
    mle.append(data["mle"])

np.sum(chi2 == 0)
# -

data_cen["mle"]
# data_cen["mle_cube"]

ind = np.argmin(chi2)
mle[ind]

np.exp(-1)

min_chi2 = np.min([chi2.min(), data_cen['best_chi2']])
print(min_chi2, data_cen['best_chi2']-min_chi2, chi2.min()-min_chi2)

# ## Get 1D errors

# +
out_dict = {}

xparams = params[:,0].reshape(30, 30)
yparams = params[:,1].reshape(30, 30)
zchi2 = chi2.reshape(30, 30)
ind2 = np.argmin(chi2)

if min_chi2 == chi2.min():
    x_min = params[ind2, 0]
    y_min = params[ind2, 1]
else:
    x_min = data_cen["mle_cosmo_cen"]["Delta2_star"]
    y_min = data_cen["mle_cosmo_cen"]["n_star"]

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
    print(np.round(sig_d2s, 3))
    plt.axvline(xparams[0, _].max())
    plt.axvline(xparams[0, _].min())
    
    xinter = np.linspace(xparams[0,:].min(), xparams[0,:].max(), 1000)
    chi2_inter = np.interp(xinter, xparams[0,:], zchi2.min(axis=0) - min_chi2)
    _ = chi2_inter < 1
    print(np.round(0.5 * (xinter[_].max() - xinter[_].min()), 3))
    plt.axvline(xinter[_].max(), color="C1")
    plt.axvline(xinter[_].min(), color="C1")

    plt.axhline(1)

# +
plot = True

if plot:
    
    plt.plot(yparams[:,0], zchi2.min(axis=1) - min_chi2)
    _ = (zchi2.min(axis=1) - min_chi2) < 1
    print(np.round(0.5 * (yparams[_, 0].max() - yparams[_, 0].min()), 3))
    plt.axvline(yparams[_, 0].max())
    plt.axvline(yparams[_, 0].min())
    
    yinter = np.linspace(yparams[:,0].min(), yparams[:,0].max(), 1000)
    chi2_inter = np.interp(yinter, yparams[:,0], zchi2.min(axis=1) - min_chi2)
    _ = chi2_inter < 1
    print(np.round(0.5 * (yinter[_].max() - yinter[_].min()), 3))
    plt.axvline(yinter[_].max(), color="C1")
    plt.axvline(yinter[_].min(), color="C1")
    
    plt.axhline(1)
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
    p = CS.collections[jj].get_paths()
    x = []
    y = []
    for ii in range(len(p)):
        v = p[ii].vertices
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
        
    np.save(file, out_dict)


ind3 = np.argsort(chi2[ind])
print((params[ind])[ind3[:3]].mean(axis=0))
print((params[ind])[ind3[:3]])
# -

out_dict["err_Delta2_star"]

# +

plot_ellipse(
    out_dict["err_Delta2_star"],
    out_dict["err_n_star"],
    out_dict["rho"],
    [out_dict["xcen_2d"], out_dict["ycen_2d"]]
)
plt.plot(out_dict["xell1"], out_dict["yell1"], color="C0")
plt.scatter(out_dict["Delta2_star"], out_dict["n_star"], marker="x", color="k")

# +
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 6))
ftsize = 16

ind = chi2 != 0

ax[1].scatter(
    params[ind, 0], 
    params[ind, 1], 
    c=chi2[ind] - min_chi2, 
    cmap="tab10", 
    vmin=vmin, 
    vmax=vmax, 
    marker="o", 
    alpha=1
)

CS = ax[0].contourf(
    xi[:, 0].reshape(nelem, nelem), 
    xi[:, 1].reshape(nelem, nelem), 
    interp.reshape(nelem, nelem) - min_chi2, 
    chi2_levels,
    colors=["C0", "C1", "C2"],
)


ax[0].scatter(
    data_cen['mle_cosmo_cen']['Delta2_star'],
    data_cen['mle_cosmo_cen']['n_star'],
    c = "k",
    marker = "X"
)

# patch1 = mpatches.Patch(color='C0', label=r"0.5 $\sigma$")
patch2 = mpatches.Patch(color='C0', label=r"1 $\sigma$")
patch3 = mpatches.Patch(color='C1', label=r"2 $\sigma$")
patch4 = mpatches.Patch(color='C2', label=r"3 $\sigma$")

ax[0].text(0.05, 0.93, r"$\chi^2_\mathrm{min}$="+str(np.round(min_chi2, 1)), 
            transform=ax[0].transAxes, fontsize=ftsize)


ax[0].set_ylabel(r"$n_\star$", fontsize=ftsize)
ax[0].legend(handles=[patch2, patch3, patch4], fontsize=ftsize)

for jj in range(2):
    ax[jj].set_xlabel(r"$\Delta^2_\star$", fontsize=ftsize)
    ax[jj].tick_params(
        axis="both", which="major", labelsize=ftsize - 2
    )
    # ax[jj].plot(x, y, "k")
    ax[jj].plot(xfit, yfit, "k--")


best_ell = {
    "Delta2_star": xfit.mean(),
    "n_star": yfit.mean(),
}
print(np.round(xfit.mean(), 2), np.round(0.5 * (xfit.max()-xfit.min()), 2))
print(np.round(yfit.mean(), 2), np.round(0.5 * (yfit.max()-yfit.min()), 2))

ind_min = np.argmin(chi2[ind])
print(chi2[ind][ind_min], 
      np.round(params[ind, :][ind_min], 2), 
      np.round(0.5 * (xfit.max()-xfit.min()), 2),
      np.round(0.5 * (yfit.max()-yfit.min()), 2)
)

plot_ellipse(
    out_dict["err_Delta2_star"],
    out_dict["err_n_star"],
    out_dict["rho"],
    [out_dict["xcen_2d"], out_dict["ycen_2d"]],
    ax=ax[0]
)

# ind_min = np.argmin(interp[ind2])
# print(interp[ind2][ind_min], np.round(xi[ind2, :][ind_min], 2))


plt.tight_layout()


# plt.savefig("compare_variations.pdf")
# -


# ## All together

# +

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# +
fig, ax = plt.subplots(figsize=(8, 6))
ftsize = 20
ls = ["-", "--"]

variations = ["DESIY1_QMLE3_mpg", "DESIY1_FFT_dir_mpg", "DESIY1_QMLE_mpg", "DESIY1_FFT_mpg", "DESIY1_QMLE3_nyx", "cov"]
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
    prob = chi2_scipy.sf(out_dict['chi2'], var_deg) * 100
    print(var, np.round(out_dict['chi2'], 1), f'{prob:.1e}')
    if ii == 0:
        dict_diff = {
            "x": out_dict["xbest"],
            "y": out_dict["ybest"]
        }

    consist = 0
    for key in ["x", "y"]:
        err1 = 0.5 * (out_dict[key+"ell1"].max()-out_dict[key+"ell1"].min())
        err2 = 0.5 * (out_dict[key+"ell2"].max()-out_dict[key+"ell2"].min())
        print(np.round(out_dict[key+"best"], 2), np.round(err1, 2), np.round(err2, 2))
        print("diff", np.round(out_dict[key+"best"] - dict_diff[key], 3), np.round(err1, 3))
        if ii == 0:
            dict_diff["e"+key] = err1
        consist += (out_dict[key+"best"] - dict_diff[key])**2/np.max([dict_diff["e"+key], err1])**2

    prob_var = chi2_scipy.sf(consist, 2) * 100
    print(np.round(prob_var, 1))

    col = "C"+str(ii)
    ax.scatter(out_dict["xbest"], out_dict["ybest"], color=col, marker="x")

    for jj in range(1, 2):
        if jj == 1:
            lab = dict_trans[var]
        else:
            lab= None
        ax.plot(out_dict["xell"+str(jj)], out_dict["yell"+str(jj)], col+ls[jj-1], label=lab)

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






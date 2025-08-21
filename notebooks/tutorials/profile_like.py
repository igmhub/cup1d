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
from cup1d.utils.fit_ellipse import fit_ellipse
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
emu = "mpg"
# emu = "nyx"

# type_prof = "prof_2d"
# nelem = 100

type_prof = "prof_2d_deep2"
nelem = 900


folder = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/DESIY1_QMLE3/"+fit_type+"/CH24_"+emu+"cen_gpr/"
chi2 = np.zeros(nelem)
params = np.zeros((nelem, 2))
for ii in range(nelem):
    try:
        data = np.load(folder + type_prof + "/profile_"+str(ii)+ ".npy", allow_pickle=True).item()
    except:
        continue
    chi2[ii] = data["chi2"]
    params[ii, 0] = data["blind_cosmo"]["Delta2_star"]
    params[ii, 1] = data["blind_cosmo"]["n_star"]

data_cen = np.load(folder + "best_dircosmo.npy", allow_pickle=True).item()
# -

# ### Interpolate data to get better fit

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

min_chi2 = np.min([chi2[ind].min(), data_cen['best_chi2'], interp[ind2].min()])
print(chi2[ind].min(), data_cen['best_chi2'])
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

p = CS.collections[1].get_paths()
x = []
y = []
for ii in range(len(p)):
    v = p[ii].vertices
    x.append(v[:,0])
    y.append(v[:,1])
x = np.concatenate(x)
y = np.concatenate(y)

xfit, yfit = fit_ellipse(x, y)

# plt.plot(x, y, "k")
plt.plot(xfit, yfit, "k--")

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


# ind_min = np.argmin(interp[ind2])
# print(interp[ind2][ind_min], np.round(xi[ind2, :][ind_min], 2))


plt.tight_layout()

ind = np.argsort(chi2)
print(params[ind[:3]].mean(axis=0))

# plt.savefig("compare_variations.pdf")
# -



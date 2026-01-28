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
# # Compare historical CMB chains
#
# Compare linear power parameters from WMAP9 and Planck (2013, 2015, 2018).
#
# All chains have free neutrino mass, CMB info only.

# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2
import numpy as np
import os
from getdist import plots
import matplotlib.pyplot as plt
from cup1d.planck import planck_chains
from cup1d.likelihood import marg_lya_like

from cup1d.utils.utils import get_path_repo


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# rcParams["text.usetex"] = True

# %%

# base = "/home/jchaves/Proyectos/projects/lya/cup1d/data/planck_linP_chains/crisjagq"
# model = "base_mnu"
# data = "desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing"

# cmb = planck_chains.get_cobaya(base, model, data, linP_tag=None)

# %%

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/cup1d/data/planck_linP_chains/crisjagq/base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing_linP/base_mnu_desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing_linP"
# cmb = {} 
# cmb["samples"] = loadMCSamples(folder)

# %%
# fontsize = 14
# g = plots.getSubplotPlotter(width_inch=10)
# g.settings.num_plot_contours = 2
# g.settings.axes_fontsize = fontsize-6
# g.settings.legend_fontsize = fontsize-6

# # from Planck
# g.triangle_plot(
#     [cmb["samples"]],
#     ["linP_DL2_star", "linP_n_star", "linP_DL2_star2", "linP_n_star2","mnu"],
# )

# %%

# %%

# %% [markdown]
# ### Read extended CMB chains from historical releases
#
# These chains are already provided in cup1d, form different CMB releases. All chains have free neutrino mass.

# %% jupyter={"outputs_hidden": false}
root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")
cmb = planck_chains.get_planck_2018(
    model='base',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)


# %%
thresholds = [2.30, 6.18]
# thresholds = [2.30]
# coeff_mult = 0.5 # to get log-like, not to plot Eq. 1, https://arxiv.org/abs/2303.00746
neff_grid, DL2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]
chi2_Mc2005 = marg_lya_like.gaussian_chi2_McDonald2005(neff_grid, DL2_grid)
chi2_PD2015 = marg_lya_like.gaussian_chi2_PalanqueDelabrouille2015(neff_grid, DL2_grid)
chi2_Ch2019 = marg_lya_like.gaussian_chi2_Chabanier2019(neff_grid, DL2_grid)
chi2_Wa2024 = marg_lya_like.gaussian_chi2_Walther2024(neff_grid, DL2_grid, ana_type="priors")
# chi2_Wa2024_np=coeff_mult * marg_lya_like.gaussian_chi2_Walther2024(neff_grid, DL2_grid, ana_type="no_priors")


# %%
np.mean(cmb["samples"]['linP_DL2_star'])

# %%
import cup1d, os
import numpy as np

path_out = os.path.join(os.path.dirname(cup1d.__path__[0]), "data", "zenodo")
store_data = {
    "black":{
        "x": np.mean(cmb["samples"]['linP_DL2_star']),
        "xerr": np.std(cmb["samples"]['linP_DL2_star']),
        "y": np.mean(cmb["samples"]['linP_n_star']),
        "yerr": np.std(cmb["samples"]['linP_n_star']),
        "r": np.corrcoef(
                cmb["samples"]['linP_DL2_star'], 
                cmb["samples"]['linP_n_star']
            )[0,1],
    },
    "blue":{
        "x": 0.379,
        "xerr": 0.032,
        "y": -2.309,
        "yerr": 0.019,
        "r":-0.1738,
    },
    "orange":{
        "x": 0.47,
        "xerr": 0.06,
        "y": -2.3,
        "yerr": 0.055,
        "r":0.6,
    },
    "green":{
        "x": 0.32,
        "xerr": 0.03,
        "y": -2.36,
        "yerr": 0.01,
        "r":0.55,
    },
    "red":{
        "x": 0.310,
        "xerr": 0.020,
        "y": -2.340,
        "yerr": 0.006,
        "r":0.512,
    },
    "purple":{
        "x": 0.388,
        "xerr": 0.045,
        "y": -2.2978,
        "yerr": 0.0067,
        "r":0.632,
    },
}
fname = os.path.join(path_out, "fig_18.npy")
np.save(fname, store_data)

# %%
store_data

# %%
import scipy.stats as stats
import matplotlib.lines as mlines

# %%
base_notebook = "/home/jchaves/Proyectos/projects/lya/cup1d/notebooks/tutorials/"
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
dat_blob = np.load(folder + "blobs.npy")
sum_mpg = np.load(folder + "summary.npy", allow_pickle=True).item()

delta2_star = dat_blob["Delta2_star"].reshape(-1)
n_star = dat_blob["n_star"].reshape(-1)

ds = delta2_star
ns = n_star

per_ds = np.percentile(ds, [5, 16, 50, 84, 95])
per_ns = np.percentile(ns, [5, 16, 50, 84, 95])
_ = (ds > per_ds[0]) & (ds < per_ds[-1]) & (ns > per_ns[0]) & (ns < per_ns[-1])
corr = np.corrcoef(ds[_], ns[_])
# corr = np.corrcoef(ds, ns)
r = corr[0, 1]
r

# %%
fake_blinding = {
    'Delta2_star': sum_mpg["delta2_star_16_50_84"][1]-np.median(cmb["samples"]["linP_DL2_star"]),
     'n_star': sum_mpg["n_star_16_50_84"][1]-np.median(cmb["samples"]["linP_n_star"]),
}
real_blinding = np.load(base_notebook + "blinding.npy", allow_pickle=True).item()

blinding = real_blinding
# blinding = fake_blinding

# %%
dict_out = {
    "Delta2_star":ds - blinding["Delta2_star"],
    "n_star":ns - blinding["n_star"],
}
np.save("final_desi-dr1.npy", dict_out)

# %%

dict_out = np.load("final_desi-dr1.npy", allow_pickle=True).item()
dict_out.keys()

# %%
desi_dr1 = {
    "Delta2_star":sum_mpg["delta2_star_16_50_84"][1] - blinding["Delta2_star"],
    "n_star":sum_mpg["n_star_16_50_84"][1] - blinding["n_star"],
    "r":r,
    "Delta2_star_err":sum_mpg['delta2_star_err'],
    "n_star_err":sum_mpg["n_star_err"],
}

# %%
desi_dr1

# %%

# %%
from scipy.stats import gaussian_kde
import numpy as np

h_d2s, bin_d2s = np.histogram(delta2_star, bins=50)
hist_d2s_x = 0.5 * (bin_d2s[:-1] + bin_d2s[1:])
hist_d2s_y = h_d2s/h_d2s.max()

h_d2s, bin_d2s = np.histogram(n_star, bins=50)
hist_ns_x = 0.5 * (bin_d2s[:-1] + bin_d2s[1:])
hist_ns_y = h_d2s/h_d2s.max()

# Delta2_star KDE
kde_d2s = gaussian_kde(delta2_star)
x_d2s = np.linspace(hist_d2s_x.min(), hist_d2s_x.max(), 200)
y_d2s = kde_d2s(x_d2s)
y_d2s /= y_d2s.max()  # normalize like histogram

# n_star KDE
kde_ns = gaussian_kde(n_star)
x_ns = np.linspace(hist_ns_x.min(), hist_ns_x.max(), 200)
y_ns = kde_ns(x_ns)
y_ns /= y_ns.max()

# %%
from cup1d.likelihood.cosmologies import set_cosmo
mpg_all = set_cosmo("mpg_0", return_all=True)

# %% jupyter={"outputs_hidden": false}
labs = [
    'DESI-DR1 (this work)', 
    'SDSS (McDonald+05)', 
    'BOSS\n(Palanque-Delabrouille+15)', 
    'eBOSS (Chabanier+19)', 
    'eBOSS + {$\\bar{F}, \\Omega_\\mathrm{m}, H_0$} priors\n(Walther+24)'
]
cmap = plt.colormaps["Blues"]
fontsize = 26
lw = [3, 2]
col = [0.7, 0.3]
alpha=[0.7, 0.7]

g = plots.getSubplotPlotter(width_inch=10)
g.settings.num_plot_contours = 2
g.settings.axes_fontsize = fontsize-6
g.settings.legend_fontsize = fontsize-6

# from Planck
g.triangle_plot(
    [cmb['samples']],
    ['linP_DL2_star', 'linP_n_star'],
    # legend_labels=[r'Planck+18 $\Lambda$CDM'],
    lws=lw,
    line_args={"color": "k", "lw": lw[1], "alpha": alpha[0]},
)


ax = g.subplots[1, 0]
ax_NS = g.subplots[1, 1]
ax_D2S = g.subplots[0, 0]
x_D2S = np.linspace(0.2, 0.55, 500)
x_NS = np.linspace(-2.4, -2.2, 500)

# from literature

chi2_list = [chi2_Mc2005, chi2_PD2015, chi2_Ch2019, chi2_Wa2024]
colors = ["C1", "C2", "C3", "C4"]
lss = [":", "-.", "--", "--"]
for ii, _chi2 in enumerate(chi2_list):
    color = colors[ii]
    CS = ax.contour(DL2_grid, neff_grid, _chi2["chi2"], levels=thresholds, colors=color, linewidths=lw, alpha=alpha[0], linestyles = lss[ii])
    pdf = stats.norm.pdf(x_D2S, _chi2["Delta2_star"], _chi2["Delta2_star_err"])
    ax_D2S.plot(x_D2S, pdf/pdf.max(),color=color, lw=lw[0], alpha=alpha[0], ls = lss[ii])
    pdf = stats.norm.pdf(x_NS, _chi2["n_star"], _chi2["n_star_err"])
    ax_NS.plot(x_NS, pdf/pdf.max(),color=color, lw=lw[0], alpha=alpha[0], ls = lss[ii])

#### From DESI-DR1
for inum, num in enumerate([0.68, 0.95]):
    if inum == 0:
        label="This work"
    else:
        label=None
    for jj in range(len(dat_mpg[num])):
        x = dat_mpg[num][jj][0] - blinding["Delta2_star"]
        y = dat_mpg[num][jj][1] - blinding["n_star"]
        ax.plot(x, y, color=cmap(col[inum]), label=label, lw=lw[inum], alpha=alpha[inum])
        ax.fill(x, y, color=cmap(col[inum]), alpha=alpha[inum])

ax_D2S.plot(x_d2s - blinding["Delta2_star"], y_d2s, color=cmap(col[0]), lw=lw[0])
ax_NS.plot(x_ns - blinding["n_star"], y_ns, color=cmap(col[0]), lw=lw[0])


for lab in mpg_all:
    if lab[-1].isdigit() | (lab == "mpg_central"):
        _cosmo = mpg_all[lab]
        ax.scatter(_cosmo["star_params"]['Delta2_star'], _cosmo["star_params"]['n_star'], color="C0")

# format plot

for ii in range(len(labs)):
    ax.axhline(y=1,color='C'+str(ii),label=labs[ii])

ax.set_ylim(-2.39, -2.24)
ax.set_xlim(0.23, 0.52)


 # Legend
handles = []
handles.append(mlines.Line2D([], [], color='black', label=r'$\mathit{Planck}$ T&E: $\Lambda$CDM', lw=3))
handles.append(mlines.Line2D([], [], color="C0", label=labs[0], lw=3, ls = "-"))
for ii in range(len(labs)-1):
    handles.append(mlines.Line2D([], [], color=colors[ii], label=labs[ii+1], lw=3, ls = lss[ii]))
    
g.subplots[1, 1].legend(
    handles=handles, 
    bbox_to_anchor=(1, 2),
    loc='upper right', 
    borderaxespad=0.,
    fontsize=fontsize-6
)

ax.tick_params(
    axis="both", which="major", labelsize=fontsize
)
ax_NS.tick_params(
    axis="both", which="major", labelsize=fontsize
)
ax.set_xticks([0.3, 0.4, 0.5])
ax.set_yticks([-2.35, -2.30, -2.25])
ax_NS.set_xticks([-2.35, -2.30, -2.25])


ax.set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
ax.set_ylabel(r"$n_\star$", fontsize=fontsize)
ax_NS.set_xlabel(r"$n_\star$", fontsize=fontsize)

# ax.grid()

for ax in g.subplots[-1]:  # last row of panels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')


plt.tight_layout()
plt.savefig("figs/star_literature.png")
plt.savefig("figs/star_literature.pdf", bbox_inches="tight")

# %%

# %%

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
from getdist import plots,loadMCSamples
import matplotlib.pyplot as plt
from cup1d.planck import planck_chains
from cup1d.likelihood import marg_lya_like
# because of black magic, getdist needs this strange order of imports
# %matplotlib inline
from cup1d.utils.utils import get_path_repo


from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# rcParams["text.usetex"] = True

# %% [markdown]
# ### Read extended CMB chains from historical releases
#
# These chains are already provided in cup1d, form different CMB releases. All chains have free neutrino mass.

# %% jupyter={"outputs_hidden": false}
# model='base_mnu'
model='base'
# WMAP9
# wmap9=planck_chains.get_planck_2013(model=model,data='WMAP')
# wmap9['label']='WMAP 9'
# # Planck 2013
# planck2013=planck_chains.get_planck_2013(model=model)
# planck2013['label']='Planck 2013'
# # Planck 2015
# planck2015=planck_chains.get_planck_2015(model=model)
# planck2015['label']='Planck 2015'
# Planck 2018
# planck2018=planck_chains.get_planck_2018(model=model)
# planck2018['label']='Planck 2018'

root_dir=os.path.join(get_path_repo("cup1d"), "data", "planck_linP_chains")
cmb = planck_chains.get_planck_2018(
    model='base',
    data='plikHM_TTTEEE_lowl_lowE_linP',
    root_dir=root_dir,
    linP_tag=None
)


# %%
# # put everything together to pass to getdist
# analyses=[wmap9,planck2013,planck2015,planck2018]
# samples=[analysis['samples'] for analysis in analyses]
# labels=[analysis['label'] for analysis in analyses]

# %%

# params=['w', 'wa']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(cmbw['samples'],params)

# %% jupyter={"outputs_hidden": false}
# # plot traditional parameters
# params=['omegach2','omegabh2','mnu','H0','tau','logA','ns']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(samples,params,legend_labels=labels)

# %% [markdown]
# ### Plot linear power parameters

# %% jupyter={"outputs_hidden": false}
# # plot traditional parameters
# params=['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(samples,params,legend_labels=labels)

# %% jupyter={"outputs_hidden": false}
# # plot traditional parameters (including now Omega_m and m_nu)
# params=['linP_DL2_star','linP_n_star','linP_alpha_star','linP_f_star','linP_g_star','omegam','mnu']
# g = plots.getSubplotPlotter(width_inch=14)
# g.settings.axes_fontsize = 12
# g.settings.legend_fontsize = 16
# g.triangle_plot(samples,params,legend_labels=labels)

# %% [markdown]
# ### Plot linear power parameters from chain and from Lya likelihoods

# %%
# fit_type = "global_opt"
# data_lab = "DESIY1_QMLE3"
# emu = "mpg"
# file = "../tutorials/out_pl/"+ data_lab + "_" + emu + "_" + fit_type + ".npy"
# out_dict = np.load(file, allow_pickle=True).item()

# %%
# true_DL2=0.35
# true_neff=-2.3

# neff_err =out_dict["err_n_star"]
# DL2_err =out_dict["err_Delta2_star"]
# r = out_dict["rho"]

# DL2_err = 0.056
# neff_err = 0.022
# r=-0.134

# create grid (note j in number of elements, crazy python)
# thresholds = [2.30,6.17,11.8]
thresholds = [2.30, 6.18]
# thresholds = [2.30]
coeff_mult = 0.5 # to get log-like, not to plot Eq. 1, https://arxiv.org/abs/2303.00746
coeff_mult = 1
neff_grid, DL2_grid = np.mgrid[-2.4:-2.2:200j, 0.2:0.65:200j]
chi2_Mc2005=coeff_mult * marg_lya_like.gaussian_chi2_McDonald2005(neff_grid, DL2_grid)
chi2_PD2015=coeff_mult * marg_lya_like.gaussian_chi2_PalanqueDelabrouille2015(neff_grid, DL2_grid)
chi2_Ch2019=coeff_mult * marg_lya_like.gaussian_chi2_Chabanier2019(neff_grid, DL2_grid)
chi2_Wa2024=coeff_mult * marg_lya_like.gaussian_chi2_Walther2024(neff_grid, DL2_grid, ana_type="priors")
chi2_Wa2024_np=coeff_mult * marg_lya_like.gaussian_chi2_Walther2024(neff_grid, DL2_grid, ana_type="no_priors")


# chi2_desi = marg_lya_like.gaussian_chi2(neff_grid, DL2_grid, out_dict["ycen_2d"], out_dict["xcen_2d"], out_dict["err_n_star"], out_dict["err_Delta2_star"], out_dict["rho"])
# chi2_desi = marg_lya_like.gaussian_chi2(neff_grid, DL2_grid, true_neff, true_DL2, )

# %%
# CS = plt.contour(DL2_grid, neff_grid,chi2_Ch2019,levels=thresholds,colors='C3')
# plt.xlim(0.25, 0.45)
# plt.ylim(-2.45, -2.25)

# xcen, ycen, xerr, yerr = print_err(CS)
# pdf = stats.norm.pdf(x_D2S, xcen, xerr)
# ax_D2S.plot(x_D2S, pdf/pdf.max(),color='C3')
# pdf = stats.norm.pdf(x_NS, ycen, yerr)
# ax_NS.plot(x_NS, pdf/pdf.max(),color='C3')

# %%
# plt.contour(DL2_grid, neff_grid,chi2_desi,levels=thresholds[:2],colors='C0')
# plt.xlim(0.4, 0.63)
# plt.ylim(-2.34, -2.2)

# %%
def print_err(ell):
    # ell is a QuadContourSet from plt.contour or corner
    paths = ell.get_paths()
    if not paths:
        raise ValueError("No paths found in contour set")
    
    # take the first contour path
    ver = paths[0].vertices
    xcen = ver[:, 0].mean()
    ycen = ver[:, 1].mean()
    xerr = 0.5 * (ver[:, 0].max() - ver[:, 0].min())
    yerr = 0.5 * (ver[:, 1].max() - ver[:, 1].min())

    print(np.round(xerr, 3), np.round(yerr, 3))
    return xcen, ycen, xerr, yerr

# %%

# g = plots.getSinglePlotter(width_inch=8)
# g.plot_2d(cmb, ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25, 0.2,0.5])


# g = plots.getSubplotPlotter(width_inch=8)
# g.settings.axes_fontsize = 10
# g.settings.legend_fontsize = 14
# g.triangle_plot([cmb['samples']],
#                 ['linP_n_star', 'linP_DL2_star'],
#                 legend_labels=[r'$\Lambda$CDM'])

# %%
import scipy.stats as stats
import matplotlib.lines as mlines

# %%
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_1/"
dat_mpg = np.load(folder + "line_sigmas.npy", allow_pickle=True).item()
dat_blob = np.load(folder + "blobs.npy")

# %%
delta2_star = dat_blob["Delta2_star"].reshape(-1)
n_star = dat_blob["n_star"].reshape(-1)

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

true_cosmo = {
    'Delta2_star': delta2_star.mean()-np.median(cmb["samples"]["linP_DL2_star"]),
     'n_star': n_star.mean()-np.median(cmb["samples"]["linP_n_star"]),
}


# %% jupyter={"outputs_hidden": false}
labs = ['DESI-DR1 (this work)', 'SDSS (McDonald+05)', 'BOSS\n(Palanque-Delabrouille+15)', 'eBOSS (Chabanier+19)', 'eBOSS + {$\\bar{F}, \\Omega_m, H_0$} priors\n(Walther+24)']
cmap = plt.colormaps["Blues"]
fontsize = 26
lw = [3, 2]
col = [0.7, 0.3]

# specify CMB chain to plot
# cmb=wmap9
# g = plots.getSinglePlotter(width_inch=8)
# g.plot_2d(cmb, ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25, 0.2,0.5])
# g.plot_2d(cmb['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])
#g.plot_2d(planck2018_mnu_BAO['samples'], ['linP_n_star', 'linP_DL2_star'],lims=[-2.4,-2.25,0.2,0.5])

g = plots.getSubplotPlotter(width_inch=10)
g.settings.num_plot_contours = 2

g.settings.axes_fontsize = fontsize-6
g.settings.legend_fontsize = fontsize-6
g.triangle_plot(
    [cmb['samples']],
    ['linP_DL2_star', 'linP_n_star'],
    # legend_labels=[r'Planck+18 $\Lambda$CDM']
)



ax = g.subplots[1, 0]
ax_NS = g.subplots[1, 1]
ax_D2S = g.subplots[0, 0]
x_D2S = np.linspace(0.2, 0.55, 500)
x_NS = np.linspace(-2.4, -2.2, 500)

for inum, num in enumerate([0.68, 0.95]):
    if inum == 0:
        label="This work"
    else:
        label=None
    for jj in range(len(dat_mpg[num])):
        x = dat_mpg[num][jj][0] - true_cosmo["Delta2_star"]
        y = dat_mpg[num][jj][1] - true_cosmo["n_star"]
        ax.plot(x, y, color=cmap(col[inum]), label=label, lw=lw[inum], alpha=0.75)
        ax.fill(x, y, color=cmap(col[inum]), alpha=0.5)

# ax_D2S.plot(hist_d2s_x- true_cosmo["Delta2_star"], hist_d2s_y, color=cmap(col[0]))
# ax_NS.plot(hist_ns_x- true_cosmo["n_star"], hist_ns_y, color=cmap(col[0]))

ax_D2S.plot(x_d2s- true_cosmo["Delta2_star"], y_d2s, color=cmap(col[0]))
ax_NS.plot(x_ns- true_cosmo["n_star"], y_ns, color=cmap(col[0]))


CS = ax.contour(DL2_grid, neff_grid,chi2_Mc2005,levels=thresholds,colors='C1')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, xcen, xerr)
ax_D2S.plot(x_D2S, pdf/pdf.max(),color='C1')
pdf = stats.norm.pdf(x_NS, ycen, yerr)
ax_NS.plot(x_NS, pdf/pdf.max(),color='C1')

CS = ax.contour(DL2_grid, neff_grid,chi2_PD2015,levels=thresholds,colors='C2')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, xcen, xerr)
ax_D2S.plot(x_D2S, pdf/pdf.max(),color='C2')
pdf = stats.norm.pdf(x_NS, ycen, yerr)
ax_NS.plot(x_NS, pdf/pdf.max(),color='C2')

CS = ax.contour(DL2_grid, neff_grid,chi2_Ch2019,levels=thresholds,colors='C3')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, xcen, xerr)
ax_D2S.plot(x_D2S, pdf/pdf.max(),color='C3')
pdf = stats.norm.pdf(x_NS, ycen, yerr)
ax_NS.plot(x_NS, pdf/pdf.max(),color='C3')

CS = ax.contour(DL2_grid, neff_grid,chi2_Wa2024,levels=thresholds,colors='C4')
xcen, ycen, xerr, yerr = print_err(CS)
pdf = stats.norm.pdf(x_D2S, xcen, xerr)
ax_D2S.plot(x_D2S, pdf/pdf.max(),color='C4')
pdf = stats.norm.pdf(x_NS, ycen, yerr)
ax_NS.plot(x_NS, pdf/pdf.max(),color='C4')

# CS = ax.contour(DL2_grid, neff_grid,chi2_Wa2024_np,levels=thresholds,colors='C5')
# xcen, ycen, xerr, yerr = print_err(CS)
# pdf = stats.norm.pdf(x_D2S, xcen, xerr)
# ax_D2S.plot(x_D2S, pdf/pdf.max(),color='C5')
# pdf = stats.norm.pdf(x_NS, ycen, yerr)
# ax_NS.plot(x_NS, pdf/pdf.max(),color='C5')


# CS = ax.contour(neff_grid,DL2_grid,chi2_desi,levels=thresholds[:2],colors='C4')
# xcen, ycen, xerr, yerr = print_err(CS)
# pdf = stats.norm.pdf(x_D2S, ycen, yerr)
# ax_D2S.plot(x_D2S, pdf/pdf.max())
# pdf = stats.norm.pdf(x_NS, xcen, xerr)
# ax_NS.plot(x_NS, pdf/pdf.max())

# plt.axhline(y=1,color='C0',label=cmb['label'])
for ii in range(len(labs)):
    ax.axhline(y=1,color='C'+str(ii),label=labs[ii])

ax.set_ylim(-2.39, -2.24)
ax.set_xlim(0.23, 0.52)

# plt.title(r'Linear power constraints at ($z=3$, $k_p=0.009$ s/km)')
# plt.grid()  

# ax.legend()

 # $\Lambda$CDM
handles = []
handles.append(mlines.Line2D([], [], color='black', label=r'Planck+18', lw=2))
for ii in range(len(labs)):
    handles.append(mlines.Line2D([], [], color='C'+str(ii), label=labs[ii], lw=2))
    
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


ax.set_xlabel(r"$\Delta^2_\star$", fontsize=fontsize)
ax.set_ylabel(r"$n_\star$", fontsize=fontsize)
ax_NS.set_xlabel(r"$n_\star$", fontsize=fontsize)


plt.tight_layout()
plt.savefig("figs/star_literature.png")
plt.savefig("figs/star_literature.pdf", bbox_inches="tight")

# %%

# %%

# CS = plt.contour(DL2_grid, neff_grid, chi2_Ch2019, levels=thresholds[:2],colors='C0')
# print_err(CS)

# # plt.ylim(-2.41, -2.29)
# # plt.ylim(-2.4, -2.25)
# plt.ylim(-2.35, -2.33)
# # plt.xlim(0.14, 0.44)
# # plt.xlim(0.20, 0.40)
# plt.xlim(0.28, 0.36)
# plt.grid()

# %%
